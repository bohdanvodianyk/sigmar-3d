"""Standalone API for SIGMAR-3D: register one point cloud onto another.

This is a thin wrapper around the frozen core (`sigmar3d.core`, the verbatim published
implementation). It takes plain `(N, 3)` numpy arrays and returns a `RegistrationResult`,
with no dependency on any larger framework. The algorithm and its parameters are NOT
modified here; this only reproduces the core's per-pair control flow (coarse SIGMA+RANSAC
-> difficulty reinforcement -> multi-scale GICP with monotone trimmed-RMSE acceptance)
without the manifest/PLY file I/O of `core.main`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

__all__ = ["RegistrationResult", "register", "register_scale_normalized"]


@dataclass
class RegistrationResult:
    """Outcome of registering a source cloud onto a target cloud."""

    source_to_target: np.ndarray            # (4, 4) rigid transform mapping source -> target
    success: bool = False
    num_correspondences: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    trimmed_rmse: Optional[float] = None
    weighted_trimmed_rmse: Optional[float] = None
    overlap: Optional[float] = None
    runtime_s: Optional[float] = None
    coarse_metrics: Dict = field(default_factory=dict)
    refined_metrics: Dict = field(default_factory=dict)
    failure_reason: Optional[str] = None


def _to_o3d(points: np.ndarray):
    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return pc


def register(
    source: np.ndarray,
    target: np.ndarray,
    initial_transform: Optional[np.ndarray] = None,
    use_coarse: bool = True,
    enable_multistart: bool = True,
    seed: int = 2025,
) -> RegistrationResult:
    """Register `source` onto `target`; returns the source->target transform + diagnostics.

    `source` and `target` are `(N, 3)` arrays of XYZ coordinates in metres. Normals are
    estimated internally by the core. With `initial_transform=None` the method starts from
    identity (the paper's protocol: no external pose prior).
    """
    import open3d as o3d

    from . import core

    src = _to_o3d(source)
    tgt = _to_o3d(target)
    if src.is_empty() or tgt.is_empty():
        return RegistrationResult(
            source_to_target=np.eye(4), success=False,
            failure_reason="empty source or target cloud",
        )

    T_init = np.eye(4) if initial_transform is None else np.asarray(initial_transform, float)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    tune = core.SelfTune(src, tgt, T_init)
    aligner = core.AdaptiveAligner()

    src_init = o3d.geometry.PointCloud(src)
    src_init.transform(T_init)

    # --- coarse: SIGMA + RANSAC ---
    T_ransac_delta = np.eye(4)
    ransac_corr = np.zeros((0, 2), dtype=int)
    n_inliers = 0
    if use_coarse and core.USE_COARSE_RANSAC:
        T_ransac_delta, ransac_corr, n_inliers = aligner.coarse_ransac(src_init, tgt, tune)
    T_before_icp = T_ransac_delta @ T_init

    # difficulty reinforcement (identical to core.main)
    if (ransac_corr.shape[0] < 300) or (tune.overlap < 0.25):
        tune.very_hard = True
        tune.icp_base = core.ICP_BASE_ITERS_HARD
        tune.icp_dist_factor = max(tune.icp_dist_factor, 0.60)

    # --- refine: monotone multi-scale (G)ICP (+ multistart if very hard) ---
    if enable_multistart and core.ENABLE_MULTISTART and tune.very_hard:
        seeds = [T_before_icp]
        for _ in range(core.MULTISTART_SEEDS - 1):
            seeds.append(core.jitter_pose(T_before_icp, rot_deg=2.5, trans_m=0.05, rng=rng))
        vox0 = tune.voxel0
        s0 = src.voxel_down_sample(vox0)
        t0p = tgt.voxel_down_sample(vox0)
        best_T, best_trim, best_log = None, np.inf, None
        for T_seed in seeds:
            T_try, log_try = aligner.refine_icp(src, tgt, T_seed, tune)
            trim_try = aligner._trimmed_rmse_after(
                T_try, s0, t0p, tune.icp_dist_factor * vox0,
                core.get_trim_frac_for_level(0, tune.levels),
            )
            if trim_try < best_trim:
                best_T, best_trim, best_log = T_try, trim_try, log_try
        T_final, accept_log = best_T, best_log
    else:
        T_final, accept_log = aligner.refine_icp(src, tgt, T_before_icp, tune)

    # Robustness guard: on very hard pairs every refinement seed can yield an infinite
    # trimmed error, leaving no accepted update -> fall back to the coarse estimate rather
    # than returning None (the original loop leaves best_T=None in that case).
    if T_final is None:
        T_final, accept_log = T_before_icp, (accept_log or [])

    runtime = time.time() - t0

    # --- diagnostics ---
    pyr = tune.voxel_pyramid()
    vfin = pyr[-1]
    sfin = src.voxel_down_sample(vfin)
    tfin = tgt.voxel_down_sample(vfin)
    trimmed = aligner._trimmed_rmse_after(
        T_final, sfin, tfin, tune.icp_dist_factor * vfin, 0.8
    )
    n_corr = int(ransac_corr.shape[0])
    inlier_ratio = (n_inliers / n_corr) if n_corr > 0 else 0.0
    success = bool(np.all(np.isfinite(T_final)) and (not use_coarse or n_corr >= 4))

    return RegistrationResult(
        source_to_target=np.asarray(T_final, dtype=float),
        success=success,
        num_correspondences=n_corr,
        num_inliers=int(n_inliers),
        inlier_ratio=float(inlier_ratio),
        trimmed_rmse=float(trimmed) if np.isfinite(trimmed) else None,
        overlap=float(tune.overlap),
        runtime_s=runtime,
        coarse_metrics={
            "very_hard": bool(tune.very_hard),
            "spacing_median": float(tune.spacing),
            "ransac_iters": int(tune.ransac_iters),
        },
        refined_metrics={"accept_log": accept_log, "voxel_finest": float(vfin)},
        failure_reason=None if success else "insufficient correspondences",
    )


def register_scale_normalized(
    source: np.ndarray,
    target: np.ndarray,
    initial_transform: Optional[np.ndarray] = None,
    target_diag_m: float = 4.0,
    **kwargs,
) -> RegistrationResult:
    """Register at a normalised scene scale, for objects far from the method's room-scale regime.

    WHY: the core's `SelfTune` carries a few absolute metric floors tuned for room-scale
    scenes (normal radii, pyramid voxel clips, overlap gate). On a small object (e.g. a
    ~17 cm scan) those floors dominate and cripple the descriptor. Rigid registration is
    scale-equivariant, so scaling both clouds by `s`, registering, and dividing the recovered
    translation by `s` is mathematically identical to making those constants scale-adaptive,
    with no change to the published algorithm. Leave this alone for room/outdoor scenes.
    """
    src = np.asarray(source, float)
    tgt = np.asarray(target, float)
    allp = np.vstack([src, tgt])
    diag = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    s = (target_diag_m / diag) if diag > 1e-9 else 1.0

    T_init = None
    if initial_transform is not None:
        T_init = np.asarray(initial_transform, float).copy()
        T_init[:3, 3] *= s

    res = register(src * s, tgt * s, T_init, **kwargs)

    T = np.asarray(res.source_to_target, float).copy()
    T[:3, 3] /= s                      # unscale translation; rotation is scale-invariant
    res.source_to_target = T
    for attr in ("trimmed_rmse", "weighted_trimmed_rmse"):
        v = getattr(res, attr, None)
        if v is not None and np.isfinite(v):
            setattr(res, attr, v / s)
    res.refined_metrics = {**(res.refined_metrics or {}),
                           "scale_normalized": True, "scale_factor": s,
                           "target_diag_m": target_diag_m, "native_diag_m": diag}
    return res
