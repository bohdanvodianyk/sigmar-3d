"""Smoke tests: the synthetic generator is correct, and SIGMAR-3D recovers a known pose.

These are the repository's end-to-end sanity checks. They need no external data: a
room-scale scene is synthesised in memory, transformed by a known rigid pose with
partial overlap and noise, and SIGMAR-3D must recover that pose from identity init.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from sigmar3d import synthesize_scene, synthesize_pair, register


def _pose_error(T_est: np.ndarray, T_gt: np.ndarray):
    """Rotation error in degrees and translation error in metres, independent of the core."""
    R_err = T_est[:3, :3].T @ T_gt[:3, :3]
    cos = (np.trace(R_err) - 1.0) / 2.0
    rre_deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    rte_m = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    return rre_deg, rte_m


def test_synthesize_pair_applies_known_transform():
    """target is T_gt applied to a partial-overlap subset of source (within noise)."""
    source = synthesize_scene(n_points=8000, seed=0)
    src, tgt, T_gt = synthesize_pair(source, rotation_deg=25.0, overlap=0.6,
                                     noise_mm=1.0, seed=0)

    # overlap cut keeps roughly the requested fraction
    assert 0.4 * len(src) < len(tgt) < 0.8 * len(src)

    # inverse-transforming the target lands every point back on the source surface
    T_inv = np.linalg.inv(T_gt)
    back = (T_inv[:3, :3] @ tgt.T).T + T_inv[:3, 3]
    d, _ = cKDTree(src).query(back)
    assert d.max() < 0.01  # 1 cm: 1 mm noise + discretisation, nowhere near the 25 deg motion


def test_register_recovers_known_transform():
    """SIGMAR-3D recovers a 20 deg / partial-overlap pose from identity initialisation."""
    source = synthesize_scene(n_points=12000, seed=1)
    src, tgt, T_gt = synthesize_pair(source, rotation_deg=20.0, overlap=0.65,
                                     noise_mm=2.0, seed=1)

    result = register(src, tgt, seed=2025)

    assert result.success
    rre_deg, rte_m = _pose_error(result.source_to_target, T_gt)
    # identity init would give ~20 deg; a real recovery is a few degrees / centimetres
    assert rre_deg < 5.0, f"rotation error too large: {rre_deg:.2f} deg"
    assert rte_m < 0.10, f"translation error too large: {rte_m*100:.1f} cm"
