#!/usr/bin/env python
"""Run SIGMAR-3D on a synthetic, data-free registration pair and report the pose error.

A room-scale scene is generated in memory, transformed by a known rigid pose with partial
overlap and noise, and SIGMAR-3D recovers that pose from identity initialisation. Because
the ground-truth transform is known, the rotation (RRE) and translation (RTE) errors are
measured exactly.

    python examples/demo_register.py
    python examples/demo_register.py --rotation-deg 35 --overlap 0.5 --noise-mm 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from a fresh checkout without `pip install` (add the repo root to sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigmar3d import register, synthesize_pair, synthesize_scene  # noqa: E402


def pose_error(T_est: np.ndarray, T_gt: np.ndarray):
    """Rotation error (degrees) and translation error (metres)."""
    R_err = T_est[:3, :3].T @ T_gt[:3, :3]
    cos = (np.trace(R_err) - 1.0) / 2.0
    rre_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    rte_m = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    return rre_deg, rte_m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-points", type=int, default=12000, help="points in the synthetic scene")
    p.add_argument("--rotation-deg", type=float, default=20.0, help="ground-truth rotation magnitude")
    p.add_argument("--overlap", type=float, default=0.65, help="fraction of the scene kept in the second view")
    p.add_argument("--noise-mm", type=float, default=2.0, help="Gaussian point noise (mm)")
    p.add_argument("--seed", type=int, default=1, help="scene/pair seed")
    args = p.parse_args()

    print("Synthesising a room-scale pair "
          f"(rotation {args.rotation_deg:g} deg, overlap {args.overlap:g}, noise {args.noise_mm:g} mm)...")
    scene = synthesize_scene(n_points=args.n_points, seed=args.seed)
    src, tgt, T_gt = synthesize_pair(scene, rotation_deg=args.rotation_deg,
                                     overlap=args.overlap, noise_mm=args.noise_mm,
                                     seed=args.seed)
    print(f"  source: {len(src):,} points   target: {len(tgt):,} points\n")

    print("Running SIGMAR-3D (identity initialisation, no external prior)...")
    result = register(src, tgt)

    rre_deg, rte_m = pose_error(result.source_to_target, T_gt)
    print("\n=== Result ===")
    print(f"  success            : {result.success}")
    print(f"  rotation error RRE : {rre_deg:6.2f} deg")
    print(f"  translation err RTE: {rte_m * 100:6.2f} cm")
    print(f"  correspondences    : {result.num_correspondences}  (inliers {result.num_inliers})")
    print(f"  trimmed RMSE       : {result.trimmed_rmse}")
    print(f"  runtime            : {result.runtime_s:.2f} s")
    print("\nRecovered source -> target transform:")
    with np.printoptions(precision=4, suppress=True):
        print(result.source_to_target)


if __name__ == "__main__":
    main()
