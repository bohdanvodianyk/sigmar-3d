#!/usr/bin/env python
"""Render a before/after registration figure from the synthetic demo (saved to assets/).

Produces `assets/demo_registration.png`: the left panel shows the source and target clouds
at the identity initialisation (misaligned); the right panel shows the source after applying
the transform recovered by SIGMAR-3D (aligned onto the target). Requires matplotlib:

    pip install -e ".[viz]"
    python examples/plot_registration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sigmar3d import register, synthesize_pair, synthesize_scene  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE_C = "#2563eb"   # blue
TARGET_C = "#f97316"   # orange


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def _subsample(pts: np.ndarray, n: int, rng) -> np.ndarray:
    if len(pts) <= n:
        return pts
    return pts[rng.choice(len(pts), n, replace=False)]


def _panel(ax, src, tgt, title):
    ax.scatter(*src.T, s=2, c=SOURCE_C, alpha=0.55, label="source", linewidths=0)
    ax.scatter(*tgt.T, s=2, c=TARGET_C, alpha=0.55, label="target", linewidths=0)
    ax.set_title(title, fontsize=13, pad=0)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=28, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, markerscale=3)


def _pose_error(T_est, T_gt):
    R_err = T_est[:3, :3].T @ T_gt[:3, :3]
    rre = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
    rte = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    return rre, rte


def main() -> None:
    rng = np.random.default_rng(0)
    scene = synthesize_scene(n_points=12000, seed=1)
    src, tgt, T_gt = synthesize_pair(scene, rotation_deg=20.0, overlap=0.65, noise_mm=2.0, seed=1)
    result = register(src, tgt)
    rre, rte = _pose_error(result.source_to_target, T_gt)

    src_p = _subsample(src, 4000, rng)
    tgt_p = _subsample(tgt, 4000, rng)
    src_aligned = _apply(result.source_to_target, src_p)

    fig = plt.figure(figsize=(11, 4.6), dpi=140)
    fig.suptitle("SIGMAR-3D: training-free pairwise registration",
                 fontsize=15, fontweight="bold", y=0.99)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _panel(ax1, src_p, tgt_p, "Unregistered (identity init)")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _panel(ax2, src_aligned, tgt_p,
           f"Aligned by SIGMAR-3D  (RRE {rre:.2f}°, RTE {rte*100:.2f} cm)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = ROOT / "assets" / "demo_registration.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  (RRE {rre:.2f} deg, RTE {rte*100:.2f} cm)")


if __name__ == "__main__":
    main()
