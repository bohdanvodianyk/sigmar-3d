"""Data-free synthetic registration pairs for the demo and smoke tests.

SIGMAR-3D is evaluated in the paper on 3DMatch (indoor RGB-D) and ETH (outdoor LiDAR).
Those datasets are large and governed by their own terms, so they are not bundled here.
Instead this module procedurally builds a *room-scale* scene (a ground plane with several
boxes) and a controlled-difficulty pair from it: one view is the scene, the other is a
known rigid transform of a partially overlapping, noisy subset. Because the transform is
known, registration error (RRE/RTE) is measurable rather than inferred.

Room scale (~4 m) is deliberate: the published implementation carries a few fixed absolute
safeguards (clipping bounds, a coarse nearest-neighbour threshold) tuned for room/outdoor
scenes, so a metre-scale scene exercises the method in its design regime without needing
the scale-normalising wrapper.

Pure numpy; no third-party point-cloud dependency.
"""
from __future__ import annotations

import numpy as np

__all__ = ["synthesize_scene", "synthesize_pair", "rot_axis_angle",
           "make_ground_truth_transform"]


def _sample_rectangle(origin, e1, e2, n, rng):
    """`n` points sampled uniformly on the rectangle spanned by `e1`, `e2` from `origin`."""
    a = rng.random((n, 1))
    b = rng.random((n, 1))
    return origin + a * e1 + b * e2


def _box_faces(center, size):
    """The six faces of an axis-aligned box as `(origin, e1, e2, area)` tuples."""
    c = np.asarray(center, float)
    s = np.asarray(size, float)
    o = c - s / 2.0
    ex = np.array([s[0], 0.0, 0.0])
    ey = np.array([0.0, s[1], 0.0])
    ez = np.array([0.0, 0.0, s[2]])
    faces = [
        (o, ex, ey),            # bottom
        (o + ez, ex, ey),       # top
        (o, ex, ez),            # near wall
        (o + ey, ex, ez),       # far wall
        (o, ey, ez),            # left wall
        (o + ex, ey, ez),       # right wall
    ]
    return [(org, e1, e2, float(np.linalg.norm(np.cross(e1, e2)))) for org, e1, e2 in faces]


def synthesize_scene(n_points: int = 12000, seed: int = 0) -> np.ndarray:
    """A deterministic room-scale point cloud: a 4x4 m floor with several boxes.

    The boxes sit at an asymmetric, fixed layout so the scene has distinctive corners and
    edges (good for descriptor matching) rather than a symmetry that would make the pose
    ambiguous. Returns an `(N, 3)` float array in metres; `N` is approximately `n_points`.
    """
    rng = np.random.default_rng(seed)

    # Floor is one big face; boxes (footprint centre x,y and size) rest on it (z = sz/2).
    floor = (np.array([0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0]),
             np.array([0.0, 4.0, 0.0]), 16.0)
    box_specs = [
        ((1.0, 1.0), (0.8, 0.6, 0.8)),
        ((2.8, 1.2), (0.5, 0.9, 0.6)),
        ((1.5, 2.8), (1.0, 0.5, 1.0)),
        ((3.0, 3.0), (0.7, 0.7, 0.7)),
        ((0.7, 3.2), (0.5, 0.6, 0.5)),
    ]
    faces = [floor]
    for (fx, fy), (sx, sy, sz) in box_specs:
        faces.extend(_box_faces((fx, fy, sz / 2.0), (sx, sy, sz)))

    total_area = sum(area for *_, area in faces)
    clouds = []
    for origin, e1, e2, area in faces:
        n_face = max(1, int(round(n_points * area / total_area)))
        clouds.append(_sample_rectangle(origin, e1, e2, n_face, rng))
    return np.vstack(clouds)


def rot_axis_angle(axis, deg: float) -> np.ndarray:
    """Rotation matrix of `deg` degrees about `axis` (Rodrigues' formula)."""
    ax = np.asarray(axis, float)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    a = np.radians(deg)
    K = np.array([[0.0, -ax[2], ax[1]],
                  [ax[2], 0.0, -ax[0]],
                  [-ax[1], ax[0], 0.0]])
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def make_ground_truth_transform(rotation_deg: float, seed: int = 0,
                                translation_m: float = 0.1) -> np.ndarray:
    """A random rigid transform with a prescribed rotation magnitude (random axis,
    translation drawn uniformly from +/- `translation_m` per axis)."""
    rng = np.random.default_rng(seed)
    T = np.eye(4)
    T[:3, :3] = rot_axis_angle(rng.normal(size=3), rotation_deg)
    T[:3, 3] = rng.uniform(-translation_m, translation_m, 3)
    return T


def synthesize_pair(source: np.ndarray, rotation_deg: float = 25.0,
                    overlap: float = 0.6, noise_mm: float = 2.0, seed: int = 0):
    """Build a controlled-difficulty `(source, target, T_gt)` from a source cloud.

    The target is a partial, transformed, noisy view: a half-space cut along a random
    direction keeps a fraction `overlap` of the points (mimicking a viewpoint change that
    hides part of the surface), which is then mapped by a known transform `T_gt` and
    perturbed by Gaussian noise of standard deviation `noise_mm`.
    """
    pts = np.asarray(source, float)
    T_gt = make_ground_truth_transform(rotation_deg, seed)
    rng = np.random.default_rng(97 * seed + int(overlap * 100))

    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    proj = pts @ axis
    keep = proj >= np.quantile(proj, 1.0 - overlap)

    kept = pts[keep]
    noise = rng.normal(0.0, noise_mm / 1000.0, kept.shape)
    target = (T_gt[:3, :3] @ kept.T).T + T_gt[:3, 3] + noise
    return pts, target, T_gt
