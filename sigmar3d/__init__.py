"""SIGMAR-3D: Scale-Invariant Guaranteed Monotone Accurate Registration for 3D scenes.

A training-free pipeline for robust pairwise point-cloud registration: the SIGMA local
descriptor (LRF + spherical binning + normal-consistency accumulation) feeding the R-3D
coarse-to-fine registration stage (ratio-test matching, RANSAC, multi-scale GICP with a
monotone trimmed-RMSE acceptance rule).

Public API
----------
    register(source, target, ...) -> RegistrationResult
    register_scale_normalized(source, target, ...) -> RegistrationResult   # small objects
    synthesize_scene(...) / synthesize_pair(...)                           # data-free demo

`sigmar3d.core` holds the verbatim published implementation and is used unmodified.
"""
from .api import RegistrationResult, register, register_scale_normalized
from .synthetic import (
    make_ground_truth_transform,
    rot_axis_angle,
    synthesize_pair,
    synthesize_scene,
)

__version__ = "1.0.0"

__all__ = [
    "register",
    "register_scale_normalized",
    "RegistrationResult",
    "synthesize_scene",
    "synthesize_pair",
    "rot_axis_angle",
    "make_ground_truth_transform",
    "__version__",
]
