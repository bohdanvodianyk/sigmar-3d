# run_mymethod_adaptive_icp_hard.py — SIGMAR-3D (very hard cases)
# - Reads BENCH_DIR/pairs_manifest.json (with T_init & T_gt)
# - Coarse: SIGMA descriptor + ANN mutual-NN ratio-test matching + RANSAC
#           with certified iterations, edge-length & distance gates, SVD refinement
# - Refine: multi-start, multi-scale (G)ICP with monotone trimmed-RMSE acceptance
#           and single gate-tightening retry per level (β factor)
# - Certificates logged per pair

from pathlib import Path
import json
import time
import math
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# ---------- PATHS ----------
SAVE_ALIGNED_PLY = True
# ---------------------------

# ---------- SWITCHES ----------
USE_COARSE_RANSAC   = True     # False -> skip coarse; rely on T_init
USE_ROBUST_LOSS     = True     # Tukey loss in ICP/GICP
PREFER_GICP         = True     # use GICP if available
LOG_CERTS           = True     # write per-pair "certificate" info
ENABLE_MULTISTART   = True     # extra restarts for very hard pairs
MULTISTART_SEEDS    = 10
# -----------------------------

# ---------- SIGMA DESCRIPTOR DEFAULTS ----------
SIGMA_B_R     = 3       # radial shells
SIGMA_B_THETA = 8       # azimuth sectors
SIGMA_B_PHI   = 4       # elevation slices
SIGMA_KAPPA   = 8.0     # support factor: r(p) = kappa * delta_tilde(p)
SIGMA_K_DELTA = 10      # neighbors for local spacing estimate
# ------------------------------------------------

# ---------- MATCHING DEFAULTS ----------
MATCH_RATIO_THRESH = 0.85   # Lowe ratio test threshold (rho)
# ----------------------------------------

# ---------- ADAPTIVE DEFAULTS ----------
MAX_SAMPLE_FOR_SPACING = 10000
PYR_MAX_LEVELS   = 8
ICP_BASE_ITERS_EASY = 300
ICP_BASE_ITERS_HARD = 1500
NN_MAX_NORMAL    = 200
# Trim schedule (coarse -> fine), extended/interpolated to match pyramid depth
TRIM_SCHEDULE_BASE = [0.60, 0.70, 0.80]
# Gate tightening factor (beta) for ICP retry
BETA_GATE_TIGHTEN = 0.80
# RANSAC
RANSAC_ALPHA = 3.0          # inlier threshold: epsilon = alpha * delta_tilde_global
RANSAC_MIN_CORR_FOR_STRICT_EDGE = 300  # below this, relax edge-length to 0.80
# ----------------------------------------

# ---------- PAPER-STYLE METRICS ----------
INLIER_DIST_THRESH = 0.10   # τ for FMR (m)
FMR_RECALL_THRESH  = 0.05   # γ
POSE_RRE_OK_DEG    = 2.0
POSE_RTE_OK_M      = 0.05
# ----------------------------------------

# ===== Utils =====
def to_mat4(lst): return np.array(lst, dtype=float).reshape(4, 4)

def transform_points(pts: np.ndarray, T: np.ndarray):
    homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    return (homo @ T.T)[:, :3]

def rotation_angle_deg_from_R(R):
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(tr))

def pose_errors(T_est: np.ndarray, T_gt: np.ndarray):
    Terr = T_est @ np.linalg.inv(T_gt)
    R_e = Terr[:3, :3]; t_e = Terr[:3, 3]
    return rotation_angle_deg_from_R(R_e), float(np.linalg.norm(t_e))

def kd_nn_distances(P: np.ndarray, Q_pcd: o3d.geometry.PointCloud):
    Q = np.asarray(Q_pcd.points)
    if Q.shape[0] == 0:
        return np.full(P.shape[0], np.inf)
    tree = cKDTree(Q)
    d, _ = tree.query(P, k=1)
    return d.astype(float)

def trimmed_rmse(d: np.ndarray, trim_frac: float):
    if d.size == 0 or not np.isfinite(d).any(): return np.inf
    d = d[np.isfinite(d)]
    keep = max(1, int(math.floor(trim_frac * d.size)))
    d_sorted = np.sort(d)[:keep]
    return float(np.sqrt(np.mean(d_sorted**2)))

def jitter_pose(T: np.ndarray, rot_deg=2.5, trans_m=0.05, rng=None):
    rng = rng or np.random.default_rng(123)
    ax = rng.normal(size=3); ax /= (np.linalg.norm(ax) + 1e-12)
    ang = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    K = np.array([[0, -ax[2], ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    Rj = np.eye(3) + math.sin(ang)*K + (1-math.cos(ang))*(K@K)
    tj = rng.uniform(-trans_m, trans_m, size=3)
    J = np.eye(4); J[:3,:3]=Rj; J[:3,3]=tj
    return J @ T

def svd_absolute_orientation(P: np.ndarray, Q: np.ndarray):
    """Closed-form rigid alignment of P -> Q via SVD (Umeyama).
    P, Q: (N, 3) matched point sets. Returns 4x4 transform."""
    assert P.shape == Q.shape and P.shape[0] >= 3
    cp = P.mean(axis=0); cq = Q.mean(axis=0)
    Pc = P - cp; Qc = Q - cq
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cq - R @ cp
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T

def get_trim_frac_for_level(lvl: int, n_levels: int):
    """Interpolate the trim schedule to the pyramid depth."""
    base = TRIM_SCHEDULE_BASE
    if n_levels <= len(base):
        return base[min(lvl, len(base)-1)]
    # Linearly interpolate schedule across levels
    frac_pos = lvl / max(1, n_levels - 1)  # 0..1
    idx_f = frac_pos * (len(base) - 1)
    lo = int(math.floor(idx_f))
    hi = min(lo + 1, len(base) - 1)
    alpha = idx_f - lo
    return base[lo] * (1 - alpha) + base[hi] * alpha


# ================================================================
# SIGMA descriptor — Sec 3.1 of the paper
# ================================================================
class SIGMADescriptor:
    """Training-free local descriptor with LRF canonicalization,
    spherical binning, and normal-consistency accumulation."""

    def __init__(self, B_r=SIGMA_B_R, B_theta=SIGMA_B_THETA, B_phi=SIGMA_B_PHI,
                 kappa=SIGMA_KAPPA, k_delta=SIGMA_K_DELTA):
        self.B_r = B_r
        self.B_theta = B_theta
        self.B_phi = B_phi
        self.B_total = B_r * B_theta * B_phi
        self.kappa = kappa
        self.k_delta = k_delta

    def compute_local_spacing(self, pts: np.ndarray, tree: cKDTree, indices: np.ndarray):
        """Compute median distance to k_delta nearest neighbors per keypoint."""
        k = min(self.k_delta + 1, pts.shape[0])  # +1 because query point is included
        d, _ = tree.query(pts[indices], k=k)
        # d[:,0] is self-distance (~0); use d[:,1:]
        if k > 1:
            return np.median(d[:, 1:], axis=1)
        else:
            return np.full(len(indices), 0.05)

    def _build_lrf(self, x_p, n_p, neighbors_xyz, neighbors_normals):
        """Build local reference frame at keypoint p.
        Returns E = [e2 | e3 | e1] (3x3), where e1 = normal."""
        e1 = n_p / (np.linalg.norm(n_p) + 1e-15)

        # Projector onto plane perpendicular to e1
        # Pi = I - e1 e1^T
        def project_to_tangent(v):
            return v - np.dot(v, e1) * e1

        # Try nearest-neighbor anchor (Eq. 1)
        if neighbors_xyz.shape[0] > 0:
            # Find nearest neighbor (excluding self — neighbors already exclude self)
            dists = np.linalg.norm(neighbors_xyz - x_p, axis=1)
            nn_idx = np.argmin(dists)
            offset = neighbors_xyz[nn_idx] - x_p
            e2_tilde = project_to_tangent(offset)
            e2_norm = np.linalg.norm(e2_tilde)

            if e2_norm > 1e-10:
                e2 = e2_tilde / e2_norm
            else:
                # Degenerate: use covariance fallback (Eq. 2)
                e2 = self._covariance_fallback(x_p, e1, neighbors_xyz, project_to_tangent)
        else:
            e2 = self._covariance_fallback(x_p, e1, neighbors_xyz, project_to_tangent)

        e3 = np.cross(e1, e2)
        e3_norm = np.linalg.norm(e3)
        if e3_norm > 1e-15:
            e3 = e3 / e3_norm
        else:
            # Fallback: pick arbitrary orthogonal
            e2, e3 = self._arbitrary_orthogonal(e1)

        # E = [e2 | e3 | e1] so that (u,v) = tangent plane, w = normal
        E = np.column_stack([e2, e3, e1])
        return E

    def _covariance_fallback(self, x_p, e1, neighbors_xyz, project_fn):
        """Estimate tangent direction from local covariance (Eq. 2)."""
        if neighbors_xyz.shape[0] < 2:
            e2, _ = self._arbitrary_orthogonal(e1)
            return e2
        offsets = neighbors_xyz - x_p
        projected = np.array([project_fn(o) for o in offsets])
        Sigma = (projected.T @ projected) / len(projected)
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        if eigvals[-1] > 1e-12:
            e2 = eigvecs[:, -1]  # principal eigenvector
            e2 = e2 / (np.linalg.norm(e2) + 1e-15)
            return e2
        else:
            e2, _ = self._arbitrary_orthogonal(e1)
            return e2

    def _arbitrary_orthogonal(self, e1):
        """Pick an arbitrary unit vector perpendicular to e1."""
        if abs(e1[0]) < 0.9:
            v = np.array([1.0, 0.0, 0.0])
        else:
            v = np.array([0.0, 1.0, 0.0])
        e2 = v - np.dot(v, e1) * e1
        e2 = e2 / (np.linalg.norm(e2) + 1e-15)
        e3 = np.cross(e1, e2)
        return e2, e3

    def compute_descriptors(self, pcd: o3d.geometry.PointCloud, keypoint_indices: np.ndarray):
        """Compute SIGMA descriptors for selected keypoints.

        Args:
            pcd: point cloud with oriented normals
            keypoint_indices: indices into pcd.points for which to compute descriptors

        Returns:
            descriptors: (n_keypoints, B_total) L2-normalized descriptor array
            local_spacings: (n_keypoints,) median local spacing per keypoint
        """
        pts = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        n_pts = pts.shape[0]
        n_kp = len(keypoint_indices)

        tree = cKDTree(pts)

        # Compute local spacing per keypoint (Eq. 7)
        local_spacings = self.compute_local_spacing(pts, tree, keypoint_indices)

        # Support radius per keypoint: r(p) = kappa * delta_tilde(p)
        support_radii = self.kappa * local_spacings

        # Bin edges (uniform spacing)
        # Radial bins: [0, r] in B_r shells
        # Azimuth: (-pi, pi] in B_theta sectors
        # Elevation: [-pi/2, pi/2] in B_phi slices

        descriptors = np.zeros((n_kp, self.B_total), dtype=np.float64)

        for ki in range(n_kp):
            p_idx = keypoint_indices[ki]
            x_p = pts[p_idx]
            n_p = normals[p_idx]
            r = support_radii[ki]

            if r < 1e-12:
                continue

            # Query neighborhood within support radius
            neighbor_idxs = tree.query_ball_point(x_p, r)
            # Remove self
            neighbor_idxs = [j for j in neighbor_idxs if j != p_idx]

            if len(neighbor_idxs) == 0:
                continue

            nb_xyz = pts[neighbor_idxs]
            nb_normals = normals[neighbor_idxs]

            # Build LRF (Eqs. 1-3)
            E = self._build_lrf(x_p, n_p, nb_xyz, nb_normals)

            # Transform neighbors to local coordinates (Eq. 3)
            offsets = nb_xyz - x_p
            local_coords = offsets @ E  # (N_nb, 3): columns are (u, v, w)

            u = local_coords[:, 0]
            v = local_coords[:, 1]
            w = local_coords[:, 2]

            # Spherical coordinates (Eq. 4)
            rho = np.sqrt(u**2 + v**2 + w**2)
            theta = np.arctan2(v, u)   # in (-pi, pi]
            phi = np.arctan2(w, np.sqrt(u**2 + v**2))  # in [-pi/2, pi/2]

            # Bin assignment
            # Radial: uniform in [0, r]
            r_bin = np.clip(np.floor(rho / r * self.B_r).astype(int), 0, self.B_r - 1)
            # Azimuth: uniform in (-pi, pi]
            theta_shifted = theta + np.pi  # now in [0, 2*pi)
            t_bin = np.clip(np.floor(theta_shifted / (2 * np.pi) * self.B_theta).astype(int),
                           0, self.B_theta - 1)
            # Elevation: uniform in [-pi/2, pi/2]
            phi_shifted = phi + np.pi / 2  # now in [0, pi]
            p_bin = np.clip(np.floor(phi_shifted / np.pi * self.B_phi).astype(int),
                           0, self.B_phi - 1)

            # Flat bin index
            flat_idx = r_bin * (self.B_theta * self.B_phi) + t_bin * self.B_phi + p_bin

            # Normal consistency (Eq. 5): phi(n_p, n_q) = max(0, n_p . n_q)
            n_p_unit = n_p / (np.linalg.norm(n_p) + 1e-15)
            dots = nb_normals @ n_p_unit
            weights = np.maximum(0.0, dots)

            # Accumulate into bins
            np.add.at(descriptors[ki], flat_idx, weights)

        # L2 normalize (Eq. 5)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        descriptors = descriptors / norms

        return descriptors, local_spacings


# ================================================================
# Feature matching — Sec 3.2 of the paper
# ================================================================
def match_descriptors(desc_src: np.ndarray, desc_tgt: np.ndarray, ratio_thresh: float = MATCH_RATIO_THRESH):
    """ANN matching with Lowe ratio test and mutual consistency.

    Returns:
        correspondences: (N, 2) array of (src_kp_idx, tgt_kp_idx) pairs
    """
    if desc_src.shape[0] == 0 or desc_tgt.shape[0] < 2:
        return np.zeros((0, 2), dtype=int)

    # Build KD-tree over target descriptors
    tree_tgt = cKDTree(desc_tgt)
    tree_src = cKDTree(desc_src)

    # Forward: for each source, find 2-NN in target
    d_fwd, idx_fwd = tree_tgt.query(desc_src, k=2)
    # Backward: for each target, find 1-NN in source (for mutual consistency)
    d_bwd, idx_bwd = tree_src.query(desc_tgt, k=1)

    correspondences = []
    for i in range(desc_src.shape[0]):
        d1, d2 = d_fwd[i, 0], d_fwd[i, 1]
        if d2 < 1e-12:
            continue
        # Ratio test (Eq. 8)
        if d1 / d2 < ratio_thresh:
            j = idx_fwd[i, 0]
            # Mutual consistency: target j's nearest source should be i
            if idx_bwd[j] == i:
                correspondences.append((i, j))

    if len(correspondences) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.array(correspondences, dtype=int)


# ================================================================
# RANSAC with SVD refinement — Sec 3.2
# ================================================================
def ransac_pose(pts_src: np.ndarray, pts_tgt: np.ndarray,
                correspondences: np.ndarray,
                epsilon: float,
                edge_len_thresh: float,
                max_iters: int = 500000,
                rng=None):
    """RANSAC with m=4 minimal set, edge-length consistency,
    and SVD absolute-orientation refinement on inliers.

    Args:
        pts_src, pts_tgt: (N, 3) full keypoint arrays
        correspondences: (M, 2) index pairs
        epsilon: inlier distance threshold (Eq. 10)
        edge_len_thresh: edge-length consistency ratio
        max_iters: iteration budget (Eq. 11)

    Returns:
        T_best: (4, 4) best rigid transform
        inlier_mask: boolean mask over correspondences
        n_inliers: count
    """
    rng = rng or np.random.default_rng(42)
    M = correspondences.shape[0]
    if M < 4:
        return np.eye(4), np.zeros(M, dtype=bool), 0

    src_corr = pts_src[correspondences[:, 0]]  # (M, 3)
    tgt_corr = pts_tgt[correspondences[:, 1]]  # (M, 3)

    best_inliers = 0
    best_mask = np.zeros(M, dtype=bool)
    best_T = np.eye(4)

    for it in range(max_iters):
        # Sample m=4 correspondences
        sample = rng.choice(M, size=4, replace=False)
        P_s = src_corr[sample]
        Q_s = tgt_corr[sample]

        # Edge-length consistency check
        valid = True
        for a in range(4):
            for b in range(a+1, 4):
                d_src = np.linalg.norm(P_s[a] - P_s[b])
                d_tgt = np.linalg.norm(Q_s[a] - Q_s[b])
                if d_src < 1e-12 or d_tgt < 1e-12:
                    valid = False
                    break
                ratio = min(d_src, d_tgt) / max(d_src, d_tgt)
                if ratio < edge_len_thresh:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Solve minimal rigid transform via SVD
        try:
            T_hyp = svd_absolute_orientation(P_s, Q_s)
        except Exception:
            continue

        R_h = T_hyp[:3, :3]; t_h = T_hyp[:3, 3]

        # Score: count inliers (Eq. 10)
        residuals = np.linalg.norm((R_h @ src_corr.T).T + t_h - tgt_corr, axis=1)
        mask = residuals < epsilon
        n_inl = mask.sum()

        if n_inl > best_inliers:
            best_inliers = n_inl
            best_mask = mask.copy()
            best_T = T_hyp.copy()

    # SVD refinement on all inliers of best hypothesis (paper: "refined in closed form")
    if best_inliers >= 4:
        try:
            best_T = svd_absolute_orientation(src_corr[best_mask], tgt_corr[best_mask])
        except Exception:
            pass

    return best_T, best_mask, best_inliers


# ===== Self-tuning policy & overlap =====
class SelfTune:
    def __init__(self, src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, T_init: np.ndarray):
        self.src = src
        self.tgt = tgt
        self.T_init = T_init
        self.rng = np.random.default_rng(42)
        self.spacing = self._median_spacing(src + tgt)
        self.diag = self._bbox_diag(src + tgt)
        self.overlap = self._quick_overlap(gate=0.8, max_pts=4000)
        self.very_hard = False
        self._derive()

    def _bbox_diag(self, pcd):
        bb = pcd.get_axis_aligned_bounding_box()
        return float(np.linalg.norm(bb.max_bound - bb.min_bound))

    def _median_spacing(self, pcd):
        pts = np.asarray(pcd.points); n = pts.shape[0]
        if n == 0: return 0.05
        k = min(MAX_SAMPLE_FOR_SPACING, n)
        idx = self.rng.choice(n, size=k, replace=False)
        tree = cKDTree(pts)
        d, _ = tree.query(pts[idx], k=2)
        valid = d[:, 1]
        return float(np.median(valid)) if len(valid) > 0 else 0.05

    def _quick_overlap(self, gate=0.8, max_pts=4000):
        s = self.src
        if len(s.points) > max_pts:
            idx = self.rng.choice(len(s.points), size=max_pts, replace=False)
            s = s.select_by_index(idx.tolist())
        P0 = transform_points(np.asarray(s.points), self.T_init)
        Q = np.asarray(self.tgt.points)
        if Q.shape[0] == 0:
            return 0.0
        tree = cKDTree(Q)
        d, _ = tree.query(P0, k=1)
        return float(np.mean(d < gate))

    def _derive(self):
        s = self.spacing; D = self.diag; ov = self.overlap
        self.very_hard = (ov < 0.25)

        # Normal estimation radius (for Open3D normal estimation)
        if self.very_hard:
            self.r_n = max(4.0 * s, 0.03)
        else:
            self.r_n = max(3.0 * s, 0.02)

        # Overlap proxy -> edge-length threshold (paper Sec 3.2)
        if ov >= 0.55:
            self.ransac_edge_min = 0.95
        elif ov >= 0.35:
            self.ransac_edge_min = 0.90
        else:
            self.ransac_edge_min = 0.85

        # RANSAC inlier threshold: epsilon = alpha * delta_tilde_global (Eq. 9)
        # delta_tilde_global is the global median spacing
        self.ransac_dist = RANSAC_ALPHA * s

        # Certified iteration budget (Eq. 11)
        p_prior = 0.35 if ov >= 0.35 else 0.20
        m = 4; eta = 0.999
        self.ransac_iters = int(np.clip(
            np.log(1 - eta) / np.log(max(1e-12, 1 - p_prior**m)),
            1e5, 1e6
        ))
        self.ransac_cert = {"p_prior": p_prior, "sample_m": m, "eta": eta, "N": self.ransac_iters}

        # Pyramid voxel schedule
        if self.very_hard:
            v0 = max(10.0 * s, 0.02 * D)
            self.voxel0 = float(np.clip(v0, 0.08, 0.35))
            self.pyr_step = 0.55
            self.levels = min(PYR_MAX_LEVELS, 7)
            self.icp_base = ICP_BASE_ITERS_HARD
            self.icp_dist_factor = 0.60
        else:
            v0 = max(8.0 * s, 0.02 * D)
            self.voxel0 = float(np.clip(v0, 0.06, 0.30))
            self.pyr_step = (0.65 if ov >= 0.55 else (0.60 if ov >= 0.35 else 0.55))
            L = int(np.ceil(np.log(self.voxel0 / max(0.75 * s, 1e-6)) / np.log(1.0 / self.pyr_step)))
            self.levels = int(np.clip(L, 3, PYR_MAX_LEVELS))
            self.icp_base = (ICP_BASE_ITERS_HARD if ov < 0.35 else ICP_BASE_ITERS_EASY)
            self.icp_dist_factor = 0.5 if s >= 0.05 else 0.4

    def voxel_pyramid(self):
        v = self.voxel0
        pyr = []
        for _ in range(self.levels):
            pyr.append(v)
            v *= self.pyr_step
        return pyr


# ===== Aligner =====
class AdaptiveAligner:
    def __init__(self):
        self.sigma = SIGMADescriptor()
        self.viewer_pos = None

    def _ensure_normals(self, pcd: o3d.geometry.PointCloud, radius: float):
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=NN_MAX_NORMAL))
        if self.viewer_pos is not None:
            pcd.orient_normals_towards_camera_location(self.viewer_pos)

    def _set_viewer(self, src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud):
        bb = (src + tgt).get_axis_aligned_bounding_box()
        c = (bb.max_bound + bb.min_bound) / 2.0
        size = bb.max_bound - bb.min_bound
        dist = float(np.linalg.norm(size)) * 1.5
        self.viewer_pos = c + np.array([0.0, -0.7 * dist, 0.7 * dist])

    def _select_keypoints(self, pcd: o3d.geometry.PointCloud, voxel_size: float):
        """Uniform subsampling for keypoint selection."""
        kp_pcd = pcd.voxel_down_sample(voxel_size)
        # Find indices of keypoints in original cloud
        pts_orig = np.asarray(pcd.points)
        pts_kp = np.asarray(kp_pcd.points)
        if pts_kp.shape[0] == 0:
            return np.array([], dtype=int), kp_pcd
        tree = cKDTree(pts_orig)
        _, indices = tree.query(pts_kp, k=1)
        return indices.astype(int), kp_pcd

    def coarse_ransac(self, src_init: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, tune: SelfTune):
        """SIGMA descriptor + ANN matching + RANSAC with SVD refinement."""
        self._set_viewer(src_init, tgt)

        # Ensure normals (oriented)
        self._ensure_normals(src_init, tune.r_n)
        self._ensure_normals(tgt, tune.r_n)

        # Keypoint selection (uniform subsampling)
        kp_voxel = max(3.0 * tune.spacing, 0.01)
        kp_idx_src, _ = self._select_keypoints(src_init, kp_voxel)
        kp_idx_tgt, _ = self._select_keypoints(tgt, kp_voxel)

        if len(kp_idx_src) < 4 or len(kp_idx_tgt) < 4:
            return np.eye(4), np.zeros((0, 2), dtype=int), 0

        # Compute SIGMA descriptors
        desc_src, spacings_src = self.sigma.compute_descriptors(src_init, kp_idx_src)
        desc_tgt, spacings_tgt = self.sigma.compute_descriptors(tgt, kp_idx_tgt)

        # Match descriptors (ANN + ratio test + mutual consistency)
        corr_local = match_descriptors(desc_src, desc_tgt, MATCH_RATIO_THRESH)

        if corr_local.shape[0] < 4:
            return np.eye(4), corr_local, 0

        # Overlap proxy (Eq. 9)
        n_kp_min = min(len(kp_idx_src), len(kp_idx_tgt))
        overlap_proxy = min(1.0, corr_local.shape[0] / max(1, n_kp_min))

        # Edge-length threshold from overlap proxy (paper Sec 3.2)
        if overlap_proxy >= 0.55:
            edge_thresh = 0.95
        elif overlap_proxy >= 0.35:
            edge_thresh = 0.90
        else:
            edge_thresh = 0.85

        # Relax if too few correspondences
        if corr_local.shape[0] < RANSAC_MIN_CORR_FOR_STRICT_EDGE:
            edge_thresh = min(edge_thresh, 0.80)

        # Map local correspondence indices to point-cloud positions
        pts_src_kp = np.asarray(src_init.points)[kp_idx_src]
        pts_tgt_kp = np.asarray(tgt.points)[kp_idx_tgt]

        # RANSAC with SVD refinement (Eqs. 10-11)
        T_best, inlier_mask, n_inliers = ransac_pose(
            pts_src_kp, pts_tgt_kp, corr_local,
            epsilon=tune.ransac_dist,
            edge_len_thresh=edge_thresh,
            max_iters=tune.ransac_iters
        )

        # Build full correspondences (in original point-cloud indices) for FMR eval
        full_corr = np.column_stack([
            kp_idx_src[corr_local[:, 0]],
            kp_idx_tgt[corr_local[:, 1]]
        ])

        return T_best, full_corr, n_inliers

    def _trimmed_rmse_after(self, T: np.ndarray, s: o3d.geometry.PointCloud,
                            t: o3d.geometry.PointCloud, dist_gate: float, trim_frac: float):
        P_T = transform_points(np.asarray(s.points), T)
        d = kd_nn_distances(P_T, t)
        d = d[d < dist_gate]
        return trimmed_rmse(d, trim_frac)

    def refine_icp(self, src_full: o3d.geometry.PointCloud, tgt_full: o3d.geometry.PointCloud,
                   T_init: np.ndarray, tune: SelfTune):
        """Multi-scale (G)ICP with monotone trimmed-RMSE acceptance
        and single gate-tightening retry (beta factor) per level."""
        self._set_viewer(src_full, tgt_full)
        pyr = tune.voxel_pyramid()
        T = T_init.copy()
        accept_log = []
        n_levels = len(pyr)

        for lvl, vox in enumerate(pyr):
            s = src_full.voxel_down_sample(vox)
            t = tgt_full.voxel_down_sample(vox)

            self._ensure_normals(s, radius=2.5 * vox)
            self._ensure_normals(t, radius=2.5 * vox)

            dist_gate = tune.icp_dist_factor * vox
            trim_here = get_trim_frac_for_level(lvl, n_levels)

            if USE_ROBUST_LOSS:
                loss = o3d.pipelines.registration.TukeyLoss(k=2.0 * vox)
            else:
                loss = None

            if PREFER_GICP and hasattr(o3d.pipelines.registration, "TransformationEstimationForGeneralizedICP"):
                est = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(loss)
                icp_fn = o3d.pipelines.registration.registration_generalized_icp
            else:
                est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
                icp_fn = o3d.pipelines.registration.registration_icp

            max_iter = int(min(280, max(80, tune.icp_base * (pyr[0] / max(vox, 1e-6)))))

            # Evaluate current pose at this level
            prev_trim = self._trimmed_rmse_after(T, s, t, dist_gate, trim_here)

            # Run ICP
            res = icp_fn(s, t, dist_gate, T, est,
                         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
            T_new = res.transformation
            new_trim = self._trimmed_rmse_after(T_new, s, t, dist_gate, trim_here)

            # Monotone acceptance (Eq. 13)
            if new_trim < prev_trim:
                T = T_new
                accept_log.append({"level": lvl, "vox": vox, "accepted": True,
                                   "prev_trim": prev_trim, "new_trim": new_trim,
                                   "dist_gate": dist_gate, "trim_frac": trim_here})
            else:
                # Tighten correspondence gate by beta and retry once (paper Sec 3.2)
                dist_gate_tight = dist_gate * BETA_GATE_TIGHTEN
                res2 = icp_fn(s, t, dist_gate_tight, T, est,
                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
                T_retry = res2.transformation
                retry_trim = self._trimmed_rmse_after(T_retry, s, t, dist_gate_tight, trim_here)

                if retry_trim < prev_trim:
                    T = T_retry
                    accept_log.append({"level": lvl, "vox": vox, "accepted": True,
                                       "retry": True, "prev_trim": prev_trim,
                                       "new_trim": retry_trim, "dist_gate": dist_gate_tight,
                                       "trim_frac": trim_here})
                else:
                    # Reject: proceed to next level without updating T
                    accept_log.append({"level": lvl, "vox": vox, "accepted": False,
                                       "prev_trim": prev_trim, "new_trim": new_trim,
                                       "retry_trim": retry_trim, "dist_gate": dist_gate,
                                       "trim_frac": trim_here})

        return T, accept_log


# ===== FMR / correspondence scoring for coarse stage =====
def fmr_and_ir_from_correspondences(src_pcd: o3d.geometry.PointCloud,
                                    tgt_pcd: o3d.geometry.PointCloud,
                                    corr_any,
                                    T_gt: np.ndarray,
                                    tau_m: float = INLIER_DIST_THRESH,
                                    gamma: float = FMR_RECALL_THRESH):
    try:
        corr = np.asarray(corr_any)
    except Exception:
        corr = np.array(list(corr_any), dtype=int)

    if corr.size == 0:
        return 0.0, False, 0
    if corr.ndim == 1:
        corr = corr.reshape(-1, 2)

    idx_s = corr[:, 0].astype(int)
    idx_t = corr[:, 1].astype(int)
    P = np.asarray(src_pcd.points)[idx_s]
    Q = np.asarray(tgt_pcd.points)[idx_t]

    P_gt = transform_points(P, T_gt)
    d = np.linalg.norm(P_gt - Q, axis=1)
    ir = float(np.mean(d < tau_m))
    return ir, bool(ir >= gamma), int(len(idx_s))


# ===== Runner =====
def load_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["pairs"]

def make_dir(d: Path): d.mkdir(parents=True, exist_ok=True)

def main(BENCH_DIR, MANIFEST, OUT_DIR):
    make_dir(OUT_DIR)
    pairs = load_manifest(MANIFEST)
    print(f"Loaded {len(pairs)} pairs from {MANIFEST}")

    aligner = AdaptiveAligner()
    rng = np.random.default_rng(2025)

    results_list = []
    T_ransac_delta_list = []; T_before_icp_list = []; T_icp_delta_list = []
    T_final_list = []; T_gt_list = []

    fmr_pass_flags = []; pose_pass_flags = []

    for idx, P in enumerate(pairs):
        src_path = Path(P["src_ply"])
        tgt_path = Path(P["tgt_ply"])
        T_init = to_mat4(P["T_init_rowmajor"])
        T_gt   = to_mat4(P["T_gt_rowmajor"]) if "T_gt_rowmajor" in P else None

        print(f"\n[{idx+1}/{len(pairs)}] {src_path.name} -> {tgt_path.name}")

        src = o3d.io.read_point_cloud(str(src_path))
        tgt = o3d.io.read_point_cloud(str(tgt_path))
        if src.is_empty() or tgt.is_empty():
            print("  [WARN] Empty cloud(s); skipping.")
            continue

        tune = SelfTune(src, tgt, T_init)

        t0 = time.time()
        src_init = o3d.geometry.PointCloud(src); src_init.transform(T_init)

        # --- Coarse: SIGMA + RANSAC (optional)
        T_ransac_delta = np.eye(4)
        ransac_corr = np.zeros((0, 2), dtype=int)
        n_inliers = 0
        ransac_cert = {"edge_min": tune.ransac_edge_min, "dist": tune.ransac_dist, **tune.ransac_cert}

        if USE_COARSE_RANSAC:
            T_ransac_delta, ransac_corr, n_inliers = aligner.coarse_ransac(src_init, tgt, tune)

        T_est_before_icp = T_ransac_delta @ T_init

        # Hardness reinforcement using coarse results
        if (ransac_corr.shape[0] < 300) or (tune.overlap < 0.25):
            tune.very_hard = True
            tune.icp_base = ICP_BASE_ITERS_HARD
            tune.icp_dist_factor = max(tune.icp_dist_factor, 0.60)

        # --- Refine with monotone acceptance (+ multistart if very hard)
        if ENABLE_MULTISTART and tune.very_hard:
            seeds = [T_est_before_icp]
            for _ in range(MULTISTART_SEEDS - 1):
                seeds.append(jitter_pose(T_est_before_icp, rot_deg=2.5, trans_m=0.05, rng=rng))

            best_T = None; best_trim = np.inf; best_log = None
            vox0 = tune.voxel0
            s0 = src.voxel_down_sample(vox0); t0p = tgt.voxel_down_sample(vox0)
            for T_seed in seeds:
                T_try, log_try = aligner.refine_icp(src, tgt, T_seed, tune)
                trim_try = aligner._trimmed_rmse_after(
                    T_try, s0, t0p, tune.icp_dist_factor*vox0,
                    get_trim_frac_for_level(0, tune.levels)
                )
                if trim_try < best_trim:
                    best_T, best_trim, best_log = T_try, trim_try, log_try
            T_est_final, accept_log = best_T, best_log
        else:
            T_est_final, accept_log = aligner.refine_icp(src, tgt, T_est_before_icp, tune)

        t1 = time.time()

        # Paper-style coarse FMR (under GT)
        ir_corr, fmr_pass, n_corr = (0.0, False, 0)
        if USE_COARSE_RANSAC and T_gt is not None:
            ir_corr, fmr_pass, n_corr = fmr_and_ir_from_correspondences(
                src_init, tgt, ransac_corr, T_gt, INLIER_DIST_THRESH, FMR_RECALL_THRESH
            )

        # Pose errors
        rre_deg, rte_m, pose_pass = (float("nan"), float("nan"), False)
        if T_gt is not None:
            rre_deg, rte_m = pose_errors(T_est_final, T_gt)
            pose_pass = (rre_deg < POSE_RRE_OK_DEG) and (rte_m < POSE_RTE_OK_M)

        # Save aligned PLY (optional)
        if SAVE_ALIGNED_PLY:
            aligned_dir = OUT_DIR / "aligned_ply"; make_dir(aligned_dir)
            src_aligned = o3d.geometry.PointCloud(src); src_aligned.transform(T_est_final)
            o3d.io.write_point_cloud(
                str(aligned_dir / f"{src_path.stem}_aligned_to_{tgt_path.stem}.ply"),
                src_aligned, write_ascii=False, compressed=False
            )

        # Deltas
        T_icp_delta = T_est_final @ np.linalg.inv(T_est_before_icp)

        # Certificates
        cert = None
        if LOG_CERTS:
            cert = {
                "spacing_median": tune.spacing,
                "bbox_diag": tune.diag,
                "overlap_quick": tune.overlap,
                "very_hard": bool(tune.very_hard),
                "sigma": {"kappa": SIGMA_KAPPA, "B_r": SIGMA_B_R,
                          "B_theta": SIGMA_B_THETA, "B_phi": SIGMA_B_PHI,
                          "k_delta": SIGMA_K_DELTA},
                "matching": {"ratio_thresh": MATCH_RATIO_THRESH},
                "ransac": ransac_cert,
                "pyramid": {"voxel0": tune.voxel0, "step": tune.pyr_step, "levels": tune.levels},
                "icp": {"base_iters": tune.icp_base, "dist_factor": tune.icp_dist_factor,
                        "trim_schedule": TRIM_SCHEDULE_BASE,
                        "beta_gate_tighten": BETA_GATE_TIGHTEN},
                "accept_log": accept_log
            }

        # Record
        item = {
            "pair_index": idx,
            "src_id": P["src_id"], "tgt_id": P["tgt_id"],
            "src_ply": str(src_path), "tgt_ply": str(tgt_path),
            "T_init_rowmajor": P["T_init_rowmajor"],
            "T_gt_rowmajor": P.get("T_gt_rowmajor", None),
            "T_ransac_delta_rowmajor": T_ransac_delta.reshape(-1).tolist(),
            "T_est_before_icp_rowmajor": T_est_before_icp.reshape(-1).tolist(),
            "T_icp_delta_rowmajor": T_icp_delta.reshape(-1).tolist(),
            "T_est_final_rowmajor": T_est_final.reshape(-1).tolist(),
            "timing_sec": round(t1 - t0, 3),
            "paper_metrics": {
                "corr_count": int(n_corr),
                "corr_inlier_ratio_gt": float(ir_corr),
                "fmr_pass": bool(fmr_pass),
                "fmr_params": {"tau_m": INLIER_DIST_THRESH, "gamma": FMR_RECALL_THRESH},
                "rre_deg": float(rre_deg),
                "rte_m": float(rte_m),
                "pose_success": bool(pose_pass),
                "pose_thresh": {"rre_deg": POSE_RRE_OK_DEG, "rte_m": POSE_RTE_OK_M}
            },
            "certificate": cert
        }

        results_list.append(item)
        T_ransac_delta_list.append(T_ransac_delta)
        T_before_icp_list.append(T_est_before_icp)
        T_icp_delta_list.append(T_icp_delta)
        T_final_list.append(T_est_final)
        if T_gt is not None: T_gt_list.append(T_gt)

        if USE_COARSE_RANSAC: fmr_pass_flags.append(bool(fmr_pass))
        pose_pass_flags.append(bool(pose_pass))

        print(f"  time={item['timing_sec']}s | "
              f"RANSAC corr={n_corr} inliers={n_inliers} IR_corr={ir_corr:.3f} FMR={'✓' if fmr_pass else '×'} | "
              f"RRE={rre_deg:.2f}° RTE={rte_m:.3f}m {'✓' if pose_pass else '×'}")

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"pairs": results_list}, f, indent=2)

    def stack_or_empty(L): return np.stack(L) if len(L) else np.zeros((0, 4, 4))
    np.savez(OUT_DIR / "results_mats.npz",
             T_ransac_delta=stack_or_empty(T_ransac_delta_list),
             T_before_icp=stack_or_empty(T_before_icp_list),
             T_icp_delta=stack_or_empty(T_icp_delta_list),
             T_est_final=stack_or_empty(T_final_list),
             T_gt=stack_or_empty(T_gt_list))

    # Quick summary
    if len(pose_pass_flags):
        pose_rr = 100.0 * (sum(pose_pass_flags) / len(pose_pass_flags))
        print("\n=== Dataset Summary (pose) ===")
        print(f"Pose Registration Recall (RRE<{POSE_RRE_OK_DEG}°, RTE<{POSE_RTE_OK_M}m): {pose_rr:.1f}%")
    if USE_COARSE_RANSAC and len(fmr_pass_flags):
        fmr_pct = 100.0 * (sum(fmr_pass_flags) / len(fmr_pass_flags))
        print(f"FMR (tau={INLIER_DIST_THRESH}m, gamma={FMR_RECALL_THRESH}): {fmr_pct:.1f}%")

    print(f"\nSaved results to:\n  {OUT_DIR / 'results.json'}\n  {OUT_DIR / 'results_mats.npz'}")
    if SAVE_ALIGNED_PLY:
        print(f"Aligned PLYs in: {OUT_DIR / 'aligned_ply'}")

if __name__ == "__main__":
    for folder in ["Gazebo2_local_frame"]:
        BENCH_DIR = Path(rf"data/{folder}/bench_apartment_full")
        MANIFEST = BENCH_DIR / "pairs_manifest.json"
        OUT_DIR = BENCH_DIR / "results_mymethod_hard"
        main(BENCH_DIR, MANIFEST, OUT_DIR)
