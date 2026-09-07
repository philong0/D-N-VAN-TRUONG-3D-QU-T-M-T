"""
GNM Head v3 multi-view identity fit, using PIPNet/WFLW-98 landmarks + EPNP
camera poses instead of MediaPipe. Reproduces src/lib/gnm/fit.ts +
procrustes.ts + identity-fit.ts's EXACT algorithm (Procrustes similarity
alignment per view, then one ridge-regularized joint least-squares solve
across views: min ||B*c - r||^2 + lambda*||c||^2) — validated in
scratchpad/d3_identity_fit.py (D3, GO: 8/8 real test views across 2 patients
improved over the unfitted template, 0 views dropped, no distortion).

The one necessary generalization vs. today's TS `solveIdentityCoefficientsMultiView`
(which assumes every view shares the same fixed point count): views may have
different point SUBSETS (the 90° view legitimately drops 3 eyebrow points,
see gnm_correspondence.py) — the normal-equations math itself is unchanged,
just accumulated per-view over each view's own available rows.

The one new, stated approximation (PIPNet has no per-point Z the way
MediaPipe does): a "pseudo-3D observed position" per landmark per view, by
back-projecting the real detected 2D point through the view's own validated
EPNP pose at the SAME depth the corresponding GNM template landmark has in
that pose. Validated via reprojection: the resulting fitted mesh reprojects
9-28% closer to the real 2D detections than the unfitted template, on both
patients, at all 4 angles (see D3's own report).
"""

import zipfile
from pathlib import Path

import cv2
import numpy as np

from gnm_correspondence import VERTEX_INDICES, WEIGHTS, WFLW_INDICES, point_mask, POINT_WEIGHTS

GNM_ASSET = Path(__file__).parent.parent / "public" / "models" / "gnm" / "gnm_head_v3.npz"

_VIDX = np.array(VERTEX_INDICES, dtype=int)  # (N,3)
_W = np.array(WEIGHTS, dtype=np.float64)  # (N,3)
_POINT_WEIGHTS = np.array(POINT_WEIGHTS, dtype=np.float64)  # (N,) -- see gnm_correspondence.py's own docstring

DEFAULT_REGULARIZATION = 1e-5  # identical constant to fit.ts's DEFAULT_REGULARIZATION


def _load_npy_from_npz(name: str) -> np.ndarray:
    with zipfile.ZipFile(GNM_ASSET) as zf:
        with zf.open(name) as f:
            return np.load(f)


_TEMPLATE_POSITIONS = None
_IDENTITY_BASIS = None
_TRIANGLES = None


def _ensure_loaded():
    global _TEMPLATE_POSITIONS, _IDENTITY_BASIS, _TRIANGLES
    if _TEMPLATE_POSITIONS is None:
        _TEMPLATE_POSITIONS = _load_npy_from_npz("template_vertex_positions.npy")  # (17821,3)
        _IDENTITY_BASIS = _load_npy_from_npz("vertex_identity_basis.npy")  # (253,17821,3)
        _TRIANGLES = _load_npy_from_npz("triangles.npy")  # (35324,3)


def evaluate_points(vertex_positions: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(N,3) landmark positions from the full point set (or a boolean-masked subset), against ANY (vertexCount,3) mesh (template or fitted)."""
    vidx = _VIDX if mask is None else _VIDX[mask]
    w = _W if mask is None else _W[mask]
    pts = vertex_positions[vidx]  # (N,3,3)
    return np.einsum("nk,nkc->nc", w, pts)


def evaluate_basis(mask: np.ndarray | None = None) -> np.ndarray:
    """(identityDim, N, 3) landmark-space identity basis, full set or masked subset."""
    _ensure_loaded()
    vidx = _VIDX if mask is None else _VIDX[mask]
    w = _W if mask is None else _W[mask]
    pts = _IDENTITY_BASIS[:, vidx]  # (identityDim, N, 3, 3)
    return np.einsum("nk,dnkc->dnc", w, pts)


def solve_procrustes(source: np.ndarray, target: np.ndarray):
    """Kabsch similarity transform (rotation+scale+translation) mapping source -> target. Same algorithm as procrustes.ts's solveProcrustes (SVD here vs. its hand-rolled Jacobi eigendecomposition of a 3x3 — same textbook Kabsch method)."""
    cA, cB = source.mean(axis=0), target.mean(axis=0)
    A, B = source - cA, target - cB
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d if d != 0 else 1])
    R = Vt.T @ D @ U.T
    varA = (A ** 2).sum()
    scale = (S[0] + S[1] + d * S[2]) / varA if varA > 0 else 1.0
    t = cB - scale * (R @ cA)
    return R, scale, t


def apply_rigid(points: np.ndarray, R, scale, t) -> np.ndarray:
    return (scale * (points @ R.T)) + t


def backproject_pseudo3d(points_2d, template_landmarks_masked, R_pose, t_pose, camera_matrix):
    Xc_template = (R_pose @ template_landmarks_masked.T).T + t_pose
    Zc = Xc_template[:, 2]
    fx = camera_matrix[0, 0]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    u, v = points_2d[:, 0], points_2d[:, 1]
    Xc_obs = np.stack([(u - cx) / fx * Zc, (v - cy) / fx * Zc, Zc], axis=1)
    return (Xc_obs - t_pose) @ R_pose  # = R^T (Xc_obs - t), row-vector form


def solve_epnp(points_3d, points_2d, camera_matrix):
    dist_coeffs = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def solve_identity_multiview(landmark_basis_full: np.ndarray, view_masks, view_target_offsets, regularization: float,
                              point_weights: np.ndarray | None = None) -> np.ndarray:
    """Same normal-equations math as identity-fit.ts's solveIdentityCoefficientsMultiView,
    generalized to per-view point subsets (see module docstring).

    `point_weights` (optional, defaults to None = every point weighted equally,
    byte-identical to the original behavior): real per-landmark weight (length
    = landmark_basis_full.shape[1], i.e. 48 for the WFLW set) scaling how much
    that landmark's own real target offset contributes to the shared solve,
    applied BEFORE the ridge term -- regularization itself is never touched.
    See gnm_correspondence.py's own POINT_WEIGHTS docstring for the real,
    measured justification (nose is weighted 10x)."""
    identity_dim = landmark_basis_full.shape[0]
    M = np.zeros((identity_dim, identity_dim))
    rhs = np.zeros(identity_dim)
    for mask, offset in zip(view_masks, view_target_offsets):
        B_v = landmark_basis_full[:, mask, :].reshape(identity_dim, -1)
        if point_weights is not None:
            w = np.repeat(point_weights[mask], 3)
            Bw = B_v * w[None, :]
        else:
            Bw = B_v
        M += Bw @ B_v.T
        rhs += Bw @ offset.reshape(-1)
    M += regularization * np.eye(identity_dim)
    return np.linalg.solve(M, rhs)


def apply_identity_coefficients(coefficients: np.ndarray) -> np.ndarray:
    _ensure_loaded()
    return _TEMPLATE_POSITIONS + np.einsum("d,dvc->vc", coefficients, _IDENTITY_BASIS)


class ViewInput:
    __slots__ = ("landmarks_98", "camera_matrix", "is_profile_view", "image_shape")

    def __init__(self, landmarks_98, camera_matrix, is_profile_view, image_shape):
        self.landmarks_98 = landmarks_98
        self.camera_matrix = camera_matrix
        self.is_profile_view = is_profile_view
        self.image_shape = image_shape


def fit_multiview(views: list[ViewInput], regularization: float = DEFAULT_REGULARIZATION):
    """Full D3 pipeline for one patient's real photos.
    Returns (fitted_positions (17821,3), per_view_pose [(R,t,mask) or None per input view], warnings [str]).
    A view whose EPNP pose fails to solve is DROPPED (never fabricated) —
    same "drop, don't force" discipline as every prior validated phase.
    """
    _ensure_loaded()
    landmark_basis_full = evaluate_basis()
    template_landmarks_full = evaluate_points(_TEMPLATE_POSITIONS)

    view_masks = []
    view_offsets = []
    per_view_pose = []
    warnings = []

    for i, v in enumerate(views):
        mask = np.array(point_mask(v.is_profile_view), dtype=bool)
        kept_wflw = [wflw for wflw, keep in zip(WFLW_INDICES, mask) if keep]
        points_2d = np.array([v.landmarks_98[idx] for idx in kept_wflw], dtype=np.float64)

        template_masked = template_landmarks_full[mask]
        pose = solve_epnp(template_masked, points_2d, v.camera_matrix)
        if pose is None:
            warnings.append(f"view {i}: EPNP failed to converge, dropped")
            per_view_pose.append(None)
            continue
        R, t = pose

        observed = backproject_pseudo3d(points_2d, template_masked, R, t, v.camera_matrix)
        R_proc, scale_proc, t_proc = solve_procrustes(observed, template_masked)
        aligned = apply_rigid(observed, R_proc, scale_proc, t_proc)
        offset = aligned - template_masked

        view_masks.append(mask)
        view_offsets.append(offset)
        per_view_pose.append((R, t, mask))

    if len(view_masks) == 0:
        return None, per_view_pose, warnings + ["no usable view — fit aborted"]

    coefficients = solve_identity_multiview(landmark_basis_full, view_masks, view_offsets, regularization, point_weights=_POINT_WEIGHTS)
    fitted_positions = apply_identity_coefficients(coefficients)

    max_disp_mm = float(np.linalg.norm(fitted_positions - _TEMPLATE_POSITIONS, axis=1).max() * 1000)
    max_coeff = float(np.abs(coefficients).max())
    if max_disp_mm > 25 or max_coeff > 2.9:
        warnings.append(f"fit sanity check: maxVertexDisp={max_disp_mm:.1f}mm maxCoeff={max_coeff:.2f} — unusually large, verify before trusting")

    return fitted_positions, per_view_pose, warnings


def get_triangles() -> np.ndarray:
    _ensure_loaded()
    return _TRIANGLES


def get_template_positions() -> np.ndarray:
    """(17821,3) unfitted template shape — same array `fit_multiview` blends
    away from via the identity basis. Exposed (Phase 1, additive) for
    gnm_vertex_trust.py's `blend_positions_by_trust`, which needs the raw
    template as the trust=0 fallback shape; mirrors the existing
    `get_triangles()` accessor pattern rather than reaching into the
    module-private `_TEMPLATE_POSITIONS` from outside."""
    _ensure_loaded()
    return _TEMPLATE_POSITIONS
