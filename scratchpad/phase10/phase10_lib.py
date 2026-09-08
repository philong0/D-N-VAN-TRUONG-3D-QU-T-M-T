"""
Phase 10 shared helpers: MediaPipe (independent detector) reprojection
metric + jaw/mouth/eye/eyebrow 3D drift metric, used by the weight-sweep and
A/B gate scripts. Not imported by production code.
"""
import sys
sys.path.insert(0, "/home/ubuntu/dr-vantruong-3d-studio/ai-engine")

import numpy as np
import cv2
import mediapipe as mp

from gnm_correspondence import VERTEX_INDICES, WEIGHTS, WFLW_INDICES

_mp_face_mesh = mp.solutions.face_mesh

# POINTS layout in gnm_correspondence.py: jaw(7) rows0-6, nose(1) row7,
# mouth(20) rows8-27, eye(10) rows28-37, eyebrow(10) rows38-47.
REGION_ROWS = {
    "jaw": slice(0, 7),
    "nose_sparse": slice(7, 8),
    "mouth": slice(8, 28),
    "eye": slice(28, 38),
    "eyebrow": slice(38, 48),
}


def landmark_positions(positions, rows_slice):
    vidx = np.array(VERTEX_INDICES[rows_slice], dtype=int)
    w = np.array(WEIGHTS[rows_slice], dtype=np.float64)
    pts = positions[vidx]
    return np.einsum("nk,nkc->nc", w, pts)


def drift_mm(before_positions, after_positions, rows_slice):
    a = landmark_positions(before_positions, rows_slice)
    b = landmark_positions(after_positions, rows_slice)
    return float(np.linalg.norm(a - b, axis=1).mean() * 1000)


def detect_mediapipe(image_bgr):
    """Real MediaPipe FaceMesh (478 pts incl. iris), independent detector
    from WFLW/PIPNet. Returns (478,2) pixel coords or None if no face."""
    h, w = image_bgr.shape[:2]
    with _mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                 refine_landmarks=True, min_detection_confidence=0.3) as fm:
        res = fm.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)


def project_points(points_3d, R, t, camera_matrix):
    """Forward pinhole projection -- exact inverse of gnm_identity_fit.py's
    backproject_pseudo3d (same R,t,camera_matrix convention: fx==fy==image
    width, cx,cy = w/2,h/2)."""
    Xc = (R @ points_3d.T).T + t
    fx = camera_matrix[0, 0]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    u = Xc[:, 0] / Xc[:, 2] * fx + cx
    v = Xc[:, 1] / Xc[:, 2] * fx + cy
    return np.stack([u, v], axis=1)


def region_reprojection_error_px(mesh_positions, region_vids, R, t, camera_matrix,
                                  mp_points_px, candidate_idx):
    """Mean nearest-neighbor pixel distance from this region's projected GNM
    vertices to the (pre-selected) candidate subset of independently-detected
    MediaPipe points. `candidate_idx` must be selected ONCE (e.g. from the
    BEFORE mesh) and reused for both before/after evaluation, so both sides
    are scored against the identical MediaPipe subset (fair comparison, not
    a whole-face nearest-neighbor that would look trivially good regardless
    of local shape accuracy)."""
    proj = project_points(mesh_positions[region_vids], R, t, camera_matrix)
    cand = mp_points_px[candidate_idx]
    d = np.linalg.norm(proj[:, None, :] - cand[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def select_candidate_mp_idx(before_mesh_positions, region_vids, R, t, camera_matrix,
                             mp_points_px, k=40):
    """Nearest-k MediaPipe point indices (in pixel space) to this region's
    own projected centroid from the BEFORE mesh -- fixed once, shared by
    before/after scoring (see region_reprojection_error_px docstring)."""
    centroid = mesh_positions_centroid(before_mesh_positions, region_vids)
    proj_centroid = project_points(centroid[None, :], R, t, camera_matrix)[0]
    d = np.linalg.norm(mp_points_px - proj_centroid[None, :], axis=1)
    return np.argsort(d)[:k]


def mesh_positions_centroid(positions, vids):
    return positions[vids].mean(axis=0)
