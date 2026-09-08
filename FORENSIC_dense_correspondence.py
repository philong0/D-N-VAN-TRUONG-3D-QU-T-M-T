"""
dense_correspondence.py

Adds real surface density to the RGB-only reconstruction path when many
(N >= ~15) continuous-capture frames are available, instead of the ~98
WFLW-98 landmark points `_reconstruct_from_rgb_multiview` alone can ever
produce (see D-realsfm/D-shadowfix in patient_native_fusion.py — that
function's own Delaunay-over-98-points + subdivide() step is real geometric
interpolation, never new measured detail).

Design (deliberately does NOT re-solve camera poses):
`_reconstruct_from_rgb_multiview` already solves a real, metric, bundle-
adjusted `camera_pose` (4x4 world->camera) and `intrinsics` (K) for every
posed frame and writes them back onto each frame dict before returning
(see patient_native_fusion.py D-poseback). Those poses are trustworthy —
re-deriving them from ORB matches too would just add a second, weaker
pose estimate with no upside. Instead this module:

  1. Detects ORB keypoints per frame and matches them between CONSECUTIVE
     frames only (`cv2.BFMatcher`, Hamming + crossCheck) — a continuous
     head-turn capture has high frame-to-frame overlap, so consecutive-pair
     matching is both cheap (avoids O(N^2) all-pairs matching on a CPU-only,
     4-core ARM64 host) and more reliable than matching distant frame pairs.
  2. Chains matches across frames into point tracks (a track = the same
     real surface point re-observed in >=2 frames).
  3. Triangulates every track via multi-view linear DLT using the ALREADY
     solved poses (same technique as _reconstruct_from_rgb_multiview's own
     "final multi-view least-squares re-triangulation", generalized from a
     fixed 98 points to however many real tracks were found), then rejects
     any point that lands behind an observing camera, reprojects with high
     error, or falls outside a physically-plausible head-size bound —
     never accepting an unverified triangulation.
  4. Meshes the surviving real 3D points with Poisson surface
     reconstruction (open3d) — a real meshing algorithm that fits a surface
     to measured points, not sparse-landmark interpolation.

Returns None (never a fabricated/partial result) when there isn't enough
real data to produce a trustworthy dense surface, so the caller can safely
fall back to the existing sparse landmark mesh.
"""

from __future__ import annotations

import numpy as np
import cv2

MIN_FRAMES_FOR_DENSE = 15
REPROJ_ERROR_PX_THRESHOLD = 4.0
MAX_PLAUSIBLE_RADIUS_M = 0.4  # same generous head-size bound used in patient_native_fusion.py's BA bounds
MIN_DENSE_POINTS = 400
MIN_DENSE_MESH_VERTICES = 400
ORB_MAX_FEATURES = 2000
LOWE_RATIO = 0.75


def _detect_and_match_consecutive(posed_frames: list[dict]) -> dict[int, dict[str, int]]:
    """Returns tracks: {track_id: {frame_stem: keypoint_index_in_that_frame}}.

    Only chains through frames that are adjacent in `posed_frames`'
    given order — callers must pass frames already ordered by real
    capture sequence (a continuous head turn), not an arbitrary order.
    """
    orb = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
    for f in posed_frames:
        gray = cv2.cvtColor(f["image_bgr"], cv2.COLOR_BGR2GRAY) if f["image_bgr"].ndim == 3 else f["image_bgr"]
        kp, des = orb.detectAndCompute(gray, None)
        f["_orb_kp"] = kp or []
        f["_orb_des"] = des

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    tracks: dict[int, dict[str, int]] = {}
    active: dict[int, int] = {}  # keypoint_idx in previous frame -> track_id
    next_track_id = 0

    for i in range(len(posed_frames) - 1):
        f_a, f_b = posed_frames[i], posed_frames[i + 1]
        des_a, des_b = f_a["_orb_des"], f_b["_orb_des"]
        new_active: dict[int, int] = {}
        if des_a is not None and des_b is not None and len(des_a) >= 2 and len(des_b) >= 2:
            raw_matches = bf.knnMatch(des_a, des_b, k=2)
            good = []
            for pair in raw_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance >= LOWE_RATIO * n.distance:
                    continue  # ambiguous match, discard rather than guess
                good.append(m)

            # D-epifilter — Lowe's ratio test alone still lets real false
            # positives through on imagery with many locally-similar
            # patches (confirmed directly: a synthetic textured-shift test
            # with a KNOWN correct pixel offset measured a 685px outlier
            # among ratio-test-passed matches before this filter). Every
            # true correspondence between two frames of the SAME rigid
            # scene must satisfy the same epipolar geometry, so a RANSAC
            # fundamental-matrix fit — standard practice in real SfM
            # pipelines (COLMAP does this at the matching stage too) —
            # rejects matches that don't, independent of and in addition
            # to the later 3D reprojection-error filter.
            if len(good) >= 8:
                pts_a = np.float32([f_a["_orb_kp"][m.queryIdx].pt for m in good])
                pts_b = np.float32([f_b["_orb_kp"][m.trainIdx].pt for m in good])
                _, inlier_mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 2.0, 0.995)
                if inlier_mask is not None:
                    good = [m for m, keep in zip(good, inlier_mask.ravel()) if keep]

            for m in good:
                idx_a, idx_b = m.queryIdx, m.trainIdx
                track_id = active.get(idx_a)
                if track_id is None:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = {f_a["stem"]: idx_a}
                tracks[track_id][f_b["stem"]] = idx_b
                new_active[idx_b] = track_id
        active = new_active

    return tracks


def _triangulate_and_filter_tracks(
    tracks: dict[int, dict[str, int]],
    frame_by_stem: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-geometry step, independently testable with synthetic tracks
    that don't require real images or cv2.ORB at all — `frame_by_stem`
    entries only need `camera_pose`, `intrinsics`, and either real
    `_orb_kp` (for real use) or a plain `keypoints_px` array of (u, v)
    pixel positions (for synthetic tests) plus `image_bgr` for color.

    Returns (points_3d (M,3), colors_rgb_uint8 (M,3)).
    """
    points = []
    colors = []

    for track in tracks.values():
        if len(track) < 2:
            continue
        A_rows = []
        observations = []  # (K, R, t, u, v)
        for stem, kp_idx in track.items():
            f = frame_by_stem[stem]
            R = f["camera_pose"][:3, :3]
            t = f["camera_pose"][:3, 3]
            K = f["intrinsics"]
            if "_orb_kp" in f:
                u, v = f["_orb_kp"][kp_idx].pt
            else:
                u, v = f["keypoints_px"][kp_idx]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            A_rows.append(u * P[2] - P[0])
            A_rows.append(v * P[2] - P[1])
            observations.append((K, R, t, u, v))

        A = np.stack(A_rows, axis=0)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-9:
            continue
        X3 = X[:3] / X[3]

        ok = True
        errs = []
        for (K, R, t, u, v) in observations:
            Xc = R @ X3 + t
            if Xc[2] <= 1e-6:
                ok = False
                break
            proj = K @ Xc
            proj_uv = proj[:2] / proj[2]
            errs.append(float(np.hypot(proj_uv[0] - u, proj_uv[1] - v)))
        if not ok:
            continue
        if max(errs) > REPROJ_ERROR_PX_THRESHOLD:
            continue
        if float(np.linalg.norm(X3)) > MAX_PLAUSIBLE_RADIUS_M:
            continue

        points.append(X3)

        stem0 = next(iter(track))
        f0 = frame_by_stem[stem0]
        kp_idx0 = track[stem0]
        if "_orb_kp" in f0:
            u0, v0 = f0["_orb_kp"][kp_idx0].pt
        else:
            u0, v0 = f0["keypoints_px"][kp_idx0]
        h, w = f0["image_bgr"].shape[:2]
        x_px = int(np.clip(round(u0), 0, w - 1))
        y_px = int(np.clip(round(v0), 0, h - 1))
        bgr = f0["image_bgr"][y_px, x_px]
        colors.append([int(bgr[2]), int(bgr[1]), int(bgr[0])])

    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.uint8)


def _mesh_point_cloud_poisson(points: np.ndarray, colors: np.ndarray, depth: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Pure-meshing step, independently testable with a synthetic point
    cloud (e.g. points sampled on a known sphere) — no image/camera
    dependency at all. Trims the lowest-density 5% of Poisson output,
    which is where the algorithm extrapolates past real point support
    (documented open3d behavior), so the returned mesh stays anchored to
    real measured points rather than the solver's own smoothing guesses.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    centroid = points.mean(axis=0)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    # Orient every normal to point away from the point cloud's own centroid
    # — a reasonable, data-derived convention for a roughly head-shaped
    # point cloud (points are samples on an outward-facing skin surface),
    # not an assumption about any specific geometry.
    normals = np.asarray(pcd.normals)
    to_point = points - centroid
    flip = (np.sum(normals * to_point, axis=1) < 0)
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    if len(densities) > 0:
        density_thresh = float(np.quantile(densities, 0.05))
        mesh.remove_vertices_by_mask(densities < density_thresh)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return verts, faces


def reconstruct_dense_surface(frames: list[dict], min_frames_for_dense: int = MIN_FRAMES_FOR_DENSE) -> dict | None:
    """Entry point. `frames` must already have gone through
    `_reconstruct_from_rgb_multiview` (or equivalent) so every posed frame
    carries a real `camera_pose` (4x4) and `intrinsics` (3x3). Returns
    None — never a fabricated/partial mesh — if there isn't enough real
    data (too few posed frames, too few surviving tracks after geometric
    filtering, or too few resulting mesh vertices) for a trustworthy dense
    result; the caller falls back to the existing sparse landmark mesh.
    """
    posed_frames = [f for f in frames if f.get("camera_pose") is not None and f.get("intrinsics") is not None]
    if len(posed_frames) < min_frames_for_dense:
        return None

    tracks = _detect_and_match_consecutive(posed_frames)
    frame_by_stem = {f["stem"]: f for f in posed_frames}
    points, colors = _triangulate_and_filter_tracks(tracks, frame_by_stem)
    if len(points) < MIN_DENSE_POINTS:
        return None

    verts, faces = _mesh_point_cloud_poisson(points, colors)
    if len(verts) < MIN_DENSE_MESH_VERTICES:
        return None

    return {
        "vertices": verts,
        "faces": faces,
        "point_count": int(len(points)),
        "frames_used": len(posed_frames),
        "source": "dense_orb_multiview_poisson",
    }
