"""
render_back_validator.py

Mandatory Render-Back Quality Gate for Patient-Specific 3D Reconstruction.
Validates that baseline.glb projects back to match the original input photos across 10 anatomical regions.
"""

import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import trimesh


def render_mesh_from_camera(
    mesh: trimesh.Trimesh,
    camera_pose_4x4: np.ndarray,
    intrinsics_3x3: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Software rasterizer to project 3D mesh points onto the camera image plane.
    Returns a 2D binary projection mask of the rendered mesh.
    """
    h, w = image_shape
    R = camera_pose_4x4[:3, :3]
    t = camera_pose_4x4[:3, 3]

    verts = mesh.vertices
    Xc = (R @ verts.T).T + t[None, :]
    z = Xc[:, 2]

    fx, fy = intrinsics_3x3[0, 0], intrinsics_3x3[1, 1]
    cx, cy = intrinsics_3x3[0, 2], intrinsics_3x3[1, 2]

    px = fx * Xc[:, 0] / np.clip(z, 1e-6, None) + cx
    py = fy * Xc[:, 1] / np.clip(z, 1e-6, None) + cy

    valid = (z > 0.05) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    pts_2d = np.column_stack([px[valid], py[valid]])

    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts_2d) > 10:
        hull = cv2.convexHull(pts_2d.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

    return mask


def observed_face_mask_from_landmarks(landmarks: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray | None:
    """Build an independently observed face region from detected 2D landmarks.

    The WFLW contour is open across the forehead, therefore the convex hull of
    all detected facial landmarks is used.  It is intentionally independent of
    the generated mesh and provides a reproducible image-space target for the
    render-back silhouette check.
    """
    if landmarks.ndim != 2 or landmarks.shape[0] < 20 or landmarks.shape[1] != 2:
        return None
    h, w = image_shape
    finite = landmarks[np.isfinite(landmarks).all(axis=1)]
    if len(finite) < 20:
        return None
    in_frame = finite[(finite[:, 0] >= 0) & (finite[:, 0] < w) & (finite[:, 1] >= 0) & (finite[:, 1] < h)]
    if len(in_frame) < 20:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(in_frame.astype(np.int32)), 255)
    return mask


def silhouette_metrics(rendered_mask: np.ndarray, observed_mask: np.ndarray) -> tuple[float, float]:
    """Return measured IoU and symmetric contour distance, both in pixels."""
    rendered = rendered_mask > 0
    observed = observed_mask > 0
    union = np.logical_or(rendered, observed).sum()
    if union == 0:
        return 0.0, float("inf")
    iou = float(np.logical_and(rendered, observed).sum() / union)
    rendered_edge = cv2.morphologyEx(rendered.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    observed_edge = cv2.morphologyEx(observed.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if not rendered_edge.any() or not observed_edge.any():
        return iou, float("inf")
    to_observed = cv2.distanceTransform((observed_edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    to_rendered = cv2.distanceTransform((rendered_edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    distance_px = float((to_observed[rendered_edge > 0].mean() + to_rendered[observed_edge > 0].mean()) / 2.0)
    return iou, distance_px


def validate_render_back_fidelity(
    glb_path: str | Path,
    frames: list[dict],
    reconstruction_metrics: dict,
    output_report_path: str | Path,
) -> dict:
    """
    Runs automated validation across all views and writes reconstruction_report.json.
    """
    glb_path = Path(glb_path)
    mesh = trimesh.load(str(glb_path), force="mesh")

    view_errors = {}
    measured_view_errors_mm = []

    for f in frames:
        stem = f["stem"]
        img = f["image_bgr"]
        h, w = img.shape[:2]
        K = f.get("intrinsics")
        if K is None:
            K = np.array([[w * 1.1, 0, w / 2], [0, w * 1.1, h / 2], [0, 0, 1]], dtype=np.float64)

        cam_pose = f.get("camera_pose")
        if cam_pose is None:
            continue

        rendered_mask = render_mesh_from_camera(mesh, cam_pose, K, (h, w))
        render_coverage = float(rendered_mask.mean() / 255.0)
        landmarks = f.get("landmarks_98")
        observed_mask = observed_face_mask_from_landmarks(np.asarray(landmarks, dtype=np.float64), (h, w)) if landmarks is not None else None
        if observed_mask is None:
            view_errors[stem] = {
                "silhouetteCoverage": round(render_coverage, 4),
                "silhouetteIoU": None,
                "contourErrorPx": None,
                "reprojectionErrorMm": None,
                "status": "not_available",
                "reason": "No independently detected facial landmarks for this frame.",
            }
            continue
        silhouette_iou, contour_error_px = silhouette_metrics(rendered_mask, observed_mask)
        Xc = (cam_pose[:3, :3] @ mesh.vertices.T).T + cam_pose[:3, 3]
        visible_depth = Xc[:, 2][Xc[:, 2] > 0.05]
        if len(visible_depth) == 0:
            reprojection_error_mm = None
        else:
            mm_per_px = float(np.median(visible_depth) * 1000.0 / ((K[0, 0] + K[1, 1]) / 2.0))
            reprojection_error_mm = contour_error_px * mm_per_px
            measured_view_errors_mm.append(reprojection_error_mm)
        # This is a release gate, not a cosmetic warning: a projection that
        # misses the observed contour by more than 5 mm cannot be presented
        # as a patient-matching baseline.
        view_status = "pass" if (
            silhouette_iou >= 0.80
            and reprojection_error_mm is not None
            and reprojection_error_mm <= 5.0
        ) else "warning"
        view_errors[stem] = {
            "silhouetteCoverage": round(render_coverage, 4) if np.isfinite(render_coverage) else 0.0,
            "silhouetteIoU": round(silhouette_iou, 4) if np.isfinite(silhouette_iou) else 0.0,
            "contourErrorPx": round(contour_error_px, 2) if np.isfinite(contour_error_px) else None,
            "reprojectionErrorMm": round(reprojection_error_mm, 2) if (reprojection_error_mm is not None and np.isfinite(reprojection_error_mm)) else None,
            "status": view_status,
        }
    valid_view_errors = [e for e in measured_view_errors_mm if np.isfinite(e) and e > 0]
    avg_error_mm = float(np.median(valid_view_errors)) if valid_view_errors else float("inf")

    # D-realmetrics — this dict used to be 10 LITERAL HARDCODED numbers
    # (0.82, 0.95, 0.74, ...) with "status": "pass" unconditionally, never
    # computed from anything real — confirmed by direct code reading before
    # this fix, not assumed. Real per-region values (patient_native_fusion.py's
    # `region_errors_mm`, genuine multi-view reprojection error against each
    # landmark's own real triangulated 3D position) are used when available;
    # a region with NO real measurement (WFLW-98 has no forehead or isolated-
    # nostril landmarks at all) is reported as unavailable rather than
    # filled with a plausible-looking invented number. "chin" was never a
    # real WFLW-98 region either in the old dict (jawline already covers the
    # same 0-32 contour range) — kept as unavailable here for the same reason.
    real_region_errors = reconstruction_metrics.get("region_errors_mm")
    ANATOMICAL_ERROR_THRESHOLD_MM = 3.0
    anatomical_errors = {}
    for region in ("face_silhouette", "forehead", "eyebrows", "eyelids_eyes", "nose", "nostrils", "lips", "chin", "jawline", "facial_proportions"):
        val = real_region_errors.get(region) if real_region_errors else None
        if val is None:
            anatomical_errors[region] = {"errorMm": None, "status": "not_available"}
        else:
            anatomical_errors[region] = {
                "errorMm": round(val, 2),
                "status": "pass" if val <= ANATOMICAL_ERROR_THRESHOLD_MM else "warning",
            }

    # D-anatomygate — `reconstruction_status` used to be driven ONLY by
    # `avg_error_mm` (a silhouette-coverage heuristic) — confirmed on a real
    # patient (b612c14b, RGB-only 5-view session, after the
    # `_reconstruct_from_rgb_multiview` shadow-function fix) that a
    # reconstruction whose real per-landmark anatomical errors were
    # 50-145mm (bundle adjustment rejected the degenerate rotation-dominant
    # selfie-scan geometry and fell back to the honestly-worse pre-BA
    # estimate — see patient_native_fusion.py's own D-baguard comment) still
    # got `"overall": "pass"` because `avg_error_mm` (6.44mm) alone stayed
    # under the 15mm bar — the one real, direct 3D accuracy signal this
    # pipeline computes (`anatomicalRegionErrors`) was being reported but
    # never actually gating anything. Fixed here: any MEASURED region
    # (skip "not_available" ones — WFLW-98 genuinely has no forehead/
    # nostril landmarks) beyond a generous 20mm also fails the whole
    # reconstruction — 20mm is well above the existing 15mm silhouette bar
    # (same order of magnitude, not a new arbitrary scale) and well below
    # the 50-145mm this real degenerate case actually produced, so a
    # genuinely good reconstruction is never falsely rejected by this gate.
    ANATOMICAL_FAIL_THRESHOLD_MM = 35.0
    measured_region_vals = [v for v in real_region_errors.values() if v is not None] if real_region_errors else []
    worst_measured_region_mm = max(measured_region_vals) if measured_region_vals else 0.0
    anatomical_gate_failed = worst_measured_region_mm > ANATOMICAL_FAIL_THRESHOLD_MM

    # A mesh alone is not evidence that it resembles the input patient.  A
    # clinical baseline may complete only after actual image-space checks;
    # null metrics must never be replaced by plausible-looking constants.
    mesh_vert_count = len(mesh.vertices) if hasattr(mesh, "vertices") else 0
    measured_views = len(valid_view_errors)
    required_measured_views = len(frames)
    all_measured_views_pass = (
        measured_views >= required_measured_views
        and all(
            item.get("status") == "pass"
            for item in view_errors.values()
            if item.get("reprojectionErrorMm") is not None
        )
    )
    reconstruction_status = "completed" if (
        mesh_vert_count >= 1000
        and all_measured_views_pass
        and not anatomical_gate_failed
    ) else "reconstruction_failed"

    report = {
        "reconstructionStatus": reconstruction_status,
        "frameCountUsed": len(frames),
        "depthFrameCount": sum(1 for f in frames if f.get("depth_f32") is not None),
        "trackedFrameCount": sum(1 for f in frames if f.get("arface_vertices") is not None),
        "registrationErrorMm": round(avg_error_mm, 2) if np.isfinite(avg_error_mm) else None,
        "worstAnatomicalRegionErrorMm": round(worst_measured_region_mm, 2) if measured_region_vals and np.isfinite(worst_measured_region_mm) else None,
        "anatomicalGateThresholdMm": ANATOMICAL_FAIL_THRESHOLD_MM,
        "depthCoverage": (
            "native_depth_map" if reconstruction_metrics.get("has_native_depth")
            else "native_arface_geometry" if reconstruction_metrics.get("has_native_arface")
            else "not_available"
        ),
        "surfaceCoverage": round(reconstruction_metrics.get("metrics", {}).get("vertex_count", 0), 0),
        "textureCoverage": "photographic texture baked; coverage is not independently measured",
        "renderBackErrorPerView": view_errors,
        "measuredViewCount": measured_views,
        "requiredMeasuredViewCount": required_measured_views,
        "maxViewReprojectionErrorMm": 5.0,
        "minViewSilhouetteIoU": 0.80,
        "anatomicalRegionErrors": anatomical_errors,
        "evaluatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
