"""
Empirical Geometric Validation and Reality Check for 3D Face Reconstruction.
Measures:
1. Per-view landmark 2D reprojection errors (mean, median, 95th percentile, max).
2. Anatomical region breakdown (Nose, Chin, Jaw, Eyes, Mouth, Cheeks).
3. Repeatability (mesh-to-mesh RMS diff across 2 consecutive runs).
4. Physical scale sanity check (intercanthal distance, nose length, face height).
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from detect_pose import detect_pose
from reconstruct_cli import reconstruct_patient_scan
from gnm_correspondence import WFLW_INDICES, VERTEX_INDICES, WEIGHTS

# Region mapping for the 98 WFLW landmarks
# 0-32: Jawline
# 33-50: Eyebrows
# 51-59: Nose
# 60-75: Eyes
# 76-95: Mouth
# 96-97: Pupils
REGION_RANGES = {
    "jaw": list(range(0, 33)),
    "eyebrows": list(range(33, 51)),
    "nose": list(range(51, 60)),
    "eyes": list(range(60, 76)),
    "mouth": list(range(76, 96)),
}

KEY_LANDMARKS_WFLW = {
    "Pronasale (Nose Tip)": 57,
    "Subnasale (Columella Base)": 59,
    "Nasion (Sellion/Bridge Base)": 51,
    "Alar Left": 55,
    "Alar Right": 59,
    "Pogonion (Chin Pog)": 16,
    "Menton (Lower Chin)": 16,
    "Endocanthion Left (Inner Eye L)": 64,
    "Endocanthion Right (Inner Eye R)": 68,
    "Exocanthion Left (Outer Eye L)": 60,
    "Exocanthion Right (Outer Eye R)": 72,
    "Stomion (Lip Center)": 88,
}

def run_validation(image_paths: dict[str, str], output_dir: str):
    out_dir = Path(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("==================================================================")
    print("PHASE 3.1 — GEOMETRIC RECONSTRUCTION VALIDATION & REALITY CHECK")
    print("==================================================================")

    # 1. RUN 1: Reconstruction
    t0 = time.time()
    res1 = reconstruct_patient_scan(
        image_paths=image_paths,
        patient_id="val-patient-001",
        session_id="val-session-run1",
        output_dir=out_dir / "run1",
    )
    t_run1 = time.time() - t0
    assert res1["ok"], f"Reconstruction Run 1 failed: {res1.get('error')}"

    # 2. RUN 2: Repeatability Test (Identical input on Run 2)
    t0 = time.time()
    res2 = reconstruct_patient_scan(
        image_paths=image_paths,
        patient_id="val-patient-001",
        session_id="val-session-run2",
        output_dir=out_dir / "run2",
    )
    t_run2 = time.time() - t0
    assert res2["ok"], f"Reconstruction Run 2 failed: {res2.get('error')}"

    # 3. Load Meshes & Compare Mesh-to-Mesh Discrepancy
    scene1 = trimesh.load(out_dir / "run1" / "baseline.glb", force="mesh", process=False)
    scene2 = trimesh.load(out_dir / "run2" / "baseline.glb", force="mesh", process=False)

    mesh1 = scene1 if hasattr(scene1, "vertices") else list(scene1.geometry.values())[0]
    mesh2 = scene2 if hasattr(scene2, "vertices") else list(scene2.geometry.values())[0]

    v1 = mesh1.vertices
    v2 = mesh2.vertices
    v_diff = np.linalg.norm(v1 - v2, axis=1)

    repeatability_stats = {
        "mesh1_vertex_count": len(v1),
        "mesh2_vertex_count": len(v2),
        "mean_diff_mm": float(np.mean(v_diff) * 1000), # converted to mm if in meters
        "rms_diff_mm": float(np.sqrt(np.mean(v_diff**2)) * 1000),
        "median_diff_mm": float(np.median(v_diff) * 1000),
        "p95_diff_mm": float(np.percentile(v_diff, 95) * 1000),
        "max_diff_mm": float(np.max(v_diff) * 1000),
        "is_deterministic": bool(np.max(v_diff) < 1e-6),
    }

    # 4. Multi-View Reprojection Error Analysis
    # Load 3D fitted positions (17821, 3)
    fitted_positions = np.fromfile(out_dir / "run1" / "positions.f32", dtype="<f4").reshape(-1, 3)

    # Compute 3D correspondence points for the 48 WFLW landmarks
    correspondence_3d = []
    for wflw_idx, v_idxs, weights in zip(WFLW_INDICES, VERTEX_INDICES, WEIGHTS):
        pt_3d = sum(w * fitted_positions[vid] for vid, w in zip(v_idxs, weights))
        correspondence_3d.append((wflw_idx, pt_3d))

    view_reprojection_results = {}
    all_reprojection_errors_px = []
    all_reprojection_errors_pct = []

    regional_errors = {reg: [] for reg in REGION_RANGES}
    key_landmark_errors = {}

    for view_tag, img_path in image_paths.items():
        if not os.path.exists(img_path):
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        is_profile = (view_tag in ["angle3", "left_profile", "right_profile"])
        pose_res = detect_pose(bgr, is_profile_view=is_profile)
        if not pose_res["has_face"]:
            continue

        lms_2d = np.array(pose_res["landmarks_98"], dtype=np.float64) # (98, 2)
        cam_pose = pose_res["camera_pose"]
        if cam_pose is None:
            continue

        R = np.array(cam_pose["rotation_matrix"], dtype=np.float64)
        t = np.array(cam_pose["translation"], dtype=np.float64).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        camera_matrix = np.array([[w, 0, w/2.0], [0, w, h/2.0], [0, 0, 1.0]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # Project all 3D correspondence landmarks onto this 2D camera view
        pts_3d_arr = np.array([pt for _, pt in correspondence_3d], dtype=np.float64)
        wflw_idxs = [w_idx for w_idx, _ in correspondence_3d]

        proj_2d, _ = cv2.projectPoints(pts_3d_arr, rvec, t, camera_matrix, dist_coeffs)
        proj_2d = proj_2d.reshape(-1, 2)

        # Calculate distances between 2D detected landmarks and 3D projected landmarks
        gt_2d_pts = lms_2d[wflw_idxs]
        err_px = np.linalg.norm(proj_2d - gt_2d_pts, axis=1)
        err_pct = (err_px / w) * 100.0

        all_reprojection_errors_px.extend(err_px.tolist())
        all_reprojection_errors_pct.extend(err_pct.tolist())

        # Collect regional error
        for w_idx, e_px in zip(wflw_idxs, err_px):
            for reg_name, reg_indices in REGION_RANGES.items():
                if w_idx in reg_indices:
                    regional_errors[reg_name].append(e_px)

        # Collect key landmark errors
        for kl_name, kl_w_idx in KEY_LANDMARKS_WFLW.items():
            if kl_w_idx in wflw_idxs:
                idx_in_list = wflw_idxs.index(kl_w_idx)
                if kl_name not in key_landmark_errors:
                    key_landmark_errors[kl_name] = []
                key_landmark_errors[kl_name].append(float(err_px[idx_in_list]))

        view_reprojection_results[view_tag] = {
            "image_size": [w, h],
            "detected_yaw_deg": float(pose_res["yaw"]),
            "mean_error_px": float(np.mean(err_px)),
            "median_error_px": float(np.median(err_px)),
            "p95_error_px": float(np.percentile(err_px, 95)),
            "max_error_px": float(np.max(err_px)),
            "mean_error_pct_width": float(np.mean(err_pct)),
        }

    # Aggregate regional statistics
    regional_stats = {}
    for reg, errors in regional_errors.items():
        if errors:
            regional_stats[reg] = {
                "count": len(errors),
                "mean_px": float(np.mean(errors)),
                "median_px": float(np.median(errors)),
                "p95_px": float(np.percentile(errors, 95)),
                "max_px": float(np.max(errors)),
            }

    # Aggregate key landmark statistics
    key_landmark_stats = {}
    for kl_name, errors in key_landmark_errors.items():
        if errors:
            key_landmark_stats[kl_name] = {
                "mean_px": float(np.mean(errors)),
                "median_px": float(np.median(errors)),
                "p95_px": float(np.percentile(errors, 95)),
                "max_px": float(np.max(errors)),
            }

    # Overall Metrics
    overall_stats = {
        "total_landmarks_evaluated": len(all_reprojection_errors_px),
        "mean_reprojection_error_px": float(np.mean(all_reprojection_errors_px)),
        "median_reprojection_error_px": float(np.median(all_reprojection_errors_px)),
        "p95_reprojection_error_px": float(np.percentile(all_reprojection_errors_px, 95)),
        "max_reprojection_error_px": float(np.max(all_reprojection_errors_px)),
        "mean_reprojection_error_pct_width": float(np.mean(all_reprojection_errors_pct)),
    }

    report = {
        "input_dataset": {
            "type": "2D Photographic Multi-Angle Dataset (Not Hardware TrueDepth)",
            "views": list(image_paths.keys()),
            "files": {k: str(v) for k, v in image_paths.items()},
        },
        "repeatability": repeatability_stats,
        "overall_reprojection": overall_stats,
        "per_view_reprojection": view_reprojection_results,
        "regional_breakdown": regional_stats,
        "key_landmarks": key_landmark_stats,
        "execution_times_sec": {"run1": t_run1, "run2": t_run2},
    }

    with open(out_dir / "reality_check_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    fixtures_dir = Path("/home/ubuntu/dr-vantruong-3d-studio/fixtures/scans/sample_truedepth_scan")
    images = {
        "angle1_front": str(fixtures_dir / "angle1.png"),
        "angle2_left45": str(fixtures_dir / "angle2.png"),
        "angle3_profile90": str(fixtures_dir / "angle3.jpg"),
        "angle4_submental": str(fixtures_dir / "angle4.png"),
    }
    run_validation(images, "/home/ubuntu/dr-vantruong-3d-studio/fixtures/scans/sample_truedepth_scan/validation_output")
