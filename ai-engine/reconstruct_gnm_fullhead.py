"""
reconstruct_gnm_fullhead.py

Full-head GNM reconstruction (identity-fit + Face Shell topology + unified
texture bake), exported as a real baseline.glb — the SAME fully-fixed
pipeline (E2 boundary-density topology, BUG A/B/C/D fixes, G1-G4
cross-patient + angle3-only neck-depth fixes) validated extensively earlier
in this project's history, now bridged to the newer .glb-based frontend
contract (Canvas3D.tsx already tries `reconstruction/baseline.glb` FIRST).

Unlike the landmark-only RGB-SfM/MediaPipe paths, get_face_shell_topology()
always returns the SAME real GNM template vertex groups (skin_exterior,
ears, eye_sockets, hockey_mask/neck) regardless of what the identity-fit
found — so ears/neck/hairline are always geometrically present, never
landmark-scoped away.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from detect_pose import detect_pose
from gnm_identity_fit import fit_multiview, get_triangles, get_template_positions, ViewInput
from gnm_dense_nose import enrich_with_dense_nose
from gnm_width_correction import apply_width_correction
from gnm_texture_bake import compute_vertex_normals, compute_person_silhouette_mask
from gnm_vertex_trust import compute_vertex_trust, blend_positions_by_trust
from gnm_face_shell import get_face_shell_topology, bake_unified_face_texture
from gnm_profile_silhouette import apply_profile_silhouette_deformation
from patient_texture_baker import export_patient_glb

SLOT_ALIASES = {
    # 5-slot scan-session naming -> the 4-slot angleN convention this
    # pipeline's own is_profile_view/silhouette-mask slot dispatch expects.
    "front": "angle1",
    "left_45": "angle2",
    "left_profile": "angle3",
    "right_45": "angle4",
    "right_profile": "angle3b",  # not a real slot in this pipeline; loaded but treated as an extra oblique-like view below
}


def reconstruct_patient_gnm(patient_id: str, photos_dir: Path, output_dir: Path) -> dict:
    """CLI/standalone entry: load images from a flat photos directory
    (angle1-4 or the 5-slot scan naming, aliased via SLOT_ALIASES), then
    delegate to `reconstruct_gnm_from_images` — the shared core also used
    by reconstruct_cli.py's live new-patient-scan path (see D-gnmprimary
    there) so both entry points run the exact same pipeline, not two
    near-copies that can drift apart."""
    images = {}
    for f in sorted(photos_dir.glob("*")):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        raw_slot = f.stem
        slot = SLOT_ALIASES.get(raw_slot, raw_slot)
        img = cv2.imread(str(f))
        if img is None:
            continue
        # Only ever keep ONE image per canonical angle1-4 slot (first match
        # wins) -- this pipeline's own multiview solver is a fixed 4-view
        # contract; right_profile has no free slot in it (angle3 is already
        # taken by left_profile) so it's intentionally not fed in here.
        if slot in ("angle1", "angle2", "angle3", "angle4") and slot not in images:
            images[slot] = img

    print(f"Loaded slots: {list(images.keys())} from {photos_dir}", flush=True)
    return reconstruct_gnm_from_images(patient_id, images, output_dir)


def reconstruct_gnm_from_images(patient_id: str, images: dict, output_dir: Path) -> dict:
    """Core pipeline: `images` is {angle1..angle4: BGR ndarray} (already
    loaded/aliased by the caller). Runs the real identity-fit + Face Shell
    bake and exports baseline.glb. Returns {"ok": False, "error": ...} on
    any real failure (no face detected, degenerate fit, etc.) rather than
    raising, so a caller (e.g. reconstruct_cli.py) can cleanly fall back to
    a different reconstruction path instead of crashing."""
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if not images:
        return {"ok": False, "error": "no-usable-photo"}

    view_inputs, view_slots, view_images, view_landmarks, warnings = [], [], [], [], []
    for slot, image_bgr in images.items():
        is_profile = slot == "angle3"
        h, w = image_bgr.shape[:2]
        result = detect_pose(image_bgr, is_profile_view=is_profile)
        if not result["has_face"]:
            warnings.append(f"{slot}: no face detected (PIPNet) - dropped")
            continue
        camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        view_inputs.append(ViewInput(result["landmarks_98"], camera_matrix, is_profile, (h, w)))
        view_slots.append(slot)
        view_images.append(image_bgr)
        view_landmarks.append(result["landmarks_98"])

    if len(view_inputs) == 0:
        return {"ok": False, "error": "no-face-detected-in-any-photo", "warnings": warnings}

    # Anatomical slot correction: ensure angle2 is genuine left (yaw < 0) and angle4 is genuine right (yaw > 0)
    # If client passed mirrored / inverted slot tags, correct them automatically:
    slot_to_idx = {s: i for i, s in enumerate(view_slots)}
    if "angle2" in slot_to_idx and "angle4" in slot_to_idx:
        i2, i4 = slot_to_idx["angle2"], slot_to_idx["angle4"]
        # In PIPNet: yaw < 0 is patient facing left, yaw > 0 is patient facing right
        res2 = detect_pose(view_images[i2], is_profile_view=False)
        res4 = detect_pose(view_images[i4], is_profile_view=False)
        y2 = float(res2.get("yaw", 0.0))
        y4 = float(res4.get("yaw", 0.0))
        if y2 > 10.0 and y4 < -10.0:
            # Inverted slots! Swap them cleanly
            view_slots[i2], view_slots[i4] = "angle4", "angle2"
            print(f"[Anatomical Slot Correction] Swapped angle2 (yaw={y2:.1f}°) and angle4 (yaw={y4:.1f}°)", flush=True)

    print(f"Views used for fit: {view_slots}", flush=True)
    fitted_positions, per_view_pose, fit_warnings = fit_multiview(view_inputs)
    warnings.extend(fit_warnings)
    if fitted_positions is None:
        return {"ok": False, "error": "identity-fit-failed", "warnings": warnings}

    fitted_positions, dense_nose_note = enrich_with_dense_nose(fitted_positions, view_inputs, dict(zip(view_slots, view_images)))
    if dense_nose_note:
        warnings.append(dense_nose_note)

    fitted_positions, width_note = apply_width_correction(fitted_positions, view_inputs, view_slots)
    if width_note:
        warnings.append(width_note)

    # Apply authentic Profile Silhouette Deformation ONLY when angle3 is a genuine profile (abs(yaw) >= 55°)
    if "angle3" in view_slots:
        p_idx = view_slots.index("angle3")
        test_pose = detect_pose(view_images[p_idx], is_profile_view=True)
        yaw_mag = abs(float(test_pose.get("yaw", 0.0)))
        if yaw_mag >= 55.0:
            pose_3 = per_view_pose[p_idx]
            if pose_3 is not None:
                R_3, t_3, _ = pose_3
                K_3 = view_inputs[p_idx].camera_matrix
                lms_3 = view_landmarks[p_idx]
                img_3 = view_images[p_idx]
                fitted_positions, silh_note = apply_profile_silhouette_deformation(
                    fitted_positions, img_3, lms_3, R_3, t_3, K_3, profile_slot="angle3"
                )
                if silh_note:
                    warnings.append(silh_note)
        else:
            print(f"[ProfileGuard] angle3 yaw={yaw_mag:.1f}° is not a true profile (>=55° required) - skipped profile silhouette to prevent distortion", flush=True)

    trust = compute_vertex_trust()
    triangles_full = get_triangles()
    normals = compute_vertex_normals(fitted_positions, triangles_full)
    blended_positions = blend_positions_by_trust(fitted_positions, get_template_positions(), trust)

    from gnm_correspondence import VERTEX_INDICES as _GNM_VIDX, WEIGHTS as _GNM_W, WFLW_INDICES as _GNM_WFLW, point_mask as _gnm_point_mask

    view_landmarks_by_slot = dict(zip(view_slots, view_landmarks))
    bake_views = []
    for slot, image_bgr in zip(view_slots, view_images):
        h, w = image_bgr.shape[:2]
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        lms_98 = view_landmarks_by_slot[slot]
        is_profile = (slot == "angle3")
        mask = _gnm_point_mask(is_profile)
        kept_wflw = [idx for idx, keep in zip(_GNM_WFLW, mask) if keep]
        points_2d = np.array([lms_98[i] for i in kept_wflw], dtype=np.float64)
        vidx = np.array(_GNM_VIDX)[mask]
        w_mat = np.array(_GNM_W)[mask]
        pts_3d = np.einsum("nk,nkc->nc", w_mat, blended_positions[vidx])
        ok, rvec, tvec = cv2.solvePnP(pts_3d, points_2d, K, np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(pts_3d, points_2d, K, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
        else:
            continue
        person_mask = compute_person_silhouette_mask(lms_98, (h, w), slot=slot)
        bake_views.append({"image": image_bgr, "R": R, "t": t, "camera_matrix": K, "person_mask": person_mask, "landmarks_98": lms_98, "slot": slot})

    if not bake_views:
        return {"ok": False, "error": "no-valid-pose-for-any-view", "warnings": warnings}

    print(f"Baking Face Shell texture from {len(bake_views)} posed views...", flush=True)
    shell_vids, shell_triangles, shell_uvs, _tpl_pos = get_face_shell_topology()
    face_hd_bgr, face_hd_png_bytes, debug_maps = bake_unified_face_texture(
        blended_positions, normals, bake_views, tex_size=2048
    )

    shell_positions = blended_positions[shell_vids].astype(np.float64)

    glb_path = output_dir / "baseline.glb"
    export_patient_glb(shell_positions, shell_triangles.astype(np.int64), shell_uvs.astype(np.float64), face_hd_png_bytes, str(glb_path))

    with open(output_dir / "face_HD.png", "wb") as f:
        f.write(face_hd_png_bytes)
    with open(output_dir / "shell_topology.json", "w") as f:
        json.dump({
            "shellVertexIndices": shell_vids.tolist(),
            "triangles": shell_triangles.tolist(),
            "uvs": shell_uvs.tolist(),
        }, f)
    with open(output_dir / "coverage_stats.json", "w") as f:
        json.dump(debug_maps["stats"], f, indent=2)
    blended_positions.astype("<f4").tofile(output_dir / "positions.f32")

    elapsed_ms = int((time.time() - t0) * 1000)
    baseline_meta = {
        "patientId": patient_id,
        "modelVersion": "GNM Full-Head Identity-Fit + Face Shell (E2 topology, BUG A-D + G1-G4 fixes)",
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processingTimeMs": elapsed_ms,
        "vertexCount": int(len(shell_vids)),
        "triangleCount": int(len(shell_triangles)),
        "viewsUsed": view_slots,
        "warnings": warnings,
        "coverageStats": debug_maps["stats"],
        "reconstructionStatus": "completed",
    }
    with open(output_dir / "baseline.json", "w") as f:
        json.dump(baseline_meta, f, indent=2)

    print(json.dumps(baseline_meta, indent=2))
    return {"ok": True, "baselineGlbPath": str(glb_path), "baselineMeta": baseline_meta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--photos-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = reconstruct_patient_gnm(args.patient_id, Path(args.photos_dir), Path(args.output_dir))
    if not result.get("ok"):
        print(json.dumps(result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
