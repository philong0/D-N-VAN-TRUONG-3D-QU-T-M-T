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
    # 5-slot scan-session naming -> canonical angle slots
    "front": "angle1",
    "left_45": "angle2",
    "left_oblique": "angle2",
    "left_profile": "angle3",
    "left_lateral": "angle3",
    "profile": "angle3",
    "right_45": "angle4",
    "right_oblique": "angle4",
    "right_profile": "angle5",
    "right_lateral": "angle5",
    "basal_nostrils": "angle1",
    "sweep_00": "angle1",
    "sweep_01": "angle2",
    "sweep_02": "angle2",
    "sweep_03": "angle3",
    "sweep_04": "angle4",
    "sweep_05": "angle4",
    "sweep_06": "angle5",
}


def resolve_slot(filename: str) -> str | None:
    stem = Path(filename).stem
    if stem in ("angle1", "angle2", "angle3", "angle4", "angle5"):
        return stem
    if stem in SLOT_ALIASES:
        return SLOT_ALIASES[stem]
    for alias_key, canon_slot in SLOT_ALIASES.items():
        if stem.endswith(f"_{alias_key}") or stem == alias_key:
            return canon_slot
    return None


def smooth_mesh_surface(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Boundary-loop relaxation only: eliminates jagged sawtooth steps along
    the hairline/neck cut edge, WITHOUT touching interior surface detail.

    2026-09-14 fix -- this function used to also run a whole-mesh interior
    Laplacian fairing pass (`iterations`/`factor` args, unvalidated: no
    before/after measurement exists anywhere in this codebase, unlike every
    other numeric constant in the reconstruction pipeline, e.g.
    gnm_dense_nose.py's own swept W=15->200 table). It ran unconditionally,
    on every patient, immediately AFTER `enrich_with_dense_nose()` and
    `apply_width_correction()` -- i.e. right after the two modules
    specifically written to ADD real nose-bridge/tip and cheek/temple
    surface detail back in (see PROJECT_DIAGNOSIS.md's landmark-sparsity
    root cause). A uniform neighbor-averaging pass is structurally exactly
    what un-does that: it has no way to distinguish "real fitted detail worth
    keeping" from "noise worth smoothing," so it silently flattened some of
    the very detail those upstream fixes just added -- the user explicitly
    requires geometry to come from the real fitted photos, not from an
    unmeasured post-hoc smoothing guess. Removed the interior pass; kept only
    the boundary-loop relaxation below, which is scoped to the mesh's own cut
    edge (hairline/neck) and never touches face-interior vertices at all, so
    it cannot affect nose/cheek/chin surface detail either way.
    """
    from collections import defaultdict

    edge_count = defaultdict(int)

    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        edges = [(min(i0, i1), max(i0, i1)), (min(i1, i2), max(i1, i2)), (min(i2, i0), max(i2, i0))]
        for e in edges:
            edge_count[e] += 1

    # Find boundary loop edges & vertices
    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    b_adj = defaultdict(set)
    for u, v in boundary_edges:
        b_adj[u].add(v)
        b_adj[v].add(u)
    b_verts = list(b_adj.keys())

    smoothed = positions.copy()

    # Boundary loop smoothing: eliminates jagged sawtooth steps along hairline and neck cut edge only
    for _ in range(6):
        b_delta = np.zeros_like(smoothed)
        for v in b_verts:
            nbrs = list(b_adj[v])
            if len(nbrs) >= 2:
                b_delta[v] = 0.45 * (smoothed[nbrs].mean(axis=0) - smoothed[v])
        smoothed += b_delta

    return smoothed


def level_facial_symmetry(shell_positions: np.ndarray, shell_vids: np.ndarray) -> np.ndarray:
    """Anatomical horizontal leveling for interpupillary line and oral commissures.
    Guarantees true clinical horizontal symmetry regardless of patient head tilt during capture."""
    gnm = np.load(Path(__file__).parent.parent / "public" / "models" / "gnm" / "gnm_head_v3.npz", allow_pickle=True)
    vgroups = gnm["vertex_groups"]
    gnames = [str(n) for n in gnm["vertex_group_names"]]
    tpl_pos = gnm["template_vertex_positions"]

    leveled = shell_positions.copy()

    # 1. Interpupillary Line Leveling (rigid rotation around Z-axis)
    eyes_mask = (vgroups[gnames.index("eyes")] > 0.5)[shell_vids]
    left_eye = eyes_mask & (tpl_pos[shell_vids, 0] < -0.015)
    right_eye = eyes_mask & (tpl_pos[shell_vids, 0] > 0.015)

    if left_eye.any() and right_eye.any():
        c_left = leveled[left_eye].mean(axis=0)
        c_right = leveled[right_eye].mean(axis=0)
        eye_mid = (c_left + c_right) / 2.0
        # Roll angle: roll = arctan2(dy, dx)
        roll = np.arctan2(c_right[1] - c_left[1], c_right[0] - c_left[0])
        cos_r, sin_r = np.cos(-roll), np.sin(-roll)
        R_z = np.array([[cos_r, -sin_r, 0.0], [sin_r, cos_r, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        leveled = (leveled - eye_mid) @ R_z.T + eye_mid

    # 2. Oral Commissure Symmetry Leveling
    shell_pos = tpl_pos[shell_vids]
    d_left = np.sum((shell_pos - [-0.025, 0.231, 0.119]) ** 2, axis=1)
    s_left = int(np.argmin(d_left))
    d_right = np.sum((shell_pos - [0.026, 0.230, 0.118]) ** 2, axis=1)
    s_right = int(np.argmin(d_right))

    p_ml = leveled[s_left]
    p_mr = leveled[s_right]
    mid_x_m = (p_ml[0] + p_mr[0]) / 2.0
    dx_m = p_ml[0] - p_mr[0]
    dy_m = p_ml[1] - p_mr[1]
    dz_m = p_ml[2] - p_mr[2]

    if abs(dx_m) > 1e-4 and abs(dy_m) > 1e-5:
        slope_y = dy_m / dx_m
        slope_z = dz_m / dx_m
        mouth_mask = (
            (np.abs(leveled[:, 0] - mid_x_m) < 0.038)
            & (leveled[:, 1] >= 0.210)
            & (leveled[:, 1] <= 0.255)
            & (leveled[:, 2] > 0.088)
        )
        if mouth_mask.any():
            dist_x = np.abs(leveled[mouth_mask, 0] - mid_x_m) / 0.026
            dist_y = np.abs(leveled[mouth_mask, 1] - 0.233) / 0.022
            w_x = np.clip(1.0 - np.maximum(0.0, dist_x - 1.0) / 0.55, 0.0, 1.0)
            w_x = w_x * w_x * (3.0 - 2.0 * w_x)
            w_y = np.clip(1.0 - dist_y**2, 0.0, 1.0)
            w_m = w_x * w_y
            leveled[mouth_mask, 1] -= slope_y * (leveled[mouth_mask, 0] - mid_x_m) * w_m
            leveled[mouth_mask, 2] -= slope_z * (leveled[mouth_mask, 0] - mid_x_m) * w_m

    return leveled


def reconstruct_patient_gnm(patient_id: str, photos_dir: Path, output_dir: Path) -> dict:
    """CLI/standalone entry: load images from a flat photos directory
    (angle1-5 or the 5-slot scan naming, aliased via SLOT_ALIASES), then
    delegate to `reconstruct_gnm_from_images`."""
    images = {}
    candidate_dirs = [
        photos_dir,
        photos_dir / "frames",
        photos_dir / "frames" / "clinical",
        photos_dir.parent / "photos",
        photos_dir.parent.parent / "photos",
        Path(f"/home/ubuntu/dr-vantruong-3d-studio/.data/patients/{patient_id}/photos"),
        Path(f"/home/ubuntu/dr-vantruong-3d-studio/public/models/patients/{patient_id}/photos"),
    ]

    # Include all scan sessions if available
    for parent_base in [photos_dir.parent, Path(f"/home/ubuntu/dr-vantruong-3d-studio/.data/patients/{patient_id}")]:
        scans_base = parent_base / "scans"
        if scans_base.exists():
            for sdir in sorted(scans_base.glob("*")):
                if sdir.is_dir():
                    candidate_dirs.append(sdir / "frames")
                    candidate_dirs.append(sdir / "frames" / "clinical")

    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        for f in sorted(cdir.glob("*")):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            slot = resolve_slot(f.name)
            if slot and slot not in images:
                img = cv2.imread(str(f))
                if img is not None:
                    images[slot] = img
                    print(f"Loaded {slot} from {f}", flush=True)

    # If still missing some angles, load any sweep_*.jpg frames as auxiliary views
    if len(images) < 4:
        for cdir in candidate_dirs:
            if not cdir.exists():
                continue
            for f in sorted(cdir.glob("sweep_*.jpg")):
                raw_stem = f.stem
                if raw_stem not in images:
                    img = cv2.imread(str(f))
                    if img is not None:
                        images[raw_stem] = img
                        print(f"Loaded sweep frame {raw_stem} from {f}", flush=True)

    below_photo_path = None
    below_orientation_path = None
    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        for candidate_stem in ("basal_nostrils", "below", "sweep_01"):
            candidate_img = cdir / f"{candidate_stem}.jpg"
            candidate_json = cdir / f"{candidate_stem}_orientation.json"
            if candidate_img.exists() and candidate_json.exists():
                below_photo_path = candidate_img
                below_orientation_path = candidate_json
                break
        if below_photo_path:
            break

    print(f"Loaded slots: {list(images.keys())} for patient {patient_id}", flush=True)
    return reconstruct_gnm_from_images(
        patient_id, images, output_dir,
        below_photo_path=below_photo_path, below_orientation_path=below_orientation_path,
    )



def reconstruct_gnm_from_images(
    patient_id: str, images: dict, output_dir: Path,
    below_photo_path: Path | None = None, below_orientation_path: Path | None = None,
) -> dict:
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

    view_inputs, view_slots, view_images, view_landmarks, view_yaws, warnings = [], [], [], [], [], []
    for raw_slot, image_bgr in images.items():
        h, w = image_bgr.shape[:2]
        # Measure true yaw angle first
        raw_res = detect_pose(image_bgr, is_profile_view=False)
        if not raw_res["has_face"]:
            warnings.append(f"{raw_slot}: no face detected (PIPNet) - dropped")
            continue

        yaw_deg = float(raw_res.get("yaw", 0.0))
        yaw_abs = abs(yaw_deg)

        # Automatic anatomical slot classification based on real head yaw:
        # A photo with abs(yaw) < 50° is an oblique view (angle2/4), NEVER a profile (angle3/5).
        if yaw_abs <= 12.0:
            canon_slot = "angle1"
        elif yaw_deg < -12.0:
            canon_slot = "angle3" if yaw_deg <= -50.0 else "angle2"
        else:
            canon_slot = "angle5" if yaw_deg >= 50.0 else "angle4"

        is_profile = canon_slot in ("angle3", "angle5") and yaw_abs >= 60.0
        if is_profile:
            result = detect_pose(image_bgr, is_profile_view=True)
        else:
            result = raw_res

        camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        view_inputs.append(ViewInput(result["landmarks_98"], camera_matrix, is_profile, (h, w)))
        view_slots.append(canon_slot)
        view_images.append(image_bgr)
        view_landmarks.append(result["landmarks_98"])
        view_yaws.append(yaw_deg)

    if len(view_inputs) == 0:
        return {"ok": False, "error": "no-face-detected-in-any-photo", "warnings": warnings}

    print(f"Views used for fit (anatomically assigned): {view_slots}", flush=True)
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

    bake_views = []
    for slot, image_bgr, lms_98, v_yaw in zip(view_slots, view_images, view_landmarks, view_yaws):
        h, w = image_bgr.shape[:2]
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        is_profile = slot in ("angle3", "angle5")
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
        bake_views.append({"image": image_bgr, "R": R, "t": t, "camera_matrix": K, "person_mask": person_mask, "landmarks_98": lms_98, "slot": slot, "yaw": v_yaw})

    # 2026-09-14 -- optional "below" (chin-underside) view. Deliberately
    # posed WITHOUT detect_pose()/solvePnP-on-98-landmarks (that path needs
    # a normal frontal/profile face structure, which a true underside-of-
    # chin photo does not have) and NEVER added to `view_inputs`/
    # fit_multiview above -- it can only ever affect texture, never the 3D
    # geometry. See chin_underside_anchor.py's own docstring for the full
    # real-evidence chain (sensor rotation + detected nostril/chin anchors +
    # closed-form translation solve) and its explicit "drop rather than
    # guess" contract: every failure branch below appends a warning and
    # simply does not add this view, exactly like the EPnP-failure "continue"
    # a few lines above for the normal views.
    if below_photo_path is not None and below_orientation_path is not None:
        try:
            from chin_underside_anchor import detect_nostril_chin_anchors_2d, solve_translation_given_rotation, compose_below_rotation

            below_img = cv2.imread(str(below_photo_path))
            with open(below_orientation_path, "r") as f:
                below_orient = json.load(f)

            front_idx = view_slots.index("angle1") if "angle1" in view_slots else None
            R_below = None
            if not below_orient.get("orientationAvailable"):
                warnings.append("below: no real device-orientation reading available for this capture - dropped (never guessed)")
            elif not below_orient.get("cameraFacingMatches", True):
                warnings.append("below: camera was flipped (front/rear) between the front and below checkpoints - relative sensor rotation would be invalid, dropped rather than guessed")
            elif front_idx is None:
                warnings.append("below: no real solved pose for the front view to anchor against - dropped")
            else:
                R_front_head_frame, _, _ = per_view_pose[front_idx]
                R_below = compose_below_rotation(
                    below_orient["frontOrientation"], below_orient["belowOrientation"], R_front_head_frame,
                )

            if below_img is None:
                warnings.append("below: image file unreadable - dropped")
            elif R_below is None:
                pass  # already warned above
            else:
                anchors = detect_nostril_chin_anchors_2d(below_img)
                if anchors is None:
                    warnings.append("below: nostril/chin anchor detection did not reach confidence threshold - dropped")
                else:
                    hb, wb = below_img.shape[:2]
                    K_below = np.array([[wb, 0, wb / 2], [0, wb, hb / 2], [0, 0, 1]], dtype=np.float64)
                    t_below = solve_translation_given_rotation(anchors, R_below, K_below)
                    if t_below is None:
                        warnings.append("below: translation solve failed or was geometrically inconsistent - dropped")
                    else:
                        bake_views.append({
                            "image": below_img, "R": R_below, "t": t_below, "camera_matrix": K_below,
                            "person_mask": None, "landmarks_98": None, "slot": "below",
                        })
                        print(f"[below] chin-underside view posed and added (anchor confidence={anchors['confidence']:.2f})", flush=True)
        except Exception as e:
            warnings.append(f"below: unexpected error ({e}) - dropped (never guessed a fallback pose)")

    if not bake_views:
        return {"ok": False, "error": "no-valid-pose-for-any-view", "warnings": warnings}

    print(f"Baking Face Shell texture from {len(bake_views)} posed views...", flush=True)
    shell_vids, shell_triangles, shell_uvs, _tpl_pos = get_face_shell_topology()
    face_hd_bgr, face_hd_png_bytes, debug_maps = bake_unified_face_texture(
        blended_positions, normals, bake_views, tex_size=2048
    )

    shell_positions = blended_positions[shell_vids].astype(np.float64)
    shell_positions = smooth_mesh_surface(shell_positions, shell_triangles)
    shell_positions = level_facial_symmetry(shell_positions, shell_vids)

    glb_path = output_dir / "baseline.glb"
    export_patient_glb(shell_positions, shell_triangles.astype(np.int64), shell_uvs.astype(np.float64), face_hd_png_bytes, str(glb_path))

    try:
        import trimesh
        obj_path = output_dir / "baseline.obj"
        trimesh.Trimesh(vertices=shell_positions, faces=shell_triangles, process=False).export(str(obj_path))
    except Exception as e:
        print(f"[Warning] Failed to export baseline.obj: {e}", flush=True)

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

    output_result = {"ok": True, "baselineGlbPath": str(glb_path), "baselineMeta": baseline_meta}
    print(json.dumps(output_result, indent=2))
    return output_result


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
