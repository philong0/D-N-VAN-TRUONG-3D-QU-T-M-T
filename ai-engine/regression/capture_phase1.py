"""
Phase 1 capture — identical to capture_baseline.py's call sequence, with the
two new Phase 1 steps (trust-blend on positions, trust-based color
smoothing) inserted at the exact same points main.py's `/fit-multiview-atlas`
handler now has them (see that file's diff). This exists so Phase 1's
regression check (verify_phase1.py) can diff against Phase 0's baseline
without needing a running FastAPI server — same offline/read-only discipline
as Phase 0.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_ENGINE_DIR.parent
sys.path.insert(0, str(AI_ENGINE_DIR))

from detect_pose import detect_pose  # noqa: E402
from gnm_identity_fit import fit_multiview, get_triangles, get_template_positions, ViewInput  # noqa: E402
from gnm_dense_nose import enrich_with_dense_nose  # noqa: E402
from gnm_width_correction import apply_width_correction  # noqa: E402
from gnm_texture_bake import compute_vertex_normals, bake_vertex_colors, compute_icm_labels  # noqa: E402
from gnm_texture_atlas import improve_neck_shoulder_vertex_colors  # noqa: E402
from gnm_vertex_trust import compute_vertex_trust, blend_positions_by_trust, smooth_colors_by_trust, compute_face_mask  # noqa: E402
from gnm_render_preview import rasterize_preview  # noqa: E402

from capture_baseline import PATIENTS, ANGLE_SLOTS, RENDER_SLOTS, _find_photo, _md5  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "phase1"


def run_one_patient(patient_id: str) -> dict:
    t0 = time.time()
    photos_dir = REPO_ROOT / ".data" / "patients" / patient_id / "photos"
    out_dir = OUT_ROOT / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    input_files: dict[str, dict] = {}
    images: dict[str, np.ndarray] = {}

    for slot in ANGLE_SLOTS:
        path = _find_photo(photos_dir, slot)
        if path is None:
            continue
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            warnings.append(f"{slot}: unreadable image file at {path}")
            continue
        images[slot] = image_bgr
        input_files[slot] = {"path": str(path), "md5": _md5(path)}

    if not images:
        raise RuntimeError(f"{patient_id}: no usable photo found under {photos_dir}")

    view_inputs, view_slots, view_images = [], [], []
    for slot, image_bgr in images.items():
        is_profile = slot == "angle3"
        h, w = image_bgr.shape[:2]
        result = detect_pose(image_bgr, is_profile_view=is_profile)
        if not result["has_face"]:
            warnings.append(f"{slot}: no face detected (PIPNet) — dropped")
            continue
        camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        view_inputs.append(ViewInput(result["landmarks_98"], camera_matrix, is_profile, (h, w)))
        view_slots.append(slot)
        view_images.append(image_bgr)

    if not view_inputs:
        raise RuntimeError(f"{patient_id}: no face detected in any photo")

    fitted_positions, per_view_pose, fit_warnings = fit_multiview(view_inputs)
    warnings.extend(fit_warnings)
    if fitted_positions is None:
        raise RuntimeError(f"{patient_id}: identity-fit-failed — {fit_warnings}")

    fitted_positions, dense_nose_note = enrich_with_dense_nose(
        fitted_positions, view_inputs, dict(zip(view_slots, view_images))
    )
    if dense_nose_note:
        warnings.append(dense_nose_note)

    fitted_positions, width_note = apply_width_correction(fitted_positions, view_inputs, view_slots)
    if width_note:
        warnings.append(width_note)

    # --- Phase 1 step 1: trust-blend positions (matches main.py's new step) ---
    trust = compute_vertex_trust()
    fitted_positions = blend_positions_by_trust(fitted_positions, get_template_positions(), trust)

    triangles = get_triangles()
    normals = compute_vertex_normals(fitted_positions, triangles)

    bake_views, views_used, view_pose_by_slot = [], [], {}
    for slot, image_bgr, pose in zip(view_slots, view_images, per_view_pose):
        if pose is None:
            continue
        R, t, _mask = pose
        h, w = image_bgr.shape[:2]
        camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        bake_views.append({"image": image_bgr, "R": R, "t": t, "camera_matrix": camera_matrix})
        views_used.append(slot)
        view_pose_by_slot[slot] = {"R": R, "t": t, "camera_matrix": camera_matrix, "image_shape": (h, w)}

    icm_result = compute_icm_labels(fitted_positions, normals, bake_views, triangles)
    vertex_color, coverage_fraction, per_view_weight = bake_vertex_colors(
        fitted_positions, normals, bake_views, triangles, icm_result=icm_result
    )

    try:
        vertex_color = improve_neck_shoulder_vertex_colors(fitted_positions, triangles, vertex_color, per_view_weight)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"neck/shoulder color improvement failed, original vertex colors kept: {exc}")

    # --- Phase 1 step 2: trust-based color smoothing (matches main.py's new step) ---
    try:
        vertex_color = smooth_colors_by_trust(vertex_color, triangles, trust)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"trust-based color smoothing failed, pre-smoothing vertex colors kept: {exc}")

    np.save(out_dir / "fitted_positions.npy", fitted_positions.astype(np.float32))
    np.save(out_dir / "vertex_colors.npy", vertex_color.astype(np.float32))
    (out_dir / "coverage_fraction.json").write_text(json.dumps({"coverage_fraction": coverage_fraction}, indent=2))
    (out_dir / "views_used.json").write_text(json.dumps({"views_used": views_used, "views_detected": view_slots}, indent=2))

    renders_written = []
    for slot, out_name in RENDER_SLOTS.items():
        pose = view_pose_by_slot.get(slot)
        if pose is None:
            warnings.append(f"render {out_name}: no usable pose for {slot} — skipped, not fabricated")
            continue
        preview_bgr = rasterize_preview(
            fitted_positions, triangles, vertex_color,
            pose["R"], pose["t"], pose["camera_matrix"], pose["image_shape"],
        )
        cv2.imwrite(str(out_dir / out_name), preview_bgr[:, :, ::-1])
        renders_written.append(out_name)

    (out_dir / "warnings.json").write_text(json.dumps(warnings, indent=2))

    elapsed_ms = int((time.time() - t0) * 1000)
    manifest = {
        "patient_id": patient_id,
        "views_detected": view_slots,
        "views_used": views_used,
        "coverage_fraction": coverage_fraction,
        "n_warnings": len(warnings),
        "renders_written": renders_written,
        "processing_time_ms": elapsed_ms,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for patient_id in PATIENTS:
        try:
            manifest = run_one_patient(patient_id)
            results.append({"patient_id": patient_id, "ok": True, **manifest})
            print(f"[OK] {patient_id}: coverage={manifest['coverage_fraction']:.4f} warnings={manifest['n_warnings']} time={manifest['processing_time_ms']}ms")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            results.append({"patient_id": patient_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {patient_id}: {type(exc).__name__}: {exc}")

    (OUT_ROOT / "_summary.json").write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(PATIENTS)} patients captured successfully (Phase 1). Summary: {OUT_ROOT / '_summary.json'}")
    sys.exit(0 if n_ok == len(PATIENTS) else 1)


if __name__ == "__main__":
    main()
