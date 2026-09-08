"""
Phase 0 — regression baseline capture, per the approved re-architecture plan
(/home/ubuntu/.claude/plans/jaunty-tinkering-aho.md).

Runs the CURRENT PRODUCTION pipeline (unmodified — every function imported
here is read-only reused from ai-engine/main.py's own `/fit-multiview-atlas`
handler, same call order, same arguments) against real photos of 5 real
patients already on disk, and snapshots fitted_positions/vertex_colors/
coverage_fraction/warnings/views_used + a software-rendered preview at each
patient's own 0°/45°/90° camera pose. This snapshot is the baseline every
later phase's regression check diffs against — in particular, Phase 1's hard
gate that `face_mask` positions/colors are byte-identical before and after
adding the vertex-trust blend.

Does NOT start the FastAPI server, does NOT call it over HTTP, does NOT
write anywhere under public/ or .data/ — output lives entirely under
ai-engine/regression/baseline/, a new directory this script creates.
"""
import hashlib
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
from gnm_identity_fit import fit_multiview, get_triangles, ViewInput  # noqa: E402
from gnm_dense_nose import enrich_with_dense_nose  # noqa: E402
from gnm_width_correction import apply_width_correction  # noqa: E402
from gnm_texture_bake import compute_vertex_normals, bake_vertex_colors, compute_icm_labels  # noqa: E402
from gnm_texture_atlas import improve_neck_shoulder_vertex_colors  # noqa: E402
from gnm_render_preview import rasterize_preview  # noqa: E402

PATIENTS = [
    "dca63e1a-7dad-4294-bc4b-69afe09f848f",
    "0913e4c9-999c-46cf-9d95-c294dff4bfd3",
    "b24a972d-bf7d-4bed-bbb3-c3732f34a9b6",
    "35e1ffef-d7e5-405d-a7ec-dae152bf883b",
    "23bf4ca6-9ed6-4542-ac0b-c6c20a1685d6",
]

ANGLE_SLOTS = ["angle1", "angle2", "angle3", "angle4"]
RENDER_SLOTS = {"angle1": "render_0.png", "angle2": "render_45.png", "angle3": "render_90.png"}

OUT_ROOT = Path(__file__).resolve().parent / "baseline"


def _find_photo(photos_dir: Path, slot: str) -> Path | None:
    matches = sorted(photos_dir.glob(f"{slot}.*"))
    return matches[0] if matches else None


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


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
        input_files[slot] = {"path": str(path), "md5": _md5(path), "width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])}

    if not images:
        raise RuntimeError(f"{patient_id}: no usable photo found under {photos_dir}")

    # ---- exact replica of main.py's /fit-multiview-atlas handler, steps 1-6 ----
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
    except Exception as exc:  # noqa: BLE001 — same best-effort discipline as production
        warnings.append(f"neck/shoulder color improvement failed, original vertex colors kept: {exc}")

    # ---- persist baseline (Phase 0 deliverables only — atlas intentionally out of scope, see script docstring) ----
    np.save(out_dir / "fitted_positions.npy", fitted_positions.astype(np.float32))
    np.save(out_dir / "vertex_colors.npy", vertex_color.astype(np.float32))
    (out_dir / "coverage_fraction.json").write_text(json.dumps({"coverage_fraction": coverage_fraction}, indent=2))
    (out_dir / "warnings.json").write_text(json.dumps(warnings, indent=2))
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
        cv2.imwrite(str(out_dir / out_name), preview_bgr[:, :, ::-1])  # RGB (from rasterize_preview) -> BGR for imwrite
        renders_written.append(out_name)

    # warnings.json above was written before render step could append more — rewrite with final list
    (out_dir / "warnings.json").write_text(json.dumps(warnings, indent=2))

    elapsed_ms = int((time.time() - t0) * 1000)
    manifest = {
        "patient_id": patient_id,
        "input_files": input_files,
        "views_detected": view_slots,
        "views_used": views_used,
        "coverage_fraction": coverage_fraction,
        "n_warnings": len(warnings),
        "renders_written": renders_written,
        "processing_time_ms": elapsed_ms,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
            print(f"[OK] {patient_id}: views_used={manifest['views_used']} coverage={manifest['coverage_fraction']:.4f} warnings={manifest['n_warnings']} time={manifest['processing_time_ms']}ms")
        except Exception as exc:  # noqa: BLE001 — script-level failure must be visible, never swallowed
            traceback.print_exc()
            results.append({"patient_id": patient_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {patient_id}: {type(exc).__name__}: {exc}")

    (OUT_ROOT / "_summary.json").write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(PATIENTS)} patients captured successfully. Summary: {OUT_ROOT / '_summary.json'}")
    sys.exit(0 if n_ok == len(PATIENTS) else 1)


if __name__ == "__main__":
    main()
