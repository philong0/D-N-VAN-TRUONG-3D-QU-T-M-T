"""
Phase 1 — CORRECTED regression methodology.

The first attempt (capture_baseline.py run, then capture_phase1.py run, as
two SEPARATE process invocations) surfaced two real findings:

1. A genuine Phase 1 bug: blending positions BEFORE bake_vertex_colors/ICM
   changes the vertex NORMAL of any face_mask-boundary vertex that shares a
   triangle with a shell vertex whose position moved (compute_vertex_normals
   sums adjacent face normals) — which changes that face_mask vertex's own
   ICM facing-weight and can flip which source photo it's colored from, even
   though its POSITION is untouched. Fixed in main.py: the position blend
   now runs strictly LAST, after bake/ICM/atlas all consume the original
   (pre-Phase-1) positions/normals — see that file's own comment.

2. A pre-existing, unrelated confound: gnm_dense_nose.py's `enrich_with_
   dense_nose` shells out to server/3ddfa_v2/reconstruct_patient.py with a
   30-second timeout. Whether that subprocess finishes in time depends on
   system load at the moment it runs — confirmed by capture_baseline.py and
   capture_phase1.py (two separate `python3` process runs, each spinning up
   its own PIPNet/YuNet models etc, so real CPU contention differs run to
   run) getting a DIFFERENT enrichment outcome for the SAME patient on
   different runs. This is real, pre-existing production non-determinism —
   Phase 1 must not "fix" it (out of scope, gnm_dense_nose.py is untouched
   per the plan), but a regression check that runs the pipeline twice in two
   separate processes cannot tell that confound apart from a real Phase 1
   regression.

This script fixes the TEST METHODOLOGY (not production code): it runs the
shared prefix (detect_pose -> fit_multiview -> dense-nose enrichment ->
width-correction -> normals -> bake_views -> ICM -> bake_vertex_colors ->
neck/shoulder improve) exactly ONCE per patient, in one process, then
branches into "pre" (=~ what today's production, i.e. Phase 0 baseline,
returns) and "post" (=~ what the now-fixed Phase 1 code in main.py returns)
purely by applying/not-applying smooth_colors_by_trust + blend_positions_by_
trust to that SAME shared result. This is the authoritative Phase 1
regression gate — capture_baseline.py's on-disk output remains the real
Phase 0 deliverable (unchanged), but is no longer used for the byte-identity
diff, for the reason above.
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

OUT_ROOT = Path(__file__).resolve().parent / "compare"


def run_one_patient(patient_id: str) -> dict:
    t0 = time.time()
    photos_dir = REPO_ROOT / ".data" / "patients" / patient_id / "photos"
    out_dir = OUT_ROOT / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    images: dict[str, np.ndarray] = {}
    for slot in ANGLE_SLOTS:
        path = _find_photo(photos_dir, slot)
        if path is None:
            continue
        image_bgr = cv2.imread(str(path))
        if image_bgr is not None:
            images[slot] = image_bgr
    if not images:
        raise RuntimeError(f"{patient_id}: no usable photo")

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
        raise RuntimeError(f"{patient_id}: identity-fit-failed")

    fitted_positions, dense_nose_note = enrich_with_dense_nose(
        fitted_positions, view_inputs, dict(zip(view_slots, view_images))
    )
    if dense_nose_note:
        warnings.append(dense_nose_note)

    fitted_positions, width_note = apply_width_correction(fitted_positions, view_inputs, view_slots)
    if width_note:
        warnings.append(width_note)

    # ---- shared prefix ends here: `fitted_positions` == today's production (Phase 0) output ----
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
        warnings.append(f"neck/shoulder color improvement failed: {exc}")

    # ---- "pre" = Phase 0 equivalent (exactly what main.py returned before Phase 1) ----
    pre_positions = fitted_positions.copy()
    pre_colors = vertex_color.copy()

    # ---- "post" = Phase 1 (now-fixed ordering: color-smooth then position-blend, both LAST) ----
    trust = compute_vertex_trust()
    post_colors = smooth_colors_by_trust(vertex_color, triangles, trust)
    post_positions = blend_positions_by_trust(fitted_positions, get_template_positions(), trust)

    np.save(out_dir / "pre_positions.npy", pre_positions.astype(np.float32))
    np.save(out_dir / "pre_colors.npy", pre_colors.astype(np.float32))
    np.save(out_dir / "post_positions.npy", post_positions.astype(np.float32))
    np.save(out_dir / "post_colors.npy", post_colors.astype(np.float32))

    renders_written = []
    for slot, out_name in RENDER_SLOTS.items():
        pose = view_pose_by_slot.get(slot)
        if pose is None:
            continue
        pre_img = rasterize_preview(pre_positions, triangles, pre_colors, pose["R"], pose["t"], pose["camera_matrix"], pose["image_shape"])
        post_img = rasterize_preview(post_positions, triangles, post_colors, pose["R"], pose["t"], pose["camera_matrix"], pose["image_shape"])
        cv2.imwrite(str(out_dir / f"pre_{out_name}"), pre_img[:, :, ::-1])
        cv2.imwrite(str(out_dir / f"post_{out_name}"), post_img[:, :, ::-1])
        renders_written.append(out_name)

    (out_dir / "warnings.json").write_text(json.dumps(warnings, indent=2))
    manifest = {
        "patient_id": patient_id,
        "views_used": views_used,
        "coverage_fraction": coverage_fraction,
        "n_warnings": len(warnings),
        "renders_written": renders_written,
        "processing_time_ms": int((time.time() - t0) * 1000),
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
            print(f"[OK] {patient_id}: coverage={manifest['coverage_fraction']:.4f} warnings={manifest['n_warnings']}")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            results.append({"patient_id": patient_id, "ok": False, "error": str(exc)})
            print(f"[FAIL] {patient_id}: {exc}")
    (OUT_ROOT / "_summary.json").write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(PATIENTS)} captured. Summary: {OUT_ROOT / '_summary.json'}")
    sys.exit(0 if n_ok == len(PATIENTS) else 1)


if __name__ == "__main__":
    main()
