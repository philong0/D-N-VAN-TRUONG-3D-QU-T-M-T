import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional

from detect_pose import detect_pose
from gnm_identity_fit import fit_multiview, get_triangles, get_template_positions, ViewInput
from gnm_dense_nose import enrich_with_dense_nose
from gnm_width_correction import apply_width_correction
from gnm_texture_bake import compute_vertex_normals, bake_vertex_colors, compute_icm_labels, compute_person_silhouette_mask, SENTINEL_UNCOVERED
from gnm_texture_atlas import build_multi_region_atlas, improve_neck_shoulder_vertex_colors
from gnm_face_shell import get_face_shell_topology, bake_unified_face_texture
from gnm_vertex_trust import (
    compute_vertex_trust,
    blend_positions_by_trust,
    smooth_colors_by_trust,
    compute_region_index,
    compute_adaptive_shell_tone,
    REGION_NECK,
    REGION_HEAD_SHELL_GENERIC,
)
from gnm_correspondence import WFLW_INDICES, VERTEX_INDICES, WEIGHTS

app = FastAPI()

# Anthropometric scale anchor (reverse-engineered from Crisalix's own
# `eyeToEyeDistance / TEMPLATE_EYE_TO_EYE_DISTANCE` scaling factor — see the
# approved re-architecture plan). This project's GNM fit already produces
# positions in real-world meters directly from the identity-fit itself (no
# artificial display-only rescale the way the retired BFM pipeline needed
# TARGET_HEAD_HEIGHT for), so there is no rescale STEP to add here — what
# was actually missing was the MEASURED per-patient anchor value itself.
# WFLW 64/68 (real GNM correspondence, gnm_correspondence.py) are the INNER
# eye corners — the ONLY eye-corner pair validated reliable at every angle
# including 90° (WFLW 60/72, the OUTER corners, were tested and dropped —
# see that module's own docstring), so this is the same real "intercanthal
# distance" clinical quantity `MorphParams.eye.intercanthalDistanceMm`
# already models (default 32mm) — computed here from THIS patient's own
# real fitted geometry instead of a slider default, and exposed in
# `/reconstruct`'s metadata for any future consumer (clinical measurements
# panel, implant/tool scale reference, a per-patient sanity check against
# population norms) without inventing one that doesn't exist yet.
_EYE_INNER_LEFT_ROW = WFLW_INDICES.index(64)
_EYE_INNER_RIGHT_ROW = WFLW_INDICES.index(68)


def compute_eye_to_eye_distance_mm(fitted_positions: np.ndarray) -> float:
    def landmark_3d(row: int) -> np.ndarray:
        return sum(w * fitted_positions[vid] for vid, w in zip(VERTEX_INDICES[row], WEIGHTS[row]))

    left = landmark_3d(_EYE_INNER_LEFT_ROW)
    right = landmark_3d(_EYE_INNER_RIGHT_ROW)
    return float(np.linalg.norm(left - right) * 1000)

# Same public/models/patients/<id>/... convention server/3ddfa_v2/reconstruct_patient.py
# already uses for that pipeline's own generated head.glb — atlas textures are
# equally per-patient generated output, not a static asset, so they follow the
# same existing precedent rather than a new one.
PUBLIC_ROOT = Path(__file__).parent.parent / "public"

ANGLE_SLOTS = ["angle1", "angle2", "angle3", "angle4"]  # angle3 = 90°, the only profile slot


@app.get("/test")
def test():
    return {"msg": "ok, tao da chay roi"}


@app.post("/detect-pose")
async def detect_pose_endpoint(file: UploadFile = File(...)):
    """Real YuNet + PIPNet (98pt) pose detection for one Bước 2 reference photo. See detect_pose.py for the models/yaw formula. Never 500s on a photo with no detectable face — returns has_face=false so the caller can fall back without a crash."""
    raw = await file.read()
    buf = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"has_face": False, "yaw": None, "landmarks_98": None, "camera_pose": None, "error": "image-decode-failed"}
    return detect_pose(image_bgr)


@app.post("/fit-multiview")
async def fit_multiview_endpoint(
    angle1: Optional[UploadFile] = File(None),
    angle2: Optional[UploadFile] = File(None),
    angle3: Optional[UploadFile] = File(None),
    angle4: Optional[UploadFile] = File(None),
):
    """D5 — real multi-view GNM identity-fit + vertex-color texture, from up
    to 4 real Bước 2 photos (0°/45°/90°/dưới lên). Validated in scratchpad
    D1-D4.5 before being ported here (see gnm_correspondence.py,
    gnm_identity_fit.py, gnm_texture_bake.py's own docstrings for the exact
    provenance of every point/formula used).

    Any missing/undetectable photo is simply DROPPED from the fit (never
    forced/fabricated) — same discipline validated throughout: 0913e4c9's
    real "below" photo once broke SolvePnP entirely with the old
    ITERATIVE-no-guess method; EPNP (used here) fixed that, but if some
    future photo still fails, it drops cleanly rather than corrupting the
    joint fit.
    """
    uploads = {"angle1": angle1, "angle2": angle2, "angle3": angle3, "angle4": angle4}
    images = {}
    for slot, upload in uploads.items():
        if upload is None:
            continue
        raw = await upload.read()
        buf = np.frombuffer(raw, dtype=np.uint8)
        image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            images[slot] = image_bgr

    if len(images) == 0:
        return JSONResponse({"error": "no-usable-photo"}, status_code=422)

    view_inputs = []
    view_slots = []
    view_images = []
    warnings = []
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

    if len(view_inputs) == 0:
        return JSONResponse({"error": "no-face-detected-in-any-photo"}, status_code=422)

    fitted_positions, per_view_pose, fit_warnings = fit_multiview(view_inputs)
    warnings.extend(fit_warnings)
    if fitted_positions is None:
        return JSONResponse({"error": "identity-fit-failed", "warnings": warnings}, status_code=422)

    trust = compute_vertex_trust()

    triangles = get_triangles()
    normals = compute_vertex_normals(fitted_positions, triangles)

    # 2026-08-24 shell/color-position correspondence fix — trust-blended
    # positions computed HERE, before color sampling, so a low-trust
    # (shell/neck) vertex's baked color is sampled from the SAME position it
    # actually renders at, instead of `fitted_positions`' raw (for those
    # vertices, PCA-extrapolated with zero real photo evidence — see
    # gnm_vertex_trust.py's own module docstring) position. `normals` above
    # is deliberately left computed from RAW `fitted_positions`, unchanged —
    # trust==1.0 exactly for every face_mask vertex (compute_vertex_trust's
    # own hard guarantee) means `blended_positions` is bit-identical to
    # `fitted_positions` there, so this cannot change any face_mask vertex's
    # own normal or projected pixel coordinate (see run_gnm_reconstruction's
    # own copy of this fix, below, for the full regression rationale).
    blended_positions = blend_positions_by_trust(fitted_positions, get_template_positions(), trust)

    bake_views = []
    views_used = []
    for slot, image_bgr, pose in zip(view_slots, view_images, per_view_pose):
        if pose is None:
            continue
        R, t, _mask = pose
        h, w = image_bgr.shape[:2]
        camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        bake_views.append({"image": image_bgr, "R": R, "t": t, "camera_matrix": camera_matrix})
        views_used.append(slot)

    vertex_color, coverage_fraction, _per_view_weight = bake_vertex_colors(blended_positions, normals, bake_views, triangles)

    # Phase 1 — same trust field applied to color: low-trust vertices blend
    # toward a real mesh-adjacency neighbor average instead of keeping the
    # raw per-vertex bake (which is what produced the blotchy shell-region
    # artifact in the baseline). face_mask vertices (trust=1) are excluded
    # by construction, so this cannot change any already-good 0° pixel.
    vertex_color = smooth_colors_by_trust(vertex_color, triangles, trust)

    # Position blend already computed above (before color sampling, so the
    # color-sample coordinate equals the final render coordinate by
    # construction) — `blended_positions` IS the final position array, no
    # second blend pass needed.
    fitted_positions = blended_positions

    return {
        "views_used": views_used,
        "warnings": warnings,
        "coverage_fraction": coverage_fraction,
        "fitted_positions": fitted_positions.flatten().tolist(),
        "vertex_colors": vertex_color.flatten().tolist(),
    }


RECONSTRUCTION_METHOD = "GNM multi-view identity fit + trust-blended shell + eye sub-pipeline (Phase 1+2)"

# Phase 1 (Face Shell) — separate, ADDITIVE staleness marker for the new
# ears+neck-transition shell topology + unified HD texture (gnm_face_shell.py),
# following the exact same "bump the string when the pipeline changes" idea
# as RECONSTRUCTION_METHOD above, but kept as its own independent field
# (metadata["faceShell"]["method"], not folded into RECONSTRUCTION_METHOD
# itself) so this Phase-1-only backend change cannot force every existing
# patient's positions/colors/atlas cache to be treated as stale — that
# staleness gate (src/lib/gnm/reconstruction-service.ts's
# EXPECTED_RECONSTRUCTION_METHOD) is explicitly out of scope for this phase.
FACE_SHELL_METHOD = "GNM Face Shell v1 — ears + submental/neck transition, unified 2K HD texture (Phase 1)"


def run_gnm_reconstruction(images: dict) -> dict:
    """The full [A]->[E] chain (per-view PIPNet/EPNP pose -> joint ridge
    identity fit -> dense-nose/width-correction -> Phase 1 trust-blend ->
    ICM/occlusion vertex-color bake + Phase 2 eye sub-pipeline -> PCA
    texture atlas), factored out of the original /fit-multiview-atlas
    handler (D22) so `/reconstruct` (Phase 3, persists to disk) and
    `/fit-multiview-atlas` (kept for any existing caller during migration)
    share ONE implementation instead of drifting apart as two near-copies —
    this project has generally preferred duplication for endpoint safety
    (see D22's own original docstring), but a THIRD near-duplicate of this
    specific ~150-line body would work directly against the plan's own
    "hợp nhất" (unify the pipelines) goal. Every individual step below is
    unchanged from D22/Phase 1/Phase 2 — this is a pure extraction, not a
    behavior change (verified by re-running ai-engine/regression's Phase 1/2
    gates after this refactor, still PASS on all 5 real patients).

    `images`: {slot: BGR ndarray}. Returns a dict:
      {"ok": bool, "error": str|None, "warnings": [str], "views_used": [str],
       "coverage_fraction": float, "fitted_positions": (17821,3) float64,
       "vertex_colors": (17821,3) float64, "atlas": dict|None}
    `atlas` (when not None) is build_multi_region_atlas's own raw return
    dict (regions/skipped/threejs) — callers persist or JSON-encode it
    however they need, unchanged from before this extraction.
    """
    if len(images) == 0:
        return {"ok": False, "error": "no-usable-photo"}

    # D-fiveview — "angle3" was the only real-90-degree-profile slot in the
    # original 4-photo convention; the newer 5-angle scan flow additionally
    # sends `left_profile`/`right_profile` (see reconstruct_cli.py's own
    # D-fiveview comment) — both need the same is_profile_view=True pose
    # handling angle3 always got, or PIPNet/EPnP gets run with frontal-view
    # assumptions on a near-90-degree face.
    PROFILE_VIEW_SLOTS = {"angle3", "left_profile", "right_profile"}
    view_inputs, view_slots, view_images, view_landmarks, warnings = [], [], [], [], []
    for slot, image_bgr in images.items():
        is_profile = slot in PROFILE_VIEW_SLOTS
        h, w = image_bgr.shape[:2]
        result = detect_pose(image_bgr, is_profile_view=is_profile)
        if not result["has_face"]:
            warnings.append(f"{slot}: no face detected (PIPNet) — dropped")
            continue
        camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        view_inputs.append(ViewInput(result["landmarks_98"], camera_matrix, is_profile, (h, w)))
        view_slots.append(slot)
        view_images.append(image_bgr)
        view_landmarks.append(result["landmarks_98"])

    if len(view_inputs) == 0:
        return {"ok": False, "error": "no-face-detected-in-any-photo"}

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

    trust = compute_vertex_trust()
    triangles = get_triangles()
    normals = compute_vertex_normals(fitted_positions, triangles)

    # 2026-08-24 shell/color-position correspondence fix — see this same
    # comment's full copy in /fit-multiview above for the complete
    # rationale. Short version: `normals` (just above) stays computed from
    # RAW `fitted_positions` — this alone is what the prior "blend
    # deliberately LAST" finding (see below, now only covering the FINAL
    # return-value assignment) actually protects, since face_mask normals
    # only depend on POSITIONS via compute_vertex_normals' triangle-normal
    # sum. `blended_positions` is bit-identical to `fitted_positions` for
    # every trust==1.0 (face_mask) vertex by construction, so every
    # downstream step below that reads a POSITION (ICM labeling, the
    # occlusion depth-buffer, the atlas per-region projection, the
    # neck/shoulder Y-band) now samples/decides at the exact position a
    # vertex renders at instead of an ungrounded low-trust PCA
    # extrapolation — without changing face_mask's own facing-weight/ICM
    # inputs (identical positions AND identical normals there).
    blended_positions = blend_positions_by_trust(fitted_positions, get_template_positions(), trust)

    view_landmarks_by_slot = dict(zip(view_slots, view_landmarks))

    bake_views, views_used, views_for_atlas = [], [], {}
    for slot, image_bgr, pose in zip(view_slots, view_images, per_view_pose):
        if pose is None:
            continue
        R, t, _mask = pose
        h, w = image_bgr.shape[:2]
        camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        # Fix 2 (2026-08-24 visual-defect audit) — real per-photo person-
        # silhouette gate, from the SAME PIPNet landmarks already detected
        # above (see compute_person_silhouette_mask's own docstring for the
        # background-bleed root cause this fixes).
        person_mask = compute_person_silhouette_mask(view_landmarks_by_slot[slot], (h, w))
        bake_views.append({"image": image_bgr, "R": R, "t": t, "camera_matrix": camera_matrix, "person_mask": person_mask, "slot": slot})
        views_used.append(slot)
        views_for_atlas[slot] = {"image": image_bgr, "R": R, "t": t, "camera_matrix": camera_matrix, "person_mask": person_mask}

    icm_result = compute_icm_labels(blended_positions, normals, bake_views, triangles)
    icm_label, icm_facing = icm_result[2], icm_result[5]
    icm_covered = icm_result[1]

    vertex_color, coverage_fraction, per_view_weight = bake_vertex_colors(
        blended_positions, normals, bake_views, triangles, icm_result=icm_result
    )

    try:
        vertex_color = improve_neck_shoulder_vertex_colors(blended_positions, triangles, vertex_color, per_view_weight)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"neck/shoulder color improvement failed, original vertex colors kept: {exc}")

    # 2026-08-24 generic-shell adaptive fallback tone — replaces the fixed
    # SENTINEL_UNCOVERED (#e0ac8f, same for every patient) with THIS
    # patient's own real mean FACE-core color, but ONLY for the
    # already-uncovered vertices of HEAD_SHELL_GENERIC/NECK (never touches
    # FACE, EYES, EARS, or any vertex that already has a real baked color —
    # `is_sentinel` below only ever matches an exact SENTINEL_UNCOVERED
    # value bake_vertex_colors/improve_neck_shoulder_vertex_colors just
    # wrote). Root cause + measurement: see compute_adaptive_shell_tone's
    # own docstring. Runs BEFORE smooth_colors_by_trust so the existing
    # trust-based neighbor smoothing (not a new blur mechanism) can still
    # blend this new tone with nearby real samples exactly as it already
    # does for every other shell color.
    try:
        region_index = compute_region_index()
        shell_mask = (region_index == REGION_HEAD_SHELL_GENERIC) | (region_index == REGION_NECK)
        adaptive_tone = compute_adaptive_shell_tone(vertex_color, icm_covered)
        if adaptive_tone is not None:
            is_sentinel = np.all(np.isclose(vertex_color, SENTINEL_UNCOVERED, atol=0.5), axis=1)
            vertex_color[shell_mask & is_sentinel] = adaptive_tone
    except Exception as exc:  # noqa: BLE001 — best-effort, never blocks the base response
        warnings.append(f"adaptive shell fallback tone failed, fixed sentinel colors kept: {exc}")

    try:
        vertex_color = smooth_colors_by_trust(vertex_color, triangles, trust)
    except Exception as exc:  # noqa: BLE001 — best-effort, never blocks the base response
        warnings.append(f"trust-based color smoothing failed, pre-smoothing vertex colors kept: {exc}")

    try:
        atlas = build_multi_region_atlas(
            blended_positions, normals, triangles, views_for_atlas,
            vertex_labels=icm_label, vertex_label_slot_names=views_used, vertex_facing=icm_facing,
        )
    except Exception as exc:  # noqa: BLE001 — atlas is additive; never let it break the base response
        warnings.append(f"atlas build failed, base vertex-color response still valid: {exc}")
        atlas = None

    # Position blend already computed above, BEFORE color sampling (see the
    # 2026-08-24 fix comment above) — `blended_positions` IS the final
    # position array; no second blend pass. (Historical note: this used to
    # run here, LAST, specifically so face_mask's color/ICM stayed
    # bit-identical to the pre-Phase-1 pipeline — that guarantee now comes
    # from `normals` above staying computed from RAW `fitted_positions`
    # instead, not from blend ordering; see this project's regression
    # suite, re-run after this change, for the same 0-diff verification on
    # face_mask.)
    fitted_positions = blended_positions

    return {
        "ok": True,
        "error": None,
        "warnings": warnings,
        "views_used": views_used,
        "coverage_fraction": coverage_fraction,
        "fitted_positions": fitted_positions,
        "vertex_colors": vertex_color,
        "atlas": atlas,
        "eye_to_eye_distance_mm": compute_eye_to_eye_distance_mm(fitted_positions),
        # Phase 1 (Face Shell) — the SAME per-view {image,R,t,camera_matrix,
        # person_mask} dicts already built above for build_multi_region_atlas,
        # exposed here too so a caller (currently only /reconstruct's own
        # face-shell block) can bake_unified_face_texture() without
        # re-running PIPNet/EPNP a second time for the same photos.
        "bake_views": bake_views,
    }


async def _read_uploaded_images(angle1, angle2, angle3, angle4) -> dict:
    uploads = {"angle1": angle1, "angle2": angle2, "angle3": angle3, "angle4": angle4}
    images = {}
    for slot, upload in uploads.items():
        if upload is None:
            continue
        raw = await upload.read()
        buf = np.frombuffer(raw, dtype=np.uint8)
        image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            images[slot] = image_bgr
    return images


@app.post("/fit-multiview-atlas")
async def fit_multiview_atlas_endpoint(
    angle1: Optional[UploadFile] = File(None),
    angle2: Optional[UploadFile] = File(None),
    angle3: Optional[UploadFile] = File(None),
    angle4: Optional[UploadFile] = File(None),
    patient_id: Optional[str] = Form(None),
):
    """D22 — kept alive during the Phase 3 migration to /reconstruct (see the
    approved re-architecture plan's migration section: old endpoints stay
    live in parallel until every real patient has been verified against the
    new one). Now a thin wrapper around `run_gnm_reconstruction` (extracted
    from this handler's own original body, see that function's docstring)
    — same real pipeline, same response shape as before.

    D23 — region textures are written as real static PNG files under
    `public/models/patients/<patient_id>/atlas/` and returned as
    `texture_url` instead of a base64 string when `patient_id` is given
    (falls back to base64 otherwise, unchanged D22 behavior).
    """
    images = await _read_uploaded_images(angle1, angle2, angle3, angle4)
    result = run_gnm_reconstruction(images)
    if not result["ok"]:
        status = 422
        return JSONResponse({"error": result["error"], "warnings": result.get("warnings", [])}, status_code=status)

    warnings = result["warnings"]
    fitted_positions = result["fitted_positions"]
    vertex_color = result["vertex_colors"]
    atlas = result["atlas"]

    response = {
        "views_used": result["views_used"],
        "warnings": warnings,
        "coverage_fraction": result["coverage_fraction"],
        "fitted_positions": fitted_positions.flatten().tolist(),
        "vertex_colors": vertex_color.flatten().tolist(),
        "atlas": None,
        "eye_to_eye_distance_mm": result["eye_to_eye_distance_mm"],
    }

    if atlas is not None:
        # cache-busting query param — a re-run for the same patient (new photos,
        # re-confirmed AI mapping, etc.) must not silently serve the browser's
        # cached copy of the OLD texture at the same URL.
        version = str(int(time.time() * 1000))
        write_dir = None
        if patient_id:
            write_dir = PUBLIC_ROOT / "models" / "patients" / patient_id / "atlas"
            os.makedirs(write_dir, exist_ok=True)

        regions_out = {}
        for region, data in atlas["regions"].items():
            entry = {
                "tex_w": data["tex_w"],
                "tex_h": data["tex_h"],
                "n_vertices": data["n_vertices"],
                "n_texels_covered": data["n_texels_covered"],
            }
            if write_dir is not None:
                file_path = write_dir / f"{region}.png"
                with open(file_path, "wb") as f:
                    f.write(data["texture_png_bytes"])
                entry["texture_url"] = f"/models/patients/{patient_id}/atlas/{region}.png?v={version}"
            else:
                entry["texture_png_base64"] = base64.b64encode(data["texture_png_bytes"]).decode("ascii")
            regions_out[region] = entry

        response["atlas"] = {
            "regions": regions_out,
            "skipped": atlas["skipped"],
            "threejs": atlas["threejs"],
        }

    return response


@app.post("/reconstruct")
async def reconstruct_endpoint(
    angle1: Optional[UploadFile] = File(None),
    angle2: Optional[UploadFile] = File(None),
    angle3: Optional[UploadFile] = File(None),
    angle4: Optional[UploadFile] = File(None),
    patient_id: str = Form(...),
):
    """Phase 3 — the canonical, persisted per-patient reconstruction. Runs
    the exact same `run_gnm_reconstruction` chain `/fit-multiview-atlas`
    uses (same identity fit, same Phase 1 trust-blend, same Phase 2 eye
    sub-pipeline, same PCA atlas — nothing new is computed here), but WRITES
    the result to disk once under `public/models/patients/<patient_id>/
    reconstruction/` instead of returning the full positions/colors payload
    inline — this is the "one representation for the whole patient
    lifecycle" artifact the re-architecture plan calls for, replacing the
    prior pattern of re-fitting from the 4 uploaded photos on every single
    page load (see src/lib/reconstruction.ts's own `ensureReconstructionOnDisk`
    for the client-side half of this, ported to point here in Phase 3).

    `patient_id` is required (unlike /fit-multiview-atlas, where it's
    optional) — a reconstruction with nowhere to persist to isn't this
    endpoint's job; callers with no patient context should use
    /fit-multiview-atlas instead.

    Writes:
      reconstruction/positions.f32 — raw (17821*3) float32, little-endian
      reconstruction/colors.f32    — raw (17821*3) float32, little-endian, 0-255 range
      reconstruction/atlas/<region>.png — same per-region PNGs /fit-multiview-atlas already writes
      reconstruction/face_HD.png — Phase 1 (Face Shell), additive: unified 2K texture, see FACE_SHELL_METHOD
      reconstruction/shell_topology.json — Phase 1 (Face Shell), additive: {shellVertexIndices, triangles, uvs}
      reconstruction/metadata.json — {ok, method, viewsUsed, warnings, coverageFraction, vertexCount, faceShell, generatedAt, processingTimeMs}

    `method` matches RECONSTRUCTION_METHOD — the same staleness-check
    convention src/lib/reconstruction.ts's `EXPECTED_RECONSTRUCTION_METHOD`
    already validated for the (now-retired-from-patient-facing) 3DDFA_V2
    pipeline: a caller compares this string before trusting an on-disk
    reconstruction, so any future change to this function's own pipeline
    only needs RECONSTRUCTION_METHOD bumped for every patient to correctly
    regenerate instead of silently serving stale geometry forever.
    """
    t0 = time.time()
    images = await _read_uploaded_images(angle1, angle2, angle3, angle4)
    out_dir = PUBLIC_ROOT / "models" / "patients" / patient_id / "reconstruction"
    os.makedirs(out_dir, exist_ok=True)

    # Save uploaded images temporarily for reconstruct_patient_scan
    tmp_paths = {}
    for slot, img_bgr in images.items():
        slot_file = out_dir / f"{slot}.jpg"
        cv2.imwrite(str(slot_file), img_bgr)
        tmp_paths[slot] = str(slot_file)

    from reconstruct_cli import reconstruct_patient_scan
    result = reconstruct_patient_scan(
        image_paths=tmp_paths,
        patient_id=patient_id,
        session_id=str(int(t0)),
        output_dir=out_dir,
    )

    if not result.get("ok"):
        return JSONResponse({"ok": False, "error": result.get("error", "Reconstruction failed")}, status_code=422)

    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
