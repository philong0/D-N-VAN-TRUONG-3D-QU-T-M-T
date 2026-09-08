"""Standalone re-run of the /fit-multiview-atlas endpoint's exact logic,
importing gnm_texture_bake/gnm_texture_atlas/gnm_identity_fit FRESH in this
process -- picks up local file edits without touching the already-running
uvicorn server (which does not use --reload). Writes atlas PNGs to a scratch
dir, not the live public/models/patients/<id>/atlas/ path, so this never
touches what the running app actually serves.
"""
import sys
sys.path.insert(0, "ai-engine")
import os
import cv2
import numpy as np

from detect_pose import detect_pose
from gnm_identity_fit import fit_multiview, get_triangles, ViewInput
from gnm_dense_nose import enrich_with_dense_nose
from gnm_width_correction import apply_width_correction
from gnm_texture_bake import compute_vertex_normals, bake_vertex_colors
from gnm_texture_atlas import build_multi_region_atlas, improve_neck_shoulder_vertex_colors

PATIENT_DIR = ".data/patients/0e9e1d90-2478-4d49-872f-c80772c4bc4f/photos"
OUT_DIR = "scratchpad/atlas_test_out"
os.makedirs(OUT_DIR, exist_ok=True)

images = {}
for slot, fname in [("angle1", "angle1.png"), ("angle2", "angle2.jpg"), ("angle3", "angle3.png"), ("angle4", "angle4.png")]:
    img = cv2.imread(f"{PATIENT_DIR}/{fname}")
    if img is not None:
        images[slot] = img

view_inputs, view_slots, view_images, warnings = [], [], [], []
for slot, image_bgr in images.items():
    is_profile = slot == "angle3"
    h, w = image_bgr.shape[:2]
    result = detect_pose(image_bgr, is_profile_view=is_profile)
    if not result["has_face"]:
        warnings.append(f"{slot}: no face")
        continue
    camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    view_inputs.append(ViewInput(result["landmarks_98"], camera_matrix, is_profile, (h, w)))
    view_slots.append(slot)
    view_images.append(image_bgr)

print("views used for pose:", view_slots, "warnings:", warnings)

fitted_positions, per_view_pose, fit_warnings = fit_multiview(view_inputs)
print("fit warnings:", fit_warnings)

fitted_positions, dense_nose_note = enrich_with_dense_nose(fitted_positions, view_inputs, dict(zip(view_slots, view_images)))
fitted_positions, width_note = apply_width_correction(fitted_positions, view_inputs, view_slots)
print("dense_nose_note:", dense_nose_note, "width_note:", width_note)

triangles = get_triangles()
normals = compute_vertex_normals(fitted_positions, triangles)

bake_views = []
views_used = []
views_for_atlas = {}
for slot, image_bgr, pose in zip(view_slots, view_images, per_view_pose):
    if pose is None:
        continue
    R, t, _mask = pose
    h, w = image_bgr.shape[:2]
    camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    bake_views.append({"image": image_bgr, "R": R, "t": t, "camera_matrix": camera_matrix})
    views_used.append(slot)
    views_for_atlas[slot] = {"image": image_bgr, "R": R, "t": t, "camera_matrix": camera_matrix}

vertex_color, coverage_fraction, per_view_weight = bake_vertex_colors(fitted_positions, normals, bake_views, triangles)
print("coverage_fraction:", coverage_fraction)

try:
    vertex_color = improve_neck_shoulder_vertex_colors(fitted_positions, triangles, vertex_color, per_view_weight)
except Exception as exc:
    print("neck/shoulder improve failed:", exc)

atlas = build_multi_region_atlas(fitted_positions, normals, triangles, views_for_atlas)

print("\nregions:", list(atlas["regions"].keys()))
print("skipped:", atlas["skipped"])
print("\nluminance corrections applied:")
for region, data in atlas["regions"].items():
    lc = data.get("luminance_corrected")
    if lc:
        print(f"  {region}: {lc}")
    else:
        print(f"  {region}: (no correction)")

for region, data in atlas["regions"].items():
    path = f"{OUT_DIR}/{region}.png"
    with open(path, "wb") as f:
        f.write(data["texture_png_bytes"])

np.save(f"{OUT_DIR}/vertex_colors.npy", vertex_color)
np.save(f"{OUT_DIR}/fitted_positions.npy", fitted_positions)
print("\nSaved to", OUT_DIR)
