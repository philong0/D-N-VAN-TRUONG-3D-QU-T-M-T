"""
Renders the REAL, FULL production pipeline offline for a given patient --
base vertex-color layer (materialIndex 0) AND the 19 atlas region textures
(materialIndex 1..N) composited together exactly the way Canvas3D.tsx's
BufferGeometry groups + per-triangle materialIndex do it in the browser.
This is the check the earlier phase1 prototype was missing (it only
rendered materialIndex 0, which is why it looked "fixed" while the real
Studio view, dominated by the atlas layer, did not change).

Usage: venv/bin/python3 scratchpad/full_pipeline_render.py <patient_id> <out_prefix>
photos must exist at .data/patients/<patient_id>/photos/angle{1..4}.{png,jpg}
"""
import sys
sys.path.insert(0, "ai-engine")
import os
import glob
import numpy as np
import cv2

from detect_pose import detect_pose
from gnm_identity_fit import fit_multiview, get_triangles, ViewInput
from gnm_dense_nose import enrich_with_dense_nose
from gnm_width_correction import apply_width_correction
from gnm_texture_bake import compute_vertex_normals, bake_vertex_colors, compute_icm_labels, project
from gnm_texture_atlas import build_multi_region_atlas, improve_neck_shoulder_vertex_colors

patient_id = sys.argv[1]
out_prefix = sys.argv[2]

PATIENT_DIR = f".data/patients/{patient_id}/photos"

images = {}
for slot in ["angle1", "angle2", "angle3", "angle4"]:
    matches = glob.glob(f"{PATIENT_DIR}/{slot}.*")
    if matches:
        img = cv2.imread(matches[0])
        if img is not None:
            images[slot] = img

view_inputs, view_slots, view_images = [], [], []
for slot, image_bgr in images.items():
    is_profile = slot == "angle3"
    h, w = image_bgr.shape[:2]
    result = detect_pose(image_bgr, is_profile_view=is_profile)
    if not result["has_face"]:
        print(f"{slot}: no face, dropped")
        continue
    camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    view_inputs.append(ViewInput(result["landmarks_98"], camera_matrix, is_profile, (h, w)))
    view_slots.append(slot)
    view_images.append(image_bgr)

fitted_positions, per_view_pose, fit_warnings = fit_multiview(view_inputs)
print("fit warnings:", fit_warnings)
fitted_positions, _ = enrich_with_dense_nose(fitted_positions, view_inputs, dict(zip(view_slots, view_images)))
fitted_positions, _ = apply_width_correction(fitted_positions, view_inputs, view_slots)

triangles = get_triangles()
normals = compute_vertex_normals(fitted_positions, triangles)
V = fitted_positions.shape[0]

bake_views = []
views_used = []
views_for_atlas = {}
render_view = None
for slot, image_bgr, pose in zip(view_slots, view_images, per_view_pose):
    if pose is None:
        continue
    R, t, _m = pose
    h, w = image_bgr.shape[:2]
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    entry = {"image": image_bgr, "R": R, "t": t, "camera_matrix": cam}
    bake_views.append(entry)
    views_used.append(slot)
    views_for_atlas[slot] = entry
    if slot == "angle1":
        render_view = entry

icm_result = compute_icm_labels(fitted_positions, normals, bake_views, triangles)
icm_label, icm_facing = icm_result[2], icm_result[5]

vertex_color, coverage_fraction, per_view_weight = bake_vertex_colors(
    fitted_positions, normals, bake_views, triangles, icm_result=icm_result
)
print("coverage_fraction:", coverage_fraction)
try:
    vertex_color = improve_neck_shoulder_vertex_colors(fitted_positions, triangles, vertex_color, per_view_weight)
except Exception as exc:
    print("neck/shoulder improve failed:", exc)

atlas = build_multi_region_atlas(
    fitted_positions, normals, triangles, views_for_atlas,
    vertex_labels=icm_label, vertex_label_slot_names=views_used, vertex_facing=icm_facing,
)
print("atlas regions:", list(atlas["regions"].keys()))
print("skipped:", atlas["skipped"])

tj = atlas["threejs"]
extra_source_vids = tj["extra_source_vids"]
V_total = V + len(extra_source_vids)

positions_full = np.zeros((V_total, 3))
positions_full[:V] = fitted_positions
for i, orig_vid in enumerate(extra_source_vids):
    positions_full[V + i] = fitted_positions[orig_vid]

vertex_color_full = np.zeros((V_total, 3))
vertex_color_full[:V] = vertex_color

uv_full = np.array(tj["uv"]).reshape(-1, 2)

index = np.array(tj["index"], dtype=np.int64)
triangles_full_idx = index.reshape(-1, 3)  # (n_tri, 3), contiguous per group by construction

region_tex = {name: data["texture_rgb"] for name, data in atlas["regions"].items()}

# per-triangle: materialIndex + region name
n_tri = triangles_full_idx.shape[0]
tri_material = np.zeros(n_tri, dtype=np.int64)
tri_region = [None] * n_tri
for g in tj["groups"]:
    t0, t1 = g["start"] // 3, (g["start"] + g["count"]) // 3
    tri_material[t0:t1] = g["materialIndex"]
    if g["materialIndex"] != 0:
        for i in range(t0, t1):
            tri_region[i] = g["region"]

assert render_view is not None
rh, rw = render_view["image"].shape[:2]
RENDER_MAX_DIM = 900
scale = min(1.0, RENDER_MAX_DIM / max(rh, rw))
out_w, out_h = max(int(rw * scale), 1), max(int(rh * scale), 1)
cam_scaled = render_view["camera_matrix"].copy()
cam_scaled[0, 0] *= scale
cam_scaled[1, 1] *= scale
cam_scaled[0, 2] *= scale
cam_scaled[1, 2] *= scale


def rasterize_composite(positions, triangles_idx, tri_material, tri_region, vertex_color, uv, region_tex, R, t, camera_matrix, w, h):
    u, v, z = project(positions, R, t, camera_matrix)
    img = np.zeros((h, w, 3), dtype=np.float64)
    depth = np.full((h, w), np.inf)
    tri_u, tri_v, tri_z = u[triangles_idx], v[triangles_idx], z[triangles_idx]
    valid_tri = (tri_z > 0).all(axis=1)
    for idx in np.nonzero(valid_tri)[0]:
        pu, pv, pz = tri_u[idx], tri_v[idx], tri_z[idx]
        x_min = max(int(np.floor(pu.min())), 0)
        x_max = min(int(np.ceil(pu.max())), w - 1)
        y_min = max(int(np.floor(pv.min())), 0)
        y_max = min(int(np.ceil(pv.max())), h - 1)
        if x_min > x_max or y_min > y_max:
            continue
        xs, ys = np.meshgrid(np.arange(x_min, x_max + 1) + 0.5, np.arange(y_min, y_max + 1) + 0.5)
        x0, y0, x1, y1, x2, y2 = pu[0], pv[0], pu[1], pv[1], pu[2], pv[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        interp_z = w0 * pz[0] + w1 * pz[1] + w2 * pz[2]
        region_d = depth[y_min:y_max + 1, x_min:x_max + 1]
        update = inside & (interp_z < region_d)
        if not update.any():
            continue
        region_d[update] = interp_z[update]

        tri3 = triangles_idx[idx]
        if tri_material[idx] == 0:
            pc = vertex_color[tri3]
            interp_c = w0[..., None] * pc[0] + w1[..., None] * pc[1] + w2[..., None] * pc[2]
        else:
            tex = region_tex[tri_region[idx]]
            th, tw = tex.shape[:2]
            puv = uv[tri3]
            interp_uv0 = w0 * puv[0, 0] + w1 * puv[1, 0] + w2 * puv[2, 0]
            interp_uv1 = w0 * puv[0, 1] + w1 * puv[1, 1] + w2 * puv[2, 1]
            tx = np.clip((interp_uv0 * (tw - 1)).astype(int), 0, tw - 1)
            ty = np.clip((interp_uv1 * (th - 1)).astype(int), 0, th - 1)
            interp_c = tex[ty, tx].astype(np.float64)

        region_c = img[y_min:y_max + 1, x_min:x_max + 1]
        for ch in range(3):
            ch_region = region_c[:, :, ch]
            ch_region[update] = interp_c[:, :, ch][update]
    return np.clip(img, 0, 255).astype(np.uint8)


out_img = rasterize_composite(
    positions_full, triangles_full_idx, tri_material, tri_region,
    vertex_color_full, uv_full, region_tex,
    render_view["R"], render_view["t"], cam_scaled, out_w, out_h,
)

os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
cv2.imwrite(f"{out_prefix}_full.png", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
print("wrote", f"{out_prefix}_full.png")
