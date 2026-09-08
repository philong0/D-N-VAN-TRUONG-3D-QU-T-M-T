"""
PHASE 1 offline prototype -- NO production files touched.

BEFORE = the REAL production per-vertex source label as
gnm_texture_bake.bake_vertex_colors actually assigns it today: per
REGION_SPEC anatomical group (+ one remainder pseudo-group), each group
picks ONE priority-ranked winning view via
gnm_texture_atlas._sample_colors_at_positions, independently per group.
That grouping is the thing hypothesized to cause hard source-boundaries
at anatomical-group edges regardless of real mesh continuity.

AFTER = same real per-vertex/per-view data (same visibility z-buffer test,
same facing-weight, same raw sampled pixel per view -- nothing recomputed,
nothing invented) but relabeled with per-vertex ICM (iterated conditional
modes) Potts-smoothness optimization directly on real mesh topology
(triangle adjacency), ignoring the anatomical grouping entirely:
  cost(v, k) = data_cost(v,k) + switch_penalty * (# neighbors whose label != k)
  data_cost(v,k) = 1 - facing_weight(v,k)   [only defined for views that
                                              really, visibly see v]
Only ever selects among views that ACTUALLY cover a vertex (same
visibility/occlusion test as production) -- never invents coverage. Output
color for a chosen label is that view's own raw sampled pixel, verbatim --
no blending, no averaging, no gain/luminance correction anywhere in this
script.

Run: venv/bin/python3 scratchpad/phase1_source_label_prototype.py
"""
import sys
sys.path.insert(0, "ai-engine")
import os
import time
import numpy as np
import cv2

from detect_pose import detect_pose
from gnm_identity_fit import fit_multiview, get_triangles, ViewInput
from gnm_dense_nose import enrich_with_dense_nose
from gnm_width_correction import apply_width_correction
from gnm_texture_bake import (
    compute_vertex_normals, view_facing_weight, project, SENTINEL_UNCOVERED,
)
import gnm_texture_atlas as gta

PATIENT_DIR = ".data/patients/0e9e1d90-2478-4d49-872f-c80772c4bc4f/photos"
OUT_DIR = "scratchpad/phase1_source_label"
os.makedirs(OUT_DIR, exist_ok=True)

ANGLE_FILES = [("angle1", "angle1.png"), ("angle2", "angle2.jpg"), ("angle3", "angle3.png"), ("angle4", "angle4.png")]
LABEL_NAMES = {0: "angle1", 1: "angle2", 2: "angle3", 3: "angle4", -1: "fallback"}
LABEL_COLORS = {  # RGB, purely for label-map visualization PNGs
    0: np.array([230, 60, 60], dtype=np.float64),
    1: np.array([60, 200, 90], dtype=np.float64),
    2: np.array([60, 110, 230], dtype=np.float64),
    3: np.array([230, 200, 40], dtype=np.float64),
    -1: np.array([140, 140, 140], dtype=np.float64),
}

t0 = time.time()

# ---------- 1. real fit, identical pipeline to /fit-multiview-atlas ----------
images = {}
for slot, fname in ANGLE_FILES:
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

print("views for pose:", view_slots, "warnings:", warnings)
fitted_positions, per_view_pose, fit_warnings = fit_multiview(view_inputs)
print("fit warnings:", fit_warnings)
fitted_positions, dn_note = enrich_with_dense_nose(fitted_positions, view_inputs, dict(zip(view_slots, view_images)))
fitted_positions, wc_note = apply_width_correction(fitted_positions, view_inputs, view_slots)
print("dense_nose_note:", dn_note, "width_note:", wc_note)

triangles = get_triangles()
normals = compute_vertex_normals(fitted_positions, triangles)
V = fitted_positions.shape[0]
print(f"V={V} triangles={len(triangles)}  (fit took {time.time()-t0:.1f}s)")

views_by_slot = {}
slot_names = []
render_view = None  # angle1's own view, reused below as the offline rasterizer's camera
for slot, image_bgr, pose in zip(view_slots, view_images, per_view_pose):
    if pose is None:
        continue
    R, t, _m = pose
    h, w = image_bgr.shape[:2]
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    idx = len(views_by_slot)
    entry = {"image": image_bgr, "R": R, "t": t, "camera_matrix": cam, "name": slot}
    views_by_slot[idx] = entry
    slot_names.append(slot)
    if slot == "angle1":
        render_view = entry

n_views = len(views_by_slot)
print("baking views (slot order):", slot_names)

# ---------- 2. shared real per-vertex/per-view data ----------
t1 = time.time()
depth_buffers = gta._build_depth_buffers(fitted_positions, triangles, views_by_slot)

valid_masks = np.zeros((n_views, V), dtype=bool)
facing = np.zeros((n_views, V), dtype=np.float64)
colors = np.zeros((n_views, V, 3), dtype=np.float64)

for k, v in views_by_slot.items():
    w_facing = view_facing_weight(fitted_positions, normals, v["R"], v["t"])
    u, vv, z = project(fitted_positions, v["R"], v["t"], v["camera_matrix"])
    ih, iw = v["image"].shape[:2]
    valid = (z > 0) & (u >= 0) & (u < iw) & (vv >= 0) & (vv < ih)
    valid &= gta._visible(fitted_positions, v["R"], v["t"], v["camera_matrix"], depth_buffers[k])
    uu = np.clip(u.astype(int), 0, iw - 1)
    vvv = np.clip(vv.astype(int), 0, ih - 1)
    c = v["image"][vvv, uu][:, ::-1].astype(np.float64)  # BGR -> RGB, real raw pixel, untouched
    valid_masks[k] = valid
    facing[k] = w_facing
    colors[k] = c

has_any = valid_masks.any(axis=0)
print(f"vertices with >=1 real valid view: {int(has_any.sum())}/{V}  (shared data took {time.time()-t1:.1f}s)")

adj = gta._build_adjacency(triangles, V)

# ---------- 3. BEFORE = real production per-REGION_SPEC-group priority selection ----------
names, groups = gta._load_vertex_groups()
before_label = np.full(V, -1, dtype=np.int32)
before_color = np.tile(SENTINEL_UNCOVERED, (V, 1))
assigned = np.zeros(V, dtype=bool)


def region_priority_select(idx):
    if len(idx) == 0:
        return
    pos = fitted_positions[idx]
    nrm = normals[idx]
    c, cov, best_slot = gta._sample_colors_at_positions(pos, nrm, views_by_slot, depth_buffers)
    before_color[idx[cov]] = c[cov]
    before_label[idx[cov]] = best_slot[cov]


for group_names, _res in gta.REGION_SPEC.values():
    m = gta._region_mask(names, groups, group_names, V) & ~assigned
    idx = np.where(m)[0]
    region_priority_select(idx)
    assigned[idx] = True
remaining = np.where(~assigned)[0]
region_priority_select(remaining)
assigned[remaining] = True

# sanity: BEFORE's own coverage must equal has_any exactly (same visibility test, nothing new)
assert np.array_equal(before_label != -1, has_any), "BEFORE coverage mismatch vs shared visibility data"

# ---------- 4. AFTER = per-vertex ICM Potts-smoothness relabel, same real data ----------
SWITCH_PENALTY = 0.6
N_PASSES = 20

valid_k_per_vertex = [np.where(valid_masks[:, v])[0] for v in range(V)]

after_label = np.full(V, -1, dtype=np.int32)
for v in range(V):
    vk = valid_k_per_vertex[v]
    if len(vk):
        after_label[v] = vk[np.argmax(facing[vk, v])]

t2 = time.time()
order = np.arange(V)
for it in range(N_PASSES):
    changed = 0
    for v in order:
        vk = valid_k_per_vertex[v]
        if len(vk) == 0:
            continue
        nbrs = adj[v]
        if not nbrs:
            best_k = vk[np.argmax(facing[vk, v])]
        else:
            nbr_labels = [after_label[n] for n in nbrs]
            best_cost = None
            best_k = after_label[v]
            for k in vk:
                data_cost = 1.0 - facing[k, v]
                smooth_cost = SWITCH_PENALTY * sum(1 for nl in nbr_labels if nl != k)
                cost = data_cost + smooth_cost
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_k = k
        if best_k != after_label[v]:
            after_label[v] = best_k
            changed += 1
    if changed == 0:
        print(f"ICM converged after {it+1} passes")
        break
    if it == N_PASSES - 1:
        print(f"ICM stopped at pass cap {N_PASSES} ({changed} still changing)")
print(f"ICM relabel took {time.time()-t2:.1f}s")

after_color = np.tile(SENTINEL_UNCOVERED, (V, 1))
for v in range(V):
    k = after_label[v]
    if k >= 0:
        after_color[v] = colors[k, v]

assert np.array_equal(after_label != -1, has_any), "AFTER coverage mismatch vs shared visibility data -- fabricated or dropped real coverage"

# ---------- 5. metrics ----------

def source_boundary_count(label):
    cnt = 0
    for a in range(V):
        for b in adj[a]:
            if a < b and label[a] != label[b]:
                cnt += 1
    return cnt


def connected_components(label):
    visited = np.zeros(V, dtype=bool)
    n_comp = 0
    for start in range(V):
        if visited[start]:
            continue
        lab = label[start]
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if not visited[w] and label[w] == lab:
                    visited[w] = True
                    stack.append(w)
        n_comp += 1
    return n_comp


before_boundary = source_boundary_count(before_label)
after_boundary = source_boundary_count(after_label)
before_cc = connected_components(before_label)
after_cc = connected_components(after_label)
before_fallback = int((before_label == -1).sum())
after_fallback = int((after_label == -1).sum())

# real pixel preservation: every covered vertex's stored color must equal
# (bit-exact) the raw sampled color of its OWN assigned view -- proves no
# blend/average/gain ever touched it in either BEFORE or AFTER.
covered_idx = np.where(has_any)[0]
before_ok = np.allclose(before_color[covered_idx], colors[before_label[covered_idx], covered_idx], atol=0)
after_ok = np.allclose(after_color[covered_idx], colors[after_label[covered_idx], covered_idx], atol=0)
pixel_preserved = before_ok and after_ok

print("\n================ RESULTS ================")
print(f"source boundary edges:  BEFORE = {before_boundary}   AFTER = {after_boundary}   (delta {after_boundary - before_boundary:+d}, {100*(after_boundary-before_boundary)/max(before_boundary,1):+.1f}%)")
print(f"connected components:   BEFORE = {before_cc}   AFTER = {after_cc}   (delta {after_cc - before_cc:+d})")
print(f"fallback vertices:      BEFORE = {before_fallback}   AFTER = {after_fallback}   (must match: {before_fallback == after_fallback})")
print(f"real pixel preservation: {'100%' if pixel_preserved else 'FAILED -- NOT 100%'}")
print("label distribution BEFORE:", {LABEL_NAMES[k]: int((before_label == k).sum()) for k in sorted(set(before_label.tolist()))})
print("label distribution AFTER: ", {LABEL_NAMES[k]: int((after_label == k).sum()) for k in sorted(set(after_label.tolist()))})

np.save(f"{OUT_DIR}/before_label.npy", before_label)
np.save(f"{OUT_DIR}/after_label.npy", after_label)
np.save(f"{OUT_DIR}/before_color.npy", before_color)
np.save(f"{OUT_DIR}/after_color.npy", after_color)
np.save(f"{OUT_DIR}/fitted_positions.npy", fitted_positions)

# ---------- 6. offline software rasterizer (Gouraud, real z-buffer) ----------

def rasterize_gouraud(positions, triangles, vertex_color, R, t, camera_matrix, w, h):
    u, v, z = project(positions, R, t, camera_matrix)
    img = np.zeros((h, w, 3), dtype=np.float64)
    depth = np.full((h, w), np.inf)
    tri_u, tri_v, tri_z = u[triangles], v[triangles], z[triangles]
    tri_c = vertex_color[triangles]
    valid_tri = (tri_z > 0).all(axis=1)
    for idx in np.nonzero(valid_tri)[0]:
        pu, pv, pz = tri_u[idx], tri_v[idx], tri_z[idx]
        pc = tri_c[idx]
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
        interp_c = w0[..., None] * pc[0] + w1[..., None] * pc[1] + w2[..., None] * pc[2]
        region_c = img[y_min:y_max + 1, x_min:x_max + 1]
        for ch in range(3):
            ch_region = region_c[:, :, ch]
            ch_region[update] = interp_c[:, :, ch][update]
    return np.clip(img, 0, 255).astype(np.uint8)


assert render_view is not None, "angle1 view missing -- cannot pick a render camera"
rh, rw = render_view["image"].shape[:2]
# cap render resolution for offline speed; camera_matrix already encodes rw x rh, scale it consistently
RENDER_MAX_DIM = 900
scale = min(1.0, RENDER_MAX_DIM / max(rh, rw))
out_w, out_h = max(int(rw * scale), 1), max(int(rh * scale), 1)
cam_scaled = render_view["camera_matrix"].copy()
cam_scaled[0, 0] *= scale
cam_scaled[1, 1] *= scale
cam_scaled[0, 2] *= scale
cam_scaled[1, 2] *= scale

t3 = time.time()
before_img = rasterize_gouraud(fitted_positions, triangles, before_color, render_view["R"], render_view["t"], cam_scaled, out_w, out_h)
after_img = rasterize_gouraud(fitted_positions, triangles, after_color, render_view["R"], render_view["t"], cam_scaled, out_w, out_h)
print(f"rasterize took {time.time()-t3:.1f}s")

cv2.imwrite(f"{OUT_DIR}/before.png", cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{OUT_DIR}/after.png", cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR))

before_label_color = np.array([LABEL_COLORS[int(k)] for k in before_label])
after_label_color = np.array([LABEL_COLORS[int(k)] for k in after_label])
before_label_img = rasterize_gouraud(fitted_positions, triangles, before_label_color, render_view["R"], render_view["t"], cam_scaled, out_w, out_h)
after_label_img = rasterize_gouraud(fitted_positions, triangles, after_label_color, render_view["R"], render_view["t"], cam_scaled, out_w, out_h)
cv2.imwrite(f"{OUT_DIR}/before_labels.png", cv2.cvtColor(before_label_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{OUT_DIR}/after_labels.png", cv2.cvtColor(after_label_img, cv2.COLOR_RGB2BGR))

print(f"\nWrote: {OUT_DIR}/before.png {OUT_DIR}/after.png {OUT_DIR}/before_labels.png {OUT_DIR}/after_labels.png")
print(f"TOTAL time: {time.time()-t0:.1f}s")
