"""Independent software rasterizer -- completely separate code path from
Canvas3D's WebGL pipeline (no Three.js, no GPU, no browser). Consumes the
EXACT runtime geometry (position/index/groups/colorSrgb/uv) exported by
inspect_runtime_geometry.mts (which itself runs the real src/lib/gnm/* code
against the live server), and renders 3 independent views:

  1. gouraud_vertexcolor.png -- pure per-vertex color, barycentric-interpolated
     (no atlas, no shader, no lighting) -- the debug-render the user asked for.
  2. materialindex_debug.png -- flat solid color per materialIndex (0=gray,
     1..19=distinct colors), zero interpolation -- to check whether visible
     artifact shapes line up with materialIndex/group boundaries.
  3. production_faithful.png -- material-0 triangles use vertex-color Gouraud,
     material>=1 triangles sample the REAL atlas PNG via UV -- closest
     reproduction of what Canvas3D would actually show, via an independent
     rasterizer (if this shows the same artifacts as the browser screenshot,
     it proves the cause is in the DATA, not WebGL/driver-specific).

If the SAME polygon-shaped artifacts appear in these images (rendered by
totally different code, no Three.js involved), that proves the cause is in
the underlying geometry/color/UV data itself, not a WebGL/shader/material
config bug specific to Canvas3D.
"""
import json
import numpy as np
import cv2

GEO_PATH = "scratchpad/runtime_geometry_export.json"
ATLAS_DIR = "public/models/patients/0e9e1d90-2478-4d49-872f-c80772c4bc4f/atlas"
OUT_DIR = "scratchpad"
W, H = 900, 1000
FOV_DEG = 35.0

d = json.load(open(GEO_PATH))
positions = np.array(d["position"], dtype=np.float64).reshape(-1, 3)
index = np.array(d["index"], dtype=np.int64)
triangles = index.reshape(-1, 3)
color = np.array(d["colorSrgb"], dtype=np.float64).reshape(-1, 3)  # RGB 0-255
groups = d["groups"]
uv = np.array(d["uv"], dtype=np.float64).reshape(-1, 2) if d["uv"] else None

n_verts_real = positions.shape[0]
print(f"vertices: {n_verts_real}  triangles: {len(triangles)}  groups: {len(groups)}")

# sanity: index bounds
print(f"max(index)={index.max()} min(index)={index.min()} position.count={n_verts_real} -> {'OK' if index.max() < n_verts_real else 'BUG: index out of bounds'}")
print(f"color.count={color.shape[0]} == position.count={n_verts_real}: {color.shape[0] == n_verts_real}")

# per-triangle materialIndex lookup from groups (index ranges are in TRIANGLE-INDEX space, i.e. group.start/count are element offsets into `index`, /3 for triangle id)
tri_material = np.full(len(triangles), -1, dtype=np.int64)
for g in groups:
    t0, t1 = g["start"] // 3, (g["start"] + g["count"]) // 3
    tri_material[t0:t1] = g["materialIndex"]
assert (tri_material >= 0).all(), "some triangle has no materialIndex assigned"

# ---- camera setup: frontal view centered on the real (masked) vertex set ----
centroid = positions.mean(axis=0)
radius = np.linalg.norm(positions - centroid, axis=1).max()
fov_rad = np.radians(FOV_DEG)
dist = (radius / np.sin(fov_rad / 2)) * 1.25
eye = centroid + np.array([0.0, 0.02, dist])
forward = centroid - eye
forward /= np.linalg.norm(forward)
world_up = np.array([0.0, 1.0, 0.0])
right = np.cross(forward, world_up)
right /= np.linalg.norm(right)
true_up = np.cross(right, forward)

rel = positions - eye[None, :]
cx = rel @ right
cy = rel @ true_up
cz = rel @ forward
focal = (H / 2) / np.tan(fov_rad / 2)
px = focal * cx / np.clip(cz, 1e-6, None) + W / 2
py = -focal * cy / np.clip(cz, 1e-6, None) + H / 2

MATERIAL_DEBUG_COLORS = {}
rng = np.random.RandomState(42)
palette = [
    (128, 128, 128),  # 0 base vertex-color
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (200, 200, 0), (0, 200, 200), (200, 0, 200), (100, 200, 100), (200, 100, 100), (100, 100, 200), (150, 150, 150),
]
for g in groups:
    MATERIAL_DEBUG_COLORS[g["materialIndex"]] = palette[g["materialIndex"] % len(palette)]

# load real atlas PNGs (RGB) for the production-faithful render
region_by_mat = {g["materialIndex"]: g["region"] for g in groups if g["region"]}
atlas_imgs = {}
for mi, region in region_by_mat.items():
    img = cv2.imread(f"{ATLAS_DIR}/{region}.png")
    if img is not None:
        atlas_imgs[mi] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)

img_gouraud = np.zeros((H, W, 3), dtype=np.float64)
img_matdebug = np.zeros((H, W, 3), dtype=np.float64)
img_prod = np.zeros((H, W, 3), dtype=np.float64)
depth = np.full((H, W), np.inf)

tri_cz = cz[triangles]
valid_tri = (tri_cz > 0.01).all(axis=1)

for ti in np.nonzero(valid_tri)[0]:
    a, b, c = triangles[ti]
    mat = tri_material[ti]
    pu = np.array([px[a], px[b], px[c]])
    pv = np.array([py[a], py[b], py[c]])
    pz = np.array([cz[a], cz[b], cz[c]])
    x_min = max(int(np.floor(pu.min())), 0)
    x_max = min(int(np.ceil(pu.max())), W - 1)
    y_min = max(int(np.floor(pv.min())), 0)
    y_max = min(int(np.ceil(pv.max())), H - 1)
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
    region_slice = depth[y_min:y_max + 1, x_min:x_max + 1]
    update = inside & (interp_z < region_slice)
    if not update.any():
        continue
    region_slice[update] = interp_z[update]

    col_a, col_b, col_c = color[a], color[b], color[c]
    gouraud = w0[..., None] * col_a + w1[..., None] * col_b + w2[..., None] * col_c
    img_gouraud[y_min:y_max + 1, x_min:x_max + 1][update] = gouraud[update]

    matcol = np.array(MATERIAL_DEBUG_COLORS.get(mat, (0, 0, 0)), dtype=np.float64)
    img_matdebug[y_min:y_max + 1, x_min:x_max + 1][update] = matcol

    if mat == 0 or uv is None or mat not in atlas_imgs:
        img_prod[y_min:y_max + 1, x_min:x_max + 1][update] = gouraud[update]
    else:
        atlas_img = atlas_imgs[mat]
        ah, aw = atlas_img.shape[:2]
        uva, uvb, uvc = uv[a], uv[b], uv[c]
        u_interp = w0 * uva[0] + w1 * uvb[0] + w2 * uvc[0]
        v_interp = w0 * uva[1] + w1 * uvb[1] + w2 * uvc[1]
        tx = np.clip((u_interp * (aw - 1)).astype(int), 0, aw - 1)
        ty = np.clip((v_interp * (ah - 1)).astype(int), 0, ah - 1)
        sampled = atlas_img[ty, tx]
        img_prod[y_min:y_max + 1, x_min:x_max + 1][update] = sampled[update]

def save(img, name):
    out = np.clip(img, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{OUT_DIR}/{name}", out_bgr)
    print("saved", name)

save(img_gouraud, "test_gouraud_vertexcolor.png")
save(img_matdebug, "test_materialindex_debug.png")
save(img_prod, "test_production_faithful.png")
