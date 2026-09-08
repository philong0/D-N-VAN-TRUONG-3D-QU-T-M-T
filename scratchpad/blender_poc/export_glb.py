"""Blender POC -- export the EXACT current production mesh/UV/texture (dense
nose W75 included, best-view texture, no geometry/reconstruction changes) as
a GLB Blender can import natively. Reads prod_response.json (a real, fresh
POST to the unmodified /fit-multiview-atlas endpoint) -- does not touch any
ai-engine/ or src/ file."""
import json, base64, io
import numpy as np
from PIL import Image
import trimesh

BPOC = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/blender_poc"

with open(f"{BPOC}/prod_response.json") as f:
    d = json.load(f)

fitted = np.array(d["fitted_positions"], dtype=np.float64).reshape(-1, 3)
vertex_colors_raw = np.array(d["vertex_colors"], dtype=np.float64).reshape(-1, 3)
V = fitted.shape[0]
print(f"fitted_positions: {V} vertices")

# D11 sRGB decode -- identical to Canvas3D.tsx's own decode for the base vertex-color layer
c = np.clip(vertex_colors_raw, 0, 255) / 255.0
vertex_colors_linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
vertex_colors_srgb_u8 = np.clip(vertex_colors_raw, 0, 255).astype(np.uint8)  # for glTF (expects sRGB-ish display color, no separate linear pass needed for vertex colors here)

tj = d["atlas"]["threejs"]
index = np.array(tj["index"], dtype=np.int64)
uv = np.array(tj["uv"], dtype=np.float64).reshape(-1, 2)
groups = tj["groups"]
extra_source_vids = tj["extra_source_vids"]
n_extra = len(extra_source_vids)
print(f"index: {len(index)} ({len(index)//3} triangles), uv: {uv.shape}, extra_source_vids: {n_extra}")

positions_full = np.zeros((V + n_extra, 3))
positions_full[:V] = fitted
colors_full = np.zeros((V + n_extra, 3), dtype=np.uint8)
colors_full[:V] = vertex_colors_srgb_u8
for i, src in enumerate(extra_source_vids):
    positions_full[V + i] = fitted[src]
    colors_full[V + i] = vertex_colors_srgb_u8[src]

# decode region textures to disk (real production best-view PNGs, base64-embedded
# because patient_id wasn't passed in this direct API call -- same real pixel
# data the production disk-file path would also produce)
region_textures = {}
for region, rd in d["atlas"]["regions"].items():
    png_bytes = base64.b64decode(rd["texture_png_base64"])
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    path = f"{BPOC}/tex_{region}.png"
    img.save(path)
    region_textures[region] = img
    print(f"  texture {region}: {img.size}")

scene = trimesh.Scene()
tri_index = index.reshape(-1, 3)

for g in groups:
    start_tri = g["start"] // 3
    count_tri = g["count"] // 3
    tris = tri_index[start_tri:start_tri + count_tri]
    used_vids = np.unique(tris)
    remap = {int(v): i for i, v in enumerate(used_vids)}
    local_tris = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in tris])
    local_pos = positions_full[used_vids]

    if g["materialIndex"] == 0:
        local_colors = colors_full[used_vids]
        mesh = trimesh.Trimesh(vertices=local_pos, faces=local_tris, vertex_colors=local_colors, process=False)
        mesh.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[255, 255, 255, 255], roughnessFactor=1.0, metallicFactor=0.0,
        )
    else:
        local_uv = uv[used_vids]
        region = g["region"]
        tex_img = region_textures[region]
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_img, roughnessFactor=1.0, metallicFactor=0.0)
        visual = trimesh.visual.TextureVisuals(uv=local_uv, material=material)
        mesh = trimesh.Trimesh(vertices=local_pos, faces=local_tris, visual=visual, process=False)

    scene.add_geometry(mesh, node_name=g["region"] if g["materialIndex"] else "base_vertex_color", geom_name=g["region"] if g["materialIndex"] else "base_vertex_color")

out_glb = f"{BPOC}/patient_mesh.glb"
scene.export(out_glb)
print(f"\nExported {out_glb}")

# also dump a small metadata json for the Blender script (camera framing needs
# a real per-patient center/radius, computed the SAME way GnmCameraFit does --
# facial-landmark-centroid + masked-frame radius -- reusing the ALREADY
# production-fetched fitted_positions, no new fitting/reconstruction)
import sys
sys.path.insert(0, "/home/ubuntu/dr-vantruong-3d-studio/ai-engine")
import gnm_identity_fit as gif
gif._ensure_loaded()
landmarks = gif.evaluate_points(fitted)
center = landmarks.mean(axis=0)

with open(f"{BPOC}/scene_meta.json", "w") as f:
    json.dump({"center": center.tolist(), "vertex_count": int(V + n_extra), "triangle_count": int(len(index)//3)}, f)
print("center:", center.tolist())
