"""
Phase 10, step 2 -- build, PER VIEW, the 3DDFA-V3 dense cheek/temple/forehead
targets in GNM space, from the already-cached (real, Phase-9-run)
angle{N}_extractTex.obj meshes. Each view is Procrustes-aligned into GNM
space INDEPENDENTLY (3DDFA-V3 has no native multi-view fusion -- see
FINAL_REPORT.md section C) via the SAME 27 shared WFLW<->classic-68
jaw+mouth points gnm_dense_nose.py already uses for 3DDFA_V2's nose
alignment, reused verbatim (imported, not duplicated) from that module.

Saves one dict per view: {region_name: (matched_gnm_vertex_ids, aligned_v3_points, distances_mm)}
"""
import pickle
import sys
sys.path.insert(0, "/home/ubuntu/dr-vantruong-3d-studio/ai-engine")

import numpy as np
import zipfile
from scipy.spatial import cKDTree

import gnm_identity_fit as gif
from gnm_correspondence import WFLW_INDICES
from gnm_dense_nose import _SHARED_WFLW_TO_C68

V3_RESULTS = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/phase9/results/3ddfa_v3_raw"
V3_FACE_MODEL = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/phase9/models/3DDFA-V3/assets/face_model.npy"
GNM_ASSET = "/home/ubuntu/dr-vantruong-3d-studio/public/models/gnm/gnm_head_v3.npz"
BASE_CACHE = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/phase10/results/base_fit_cache.pkl"
OUT = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/phase10/results/v3_dense_targets.pkl"

REGIONS = ["forehead_region", "left_temple_region", "right_temple_region",
           "left_cheek_region", "right_cheek_region"]

ANGLE_SLOTS = ["angle1", "angle2", "angle3", "angle4"]

# Per-region, per-point match-distance gate (mm) -- points whose nearest
# aligned-V3 vertex is farther than this are dropped (occlusion / bad local
# alignment), same discipline as gnm_dense_nose's nose match-distance gate
# (there: a single mean-based gate at 15mm; here: per-point, since a region
# straddling a profile view's silhouette can have PART of it well-observed
# and part occluded -- dropping the whole region on one bad mean would waste
# real good data on the visible side).
MATCH_GATE_MM = 20.0


def load_obj(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(verts)


def main():
    with open(BASE_CACHE, "rb") as f:
        cache = pickle.load(f)

    gif._ensure_loaded()
    TEMPLATE = gif._TEMPLATE_POSITIONS

    with zipfile.ZipFile(GNM_ASSET) as zf:
        with zf.open("vertex_groups.npy") as f:
            groups = np.load(f)
        with zf.open("vertex_group_names.npy") as f:
            names = [str(n) for n in np.load(f, allow_pickle=True)]
    region_vids = {r: np.where(groups[names.index(r)] > 0.5)[0] for r in REGIONS}
    for r, v in region_vids.items():
        print(r, len(v), "members")

    v3_ldm68 = np.load(V3_FACE_MODEL, allow_pickle=True).item()["ldm68"]  # (68,) vertex ids, verified via correlation=0.9995

    # fitted GNM-space landmark positions (same target for every view's alignment,
    # since it's the already joint-fitted identity -- consistent across views)
    fitted_positions = cache["prod_final_positions"]  # use current-production final mesh as the alignment target (post nose+width)
    gnm_lm48 = gif.evaluate_points(fitted_positions)

    per_view = {}
    for i, slot in enumerate(cache["view_slots"]):
        obj_path = f"{V3_RESULTS}/{slot}/{slot}_extractTex.obj"
        v3_verts = load_obj(obj_path)
        print(f"\n{slot}: {v3_verts.shape[0]} verts loaded from {obj_path}")

        src, dst = [], []
        for wflw, c68 in _SHARED_WFLW_TO_C68.items():
            row = WFLW_INDICES.index(wflw)
            src.append(v3_verts[v3_ldm68[c68]])
            dst.append(gnm_lm48[row])
        src = np.array(src)
        dst = np.array(dst)
        R, scale, t = gif.solve_procrustes(src, dst)
        anchor_residual_mm = float(np.linalg.norm(gif.apply_rigid(src, R, scale, t) - dst, axis=1).mean() * 1000)
        print(f"  alignment anchor residual: {anchor_residual_mm:.2f}mm (27 shared jaw+mouth pts)")

        aligned_v3 = gif.apply_rigid(v3_verts, R, scale, t)

        tree = cKDTree(aligned_v3)
        region_data = {}
        for r in REGIONS:
            vids = region_vids[r]
            target_pts = TEMPLATE[vids]
            dist, nn = tree.query(target_pts)
            dist_mm = dist * 1000
            keep = dist_mm < MATCH_GATE_MM
            kept_vids = vids[keep]
            kept_pts = aligned_v3[nn[keep]]
            region_data[r] = {
                "vids": kept_vids,
                "aligned_pts": kept_pts,
                "mean_dist_mm": float(dist_mm[keep].mean()) if keep.sum() else None,
                "n_total": len(vids),
                "n_kept": int(keep.sum()),
            }
            print(f"  {r}: {keep.sum()}/{len(vids)} kept, mean match dist {region_data[r]['mean_dist_mm']}")

        per_view[slot] = {
            "anchor_residual_mm": anchor_residual_mm,
            "regions": region_data,
        }

    with open(OUT, "wb") as f:
        pickle.dump(per_view, f)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
