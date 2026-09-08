"""
Offline test for Fix 1 (identity-mask docs only, no geometry cut change to
verify here), Fix 2 (background contamination), Fix 3 (jaw/ears seam) —
2026-08-24 visual-defect audit implementation.

Read-only against real patient data: imports `run_gnm_reconstruction` and
runs it directly against patient 257d9bfe-b246-4d82-a8c6-60a7ec076825's real
uploaded photos (same as the regression scripts' own discipline — no FastAPI
server, no HTTP, no restart of the live ai-engine process). Output is
written ONLY to scratchpad/ — the real
public/models/patients/257d9bfe.../reconstruction/ artifact is never
touched.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ai-engine"))

import cv2
import numpy as np

import main as m

PID = "257d9bfe-b246-4d82-a8c6-60a7ec076825"
PHOTOS_DIR = Path(__file__).parent.parent / ".data" / "patients" / PID / "photos"
OUT_DIR = Path(__file__).parent / "fix123_test_257d9bfe"
OUT_DIR.mkdir(exist_ok=True)

images = {}
for slot in ["angle1", "angle2", "angle3", "angle4"]:
    p = PHOTOS_DIR / f"{slot}.png"
    if p.exists():
        images[slot] = cv2.imread(str(p))

print(f"Loaded {len(images)} photos: {list(images.keys())}")

t0 = time.time()
result = m.run_gnm_reconstruction(images)
elapsed = time.time() - t0
print(f"run_gnm_reconstruction: ok={result['ok']} elapsed={elapsed:.1f}s warnings={result.get('warnings')}")

if not result["ok"]:
    print("FAILED:", result.get("error"))
    sys.exit(1)

fitted_positions = result["fitted_positions"]
vertex_color = result["vertex_colors"]
atlas = result["atlas"]

np.save(OUT_DIR / "positions.npy", fitted_positions)
np.save(OUT_DIR / "vertex_colors.npy", vertex_color)

# === Evidence 1: person_mask actually rejects background pixels ===
# Re-derive the person mask for angle2 (45°, one of the two views the
# original white/gray patches were measured on) and report coverage.
import gnm_texture_bake as gtb
from detect_pose import detect_pose

img2 = images["angle2"]
h, w = img2.shape[:2]
det = detect_pose(img2, is_profile_view=False)
mask = gtb.compute_person_silhouette_mask(det["landmarks_98"], (h, w))
print(f"angle2 person_mask: {mask.sum()}/{mask.size} px inside ({100*mask.sum()/mask.size:.1f}%)")
cv2.imwrite(str(OUT_DIR / "angle2_person_mask.png"), (mask.astype(np.uint8) * 255))
overlay = img2.copy()
overlay[~mask] = (overlay[~mask] * 0.3).astype(np.uint8)
cv2.imwrite(str(OUT_DIR / "angle2_mask_overlay.png"), overlay)

# === Evidence 2: temple/parotid vertex colors — background-like fraction ===
import zipfile
GNM = Path(__file__).parent.parent / "public" / "models" / "gnm" / "gnm_head_v3.npz"
with zipfile.ZipFile(GNM) as zf:
    names = list(np.load(zf.open("vertex_group_names.npy"), allow_pickle=True))
    groups = np.load(zf.open("vertex_groups.npy"), allow_pickle=True)


def region_mask(name):
    return groups[names.index(name)] > 0.5


for region in ["left_temple_region", "right_temple_region", "left_parotid_region", "right_parotid_region"]:
    m_ = region_mask(region)
    colors = vertex_color[m_]
    bg_like = (
        (colors[:, 0] > 170)
        & (colors[:, 0] < 235)
        & (np.abs(colors[:, 0] - colors[:, 1]) < 15)
        & (np.abs(colors[:, 1] - colors[:, 2]) < 15)
    )
    print(f"{region}: {bg_like.sum()}/{len(colors)} vertices background-like ({100*bg_like.sum()/len(colors):.1f}%)")

# === Evidence 3: atlas region coverage (jaw/ears still skipped as expected) ===
if atlas:
    print("atlas skipped:", atlas["skipped"])
    for region, data in atlas["regions"].items():
        if region not in ("left_temple", "right_temple", "left_parotid", "right_parotid"):
            continue
        tex = data["texture_png_bytes"]
        with open(OUT_DIR / f"{region}.png", "wb") as f:
            f.write(tex)
        print(f"atlas[{region}]: n_texels_covered={data['n_texels_covered']} tex={data['tex_w']}x{data['tex_h']}")

print("DONE — output in", OUT_DIR)
