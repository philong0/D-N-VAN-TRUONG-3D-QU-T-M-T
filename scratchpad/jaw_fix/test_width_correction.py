import sys, os, glob
sys.path.insert(0, "/home/ubuntu/dr-vantruong-3d-studio/ai-engine")
os.chdir("/home/ubuntu/dr-vantruong-3d-studio")
import numpy as np
import cv2

import gnm_identity_fit as gif
from gnm_identity_fit import evaluate_points
from gnm_correspondence import WFLW_INDICES, VERTEX_INDICES, WEIGHTS

gif._ensure_loaded()
TEMPLATE = gif._TEMPLATE_POSITIONS

views = np.load("scratchpad/jaw_fix/views_cache.npy", allow_pickle=True)
fitted = np.load("scratchpad/jaw_fix/fitted_jw1.npy")  # baseline, jaw_weight=1 (production-equivalent)

row0 = WFLW_INDICES.index(0)
row32 = WFLW_INDICES.index(32)
p0_3d = sum(w * fitted[vid] for vid, w in zip(VERTEX_INDICES[row0], WEIGHTS[row0]))
p32_3d = sum(w * fitted[vid] for vid, w in zip(VERTEX_INDICES[row32], WEIGHTS[row32]))

frontal = next(v for v in views if v["slot"] == "angle1")
rvec, _ = cv2.Rodrigues(frontal["Rp"])
proj, _ = cv2.projectPoints(np.array([p0_3d, p32_3d]), rvec, frontal["tp"], frontal["cam"], np.zeros((4, 1)))
proj = proj.reshape(2, 2)
fitted_width_px = np.linalg.norm(proj[0] - proj[1])
real_width_px = np.linalg.norm(frontal["landmarks_98"][0] - frontal["landmarks_98"][32])
correction_factor = real_width_px / fitted_width_px
print(f"fitted_width_px={fitted_width_px:.1f} real_width_px={real_width_px:.1f} correction_factor={correction_factor:.4f}")

landmarks = evaluate_points(fitted)
center = landmarks.mean(axis=0)
center_x = center[0]
print("center_x:", center_x)

half_width_3d = abs((p0_3d[0] - p32_3d[0]) / 2)
print("half_width_3d:", half_width_3d)


def apply_width_correction(positions, center_x, half_width, correction_factor):
    dx = positions[:, 0] - center_x
    t = np.clip(np.abs(dx) / half_width, 0, 1)
    # smoothstep for a gentle, seamless transition (no hard edge)
    t = t * t * (3 - 2 * t)
    local_scale = 1.0 - (1.0 - correction_factor) * t
    out = positions.copy()
    out[:, 0] = center_x + dx * local_scale
    return out

corrected = apply_width_correction(fitted, center_x, half_width_3d, correction_factor)

# re-measure on corrected mesh to confirm the fix actually lands near ratio=1.0
p0c = sum(w * corrected[vid] for vid, w in zip(VERTEX_INDICES[row0], WEIGHTS[row0]))
p32c = sum(w * corrected[vid] for vid, w in zip(VERTEX_INDICES[row32], WEIGHTS[row32]))
projc, _ = cv2.projectPoints(np.array([p0c, p32c]), rvec, frontal["tp"], frontal["cam"], np.zeros((4, 1)))
projc = projc.reshape(2, 2)
corrected_width_px = np.linalg.norm(projc[0] - projc[1])
print(f"corrected_width_px={corrected_width_px:.1f} (target real={real_width_px:.1f}) new_ratio={corrected_width_px/real_width_px:.3f}")

np.save("scratchpad/jaw_fix/fitted_baseline.npy", fitted)
np.save("scratchpad/jaw_fix/fitted_corrected.npy", corrected)
np.save("scratchpad/jaw_fix/correction_meta.npy", np.array([center_x, half_width_3d, correction_factor]))
print("saved")
