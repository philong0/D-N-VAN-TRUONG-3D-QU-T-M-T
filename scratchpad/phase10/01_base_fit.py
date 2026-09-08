"""
Phase 10, step 1 -- real PIPNet detection + fit_multiview + CURRENT PRODUCTION
dense-nose enrichment (unchanged) for patient dca63e1a, using the exact same
call sequence main.py's /fit-multiview-atlas uses. Caches everything needed
for the weight-sweep step so we don't re-run PIPNet/EPNP repeatedly.
"""
import pickle
import sys
sys.path.insert(0, "/home/ubuntu/dr-vantruong-3d-studio/ai-engine")

import cv2
import numpy as np

from detect_pose import detect_pose
from gnm_identity_fit import fit_multiview, ViewInput
from gnm_dense_nose import enrich_with_dense_nose
from gnm_width_correction import apply_width_correction

sys.path.insert(0, "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/phase10")
from phase10_lib import detect_mediapipe

PATIENT_DIR = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/phase9/patient_dca63e1a_in"
OUT = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/phase10/results/base_fit_cache.pkl"

ANGLE_SLOTS = ["angle1", "angle2", "angle3", "angle4"]
EXT = {"angle1": "png", "angle2": "png", "angle3": "jpg", "angle4": "png"}


def main():
    images = {}
    for slot in ANGLE_SLOTS:
        path = f"{PATIENT_DIR}/{slot}.{EXT[slot]}"
        img = cv2.imread(path)
        assert img is not None, path
        images[slot] = img

    view_inputs = []
    view_slots = []
    view_images = []
    warnings = []
    for slot, image_bgr in images.items():
        is_profile = slot == "angle3"
        h, w = image_bgr.shape[:2]
        result = detect_pose(image_bgr, is_profile_view=is_profile)
        if not result["has_face"]:
            warnings.append(f"{slot}: no face detected")
            continue
        camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        view_inputs.append(ViewInput(result["landmarks_98"], camera_matrix, is_profile, (h, w)))
        view_slots.append(slot)
        view_images.append(image_bgr)

    print("views used:", view_slots, "warnings:", warnings)

    fitted_positions, per_view_pose, fit_warnings = fit_multiview(view_inputs)
    print("fit_multiview warnings:", fit_warnings)
    assert fitted_positions is not None

    # current production: dense-nose enrichment (W=75, unchanged) then width correction
    prod_positions, dense_nose_note = enrich_with_dense_nose(
        fitted_positions, view_inputs, dict(zip(view_slots, view_images))
    )
    print("dense_nose_note:", dense_nose_note)

    prod_positions_final, width_note = apply_width_correction(prod_positions, view_inputs, view_slots)
    print("width_note:", width_note)

    mp_points = {}
    for slot, img in zip(view_slots, view_images):
        pts = detect_mediapipe(img)
        print(f"{slot}: mediapipe face detected = {pts is not None}")
        mp_points[slot] = pts

    cache = {
        "view_slots": view_slots,
        "landmarks_98": [v.landmarks_98 for v in view_inputs],
        "camera_matrix": [v.camera_matrix for v in view_inputs],
        "is_profile_view": [v.is_profile_view for v in view_inputs],
        "image_shape": [v.image_shape for v in view_inputs],
        "base_fitted_positions": fitted_positions,          # pre-nose, pre-width (raw fit_multiview)
        "prod_dense_nose_positions": prod_positions,          # after dense-nose, before width
        "prod_final_positions": prod_positions_final,         # full current production output
        "dense_nose_note": dense_nose_note,
        "width_note": width_note,
        "per_view_pose": per_view_pose,  # [(R, t, mask) or None] aligned to view_slots order
        "mp_points_px": mp_points,       # {slot: (478,2) or None}
    }
    with open(OUT, "wb") as f:
        pickle.dump(cache, f)
    print("saved", OUT)


if __name__ == "__main__":
    main()
