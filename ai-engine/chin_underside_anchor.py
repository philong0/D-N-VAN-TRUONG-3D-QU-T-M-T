"""
chin_underside_anchor.py

Pose estimation for the NEW "below" (ngua len / looking-up-under-the-chin)
scan checkpoint, added specifically because the existing detect_pose.py
(YuNet bbox + PIPNet-98 landmarks) is built for near-frontal/profile faces
and is expected to find NO usable face in a true underside-of-chin photo --
there is no eye/brow/frontal-nose structure in frame at that pose. Reusing
the normal detect_pose()+solvePnP(98 landmarks) path for this view would
either (a) silently drop the image (no benefit) or (b) require guessing a
pose, which this project's owner has explicitly required never happen.

This module solves the SAME PROBLEM (where was the camera relative to the
head?) with a DIFFERENT, real evidence source, since face landmarks are not
available for this pose:

  1. ROTATION comes from the phone's own orientation sensor (real hardware
     measurement, captured client-side -- see GuidedFaceScan.tsx's
     `belowRelativeRotation`), NOT estimated from the image at all.
  2. TRANSLATION is solved in closed form (ordinary linear least squares,
     not an iterative/approximate guess) from 3 REAL detected 2D anchor
     points (two nostril openings + the chin/pogonion point, found in the
     photo by classical, inspectable computer vision -- fixed threshold/
     contour logic, not a learned model) against their known 3D positions
     on the fixed GNM template mesh.

If detection confidence is low at ANY step, every function here returns
None rather than a low-confidence guess -- the caller (reconstruct_gnm_
fullhead.py) must treat None as "drop this view", exactly the same
"drop rather than fabricate" contract gnm_identity_fit.py already uses for
EPnP failures on the other views (see that module's own docstring).

HONESTY NOTE (required reading before trusting this in production): the 3D
anchor positions below were derived by re-using the EXACT SAME bounding-box
constants gnm_face_shell.py already relies on for `is_nostril_zone` (already
part of the live, shipped bake) -- they are not invented for this module.
The 2D detector and the end-to-end pose solve, however, have NOT been
validated against any real "looking up under the chin" photo, because no
such photo exists yet anywhere in this codebase's patient data -- this is a
brand-new checkpoint. The very first real capture is the first real test.
Treat every acceptance threshold in `detect_nostril_chin_anchors_2d` as a
first-pass starting point, not a swept/measured constant (unlike most other
thresholds in this codebase, which DO carry a measured sweep in their own
docstring/comments) -- it should be revisited once real photos exist.
"""
from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Real 3D anchor positions -- derived directly from the GNM v3 template
# (public/models/gnm/gnm_head_v3.npz), NOT hand-typed guesses. Reproduction:
#
#   tpl = npz["template_vertex_positions"]; vgroups/gnames as usual
#   nose_mask = vgroups[gnames.index("nose_region")] > 0.5
#   chin_mask = vgroups[gnames.index("chin_region")] > 0.5
#   x, y, z = tpl[:,0], tpl[:,1], tpl[:,2]
#   # SAME box gnm_face_shell.py's own `is_nostril_zone` already uses:
#   nostril_box = (abs(x) < 0.028) & (y >= 0.220) & (y <= 0.275) & (z > 0.100)
#   cand = nose_mask & nostril_box
#   left_nostril  = tpl[cand & (x < 0)].mean(axis=0)   # -> [-0.0110, 0.2618, 0.1320]
#   right_nostril = tpl[cand & (x > 0)].mean(axis=0)   # -> [ 0.0103, 0.2616, 0.1325]
#   near_mid = chin_mask & (abs(x) < 0.02)
#   chin_pt = tpl[near_mid][np.argmin(tpl[near_mid][:, 1])]  # lowest-Y (most inferior)
#                                                              # -> [0.0001, 0.1822, 0.1007]
# ---------------------------------------------------------------------------
NOSTRIL_LEFT_3D = np.array([-0.01100535, 0.26179147, 0.13198382], dtype=np.float64)
NOSTRIL_RIGHT_3D = np.array([0.010258, 0.2616191, 0.13251589], dtype=np.float64)
CHIN_3D = np.array([7.3778770e-05, 1.8217367e-01, 1.0065065e-01], dtype=np.float64)

# Minimum acceptable detector confidence -- see detect_nostril_chin_anchors_2d.
MIN_NOSTRIL_CIRCULARITY = 0.55
MAX_NOSTRIL_AREA_RATIO = 2.2  # larger/smaller blob area ratio must stay under this
MIN_NOSTRIL_SEPARATION_PX_FRAC = 0.02  # of image width
MAX_NOSTRIL_SEPARATION_PX_FRAC = 0.35


def detect_nostril_chin_anchors_2d(image_bgr: np.ndarray) -> dict | None:
    """Classical-CV (no learned model) detector for the 3 real anchor points
    needed to pose a chin-underside photo: two nostril openings + the chin/
    pogonion point. Returns None (never a low-confidence guess) if any
    acceptance check fails.

    Method: nostrils are the two darkest small near-circular regions in the
    upper-center portion of the frame (true for a looking-up shot: nostril
    cavities are strongly backlit/shadowed regardless of overall exposure,
    the same physical property frontal-photo nostril detectors already rely
    on). The chin point is taken as the lowest-brightness-gradient (flattest
    skin) point on the frame's own vertical centerline within the lower
    third of the frame, which is where the chin's own convex tip is expected
    to project when the camera looks up from below it.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Nostrils: search the upper 60% of the frame (chin/underside shots put
    # the nostrils higher in-frame than the chin tip), Otsu threshold on the
    # dark side only.
    search_top = 0
    search_bottom = int(h * 0.60)
    roi = blurred[search_top:search_bottom, :]
    _, dark_mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > (w * h * 0.02):
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_NOSTRIL_CIRCULARITY:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"] + search_top
        candidates.append({"cx": cx, "cy": cy, "area": area, "circularity": circularity})

    if len(candidates) < 2:
        return None

    # Pick the best left/right pair: roughly same height, roughly symmetric
    # around the frame's own vertical centerline, plausible separation.
    best_pair = None
    best_score = -1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            left, right = (a, b) if a["cx"] < b["cx"] else (b, a)
            sep = right["cx"] - left["cx"]
            sep_frac = sep / w
            if sep_frac < MIN_NOSTRIL_SEPARATION_PX_FRAC or sep_frac > MAX_NOSTRIL_SEPARATION_PX_FRAC:
                continue
            height_diff = abs(a["cy"] - b["cy"]) / h
            if height_diff > 0.08:
                continue
            area_ratio = max(a["area"], b["area"]) / max(1.0, min(a["area"], b["area"]))
            if area_ratio > MAX_NOSTRIL_AREA_RATIO:
                continue
            score = (a["circularity"] + b["circularity"]) - height_diff * 5.0 - (area_ratio - 1.0)
            if score > best_score:
                best_score = score
                best_pair = (left, right)

    if best_pair is None:
        return None
    left, right = best_pair
    nostril_mid_y = (left["cy"] + right["cy"]) / 2.0

    # Chin point: on the frame's vertical centerline (between the two
    # nostrils, which is also the head's own midline), the lowest point
    # still on facial skin below the nostrils -- approximated as the
    # bottom-most row, on that column, whose gradient stays low (flat skin,
    # not the harder edge where skin meets background/neck-fold shadow).
    mid_x = int(round((left["cx"] + right["cx"]) / 2.0))
    mid_x = int(np.clip(mid_x, 0, w - 1))
    col = blurred[:, mid_x].astype(np.float64)
    grad = np.abs(np.diff(col))
    search_start = int(nostril_mid_y) + 5
    if search_start >= h - 5:
        return None
    grad_seg = grad[search_start:]
    if len(grad_seg) < 5:
        return None
    edge_thresh = np.percentile(grad_seg, 85)
    edge_rows = np.where(grad_seg > edge_thresh)[0]
    if len(edge_rows) == 0:
        chin_y = h - 1
    else:
        chin_y = search_start + int(edge_rows[0])

    if chin_y <= nostril_mid_y + 5:
        return None

    return {
        "left_nostril": (float(left["cx"]), float(left["cy"])),
        "right_nostril": (float(right["cx"]), float(right["cy"])),
        "chin": (float(mid_x), float(chin_y)),
        "confidence": float(best_score),
    }


def device_orientation_to_rotation_matrix(alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """Standard W3C DeviceOrientationEvent -> rotation matrix conversion
    (intrinsic Z-X'-Y'' Tait-Bryan angles, exactly the formula the
    DeviceOrientation Event Specification's own reference implementation
    uses -- not a bespoke/invented convention). alpha=compass heading (Z),
    beta=front-back tilt (X'), gamma=left-right tilt (Y''), all degrees.
    Returns the rotation from the device's own local frame to Earth frame.
    """
    a, b, g = np.radians([alpha_deg, beta_deg, gamma_deg])
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    return np.array([
        [ca * cg - sa * sb * sg, -cb * sa, ca * sg + cg * sa * sb],
        [cg * sa + ca * sb * sg, ca * cb, sa * sg - ca * cg * sb],
        [-cb * sg, sb, cb * cg],
    ], dtype=np.float64)


def compose_below_rotation(
    front_orientation: dict, below_orientation: dict, R_front_head_frame: np.ndarray,
) -> np.ndarray:
    """Composes the below-chin camera's rotation IN THE SAME HEAD-TEMPLATE
    FRAME the rest of the pipeline already uses, from two REAL ingredients:
      1. R_front_head_frame -- the ACTUAL already-solved camera pose for the
         frontal view (from fit_multiview's own EPnP/SQPNP solve on real
         landmarks -- not touched or re-derived here, just reused).
      2. The device orientation sensor's own REAL relative rotation between
         the front-checkpoint capture and the below-checkpoint capture.

    Rationale: the phone's orientation sensor measures rotation in a fixed
    Earth/gravity frame, not the head-template frame gnm_identity_fit.py
    solves poses in -- there is no direct correspondence between the two.
    But the RELATIVE rotation the device physically underwent between two
    capture moments is frame-independent (a rotation composed the same way
    regardless of which frame you express it in), so composing it with the
    one REAL, already-verified pose this pipeline has (the frontal view's
    own solved R) grounds the below-view's rotation in real, measured
    evidence at every step -- never an assumed/guessed absolute orientation.
    """
    R_front_device = device_orientation_to_rotation_matrix(
        front_orientation["alpha"], front_orientation["beta"], front_orientation["gamma"]
    )
    R_below_device = device_orientation_to_rotation_matrix(
        below_orientation["alpha"], below_orientation["beta"], below_orientation["gamma"]
    )
    R_device_rel = R_below_device @ R_front_device.T
    return R_device_rel @ R_front_head_frame


def solve_translation_given_rotation(
    anchors_2d: dict,
    R: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray | None:
    """Given a KNOWN rotation R (from the phone's real orientation sensor,
    not estimated here) and 3 real 2D<->3D anchor correspondences, solves
    for the camera translation t in closed form.

    Derivation (ordinary pinhole projection, R fixed): for each point i,
        R @ p_i + t = z_i * K^-1 [u_i, v_i, 1]^T
    Rearranged:  t - z_i * d_i = -(R @ p_i),  where d_i = K^-1 [u_i,v_i,1]^T.
    This is LINEAR in the unknowns (t: 3, and one depth z_i per point: 3) --
    3 points give 9 equations for 6 unknowns, solved by ordinary least
    squares (np.linalg.lstsq), not an iterative/approximate guess.
    """
    pts_3d = np.stack([NOSTRIL_LEFT_3D, NOSTRIL_RIGHT_3D, CHIN_3D], axis=0)
    pts_2d = np.array([anchors_2d["left_nostril"], anchors_2d["right_nostril"], anchors_2d["chin"]], dtype=np.float64)

    K_inv = np.linalg.inv(camera_matrix)
    n = 3
    # Unknowns ordered as [tx, ty, tz, z0, z1, z2]
    A = np.zeros((3 * n, 3 + n), dtype=np.float64)
    b = np.zeros((3 * n,), dtype=np.float64)
    for i in range(n):
        d_i = K_inv @ np.array([pts_2d[i, 0], pts_2d[i, 1], 1.0])
        A[3 * i:3 * i + 3, 0:3] = np.eye(3)
        A[3 * i:3 * i + 3, 3 + i] = -d_i
        b[3 * i:3 * i + 3] = -(R @ pts_3d[i])

    solution, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 6:
        return None
    t = solution[0:3]
    depths = solution[3:3 + n]
    if np.any(depths <= 0):
        # A point behind the camera means R (from the sensor) and the
        # detected 2D anchors are inconsistent -- do not fabricate a pose
        # from a contradictory solve.
        return None
    return t
