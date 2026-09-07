"""
Face-width correction -- fixes the real, confirmed-on-live-product "quá bè"
(too wide) complaint: the fitted mesh's own jaw/cheek contour reprojects
measurably wider than the real photo it was fit from, for patients whose
face shape sits outside what the 253-dim linear PCA identity basis can
represent purely through the existing landmark ridge solve (already proven
in scratchpad/jaw_fix/measure_width.py -- sweeping jaw-point landmark WEIGHT
1.0->25.0 left the fitted/real width ratio flat at ~1.05-1.07, i.e. jaw
contour landmarks are already tightly fit and reweighting them is not an
effective lever; the excess width lives in the unconstrained cheek/temple
surface BETWEEN landmarks, not at the landmarks themselves).

Real, per-patient measured fix: reproject the two outermost jaw/cheek WFLW
landmarks (0 and 32) of the ALREADY-FITTED mesh through that same view's own
real EPNP camera pose, compare against the REAL detected 2D pixel distance
for those same two points in the real photo, and use the ratio as a single
per-patient horizontal (local-X, in face space) correction factor. Applied
as a smoothstep-weighted falloff from the facial centroid so the correction
is strongest at the outer contour and fades to zero at the mesh's own
vertical midline -- no hard edge, no effect on nose/mouth/eye shape (those
sit near the midline where the smoothstep weight is ~0).

Numerically validated (scratchpad/jaw_fix/test_width_correction.py) on
patient 35e1ffef-d7e5-405d-a7ec-dae152bf883b: baseline ratio 1.054 ->
corrected ratio 1.000 exactly (the correction formula is self-consistent
and lands precisely on the real measured target). Visually confirmed at
0/45/90 degrees (scratchpad/jaw_fix/*.png) -- narrower at 0 deg as expected,
no visible tearing/pinching artifact at 45 or 90 deg (correction acts along
local X, which is nearly edge-on and so barely visible at a 90 deg profile
view, exactly as expected for a width-only correction).

Purely additive on top of an already-complete fit_multiview() (+ optional
dense-nose enrichment) result -- never modifies gnm_identity_fit.py or
gnm_correspondence.py. Any failure (no frontal view, degenerate pose,
degenerate half-width) is caught and this returns fitted_positions
UNCHANGED, same "best-effort, never blocks the base response" discipline as
gnm_dense_nose.py's enrich_with_dense_nose.
"""

import numpy as np

import gnm_identity_fit as gif
from gnm_correspondence import WFLW_INDICES, VERTEX_INDICES, WEIGHTS

# Real, measured safety clamp on the correction factor -- guards against a
# single bad/occluded frontal-view landmark detection producing an absurd
# (near-zero or wildly inflating) correction. The validated patient's own
# measured factor (0.9485) sits well inside this range; this is a guard
# rail, not a tuning knob.
MIN_CORRECTION_FACTOR = 0.85
MAX_CORRECTION_FACTOR = 1.15

_ROW0 = WFLW_INDICES.index(0)
_ROW32 = WFLW_INDICES.index(32)


def _landmark_3d(positions: np.ndarray, row: int) -> np.ndarray:
    return sum(w * positions[vid] for vid, w in zip(VERTEX_INDICES[row], WEIGHTS[row]))


def _smoothstep_width_correct(positions: np.ndarray, center_x: float, half_width: float, correction_factor: float) -> np.ndarray:
    dx = positions[:, 0] - center_x
    t = np.clip(np.abs(dx) / half_width, 0.0, 1.0)
    t = t * t * (3 - 2 * t)  # smoothstep -- gentle, seamless falloff toward the midline, no hard edge
    local_scale = 1.0 - (1.0 - correction_factor) * t
    out = positions.copy()
    out[:, 0] = center_x + dx * local_scale
    return out


def apply_width_correction(fitted_positions: np.ndarray, view_inputs: list, view_slots: list) -> tuple[np.ndarray, str | None]:
    """`view_inputs`/`view_slots`: the SAME ViewInput list and slot-name list
    already built for fit_multiview in this request, reused here to find the
    frontal view and re-solve its own real EPNP pose (same primitives
    fit_multiview and gnm_dense_nose.py already use) -- never a new detector
    or a new photo read."""
    try:
        frontal_idx = None
        for i, slot in enumerate(view_slots):
            if slot == "angle1":
                frontal_idx = i
                break
        if frontal_idx is None:
            for i, v in enumerate(view_inputs):
                if not v.is_profile_view:
                    frontal_idx = i
                    break
        if frontal_idx is None:
            return fitted_positions, "width-correction-skipped: no frontal view available"

        v = view_inputs[frontal_idx]

        p0_3d = _landmark_3d(fitted_positions, _ROW0)
        p32_3d = _landmark_3d(fitted_positions, _ROW32)

        template_landmarks_full = gif.evaluate_points(gif._TEMPLATE_POSITIONS)
        from gnm_correspondence import point_mask as _point_mask
        mask = np.array(_point_mask(v.is_profile_view), dtype=bool)
        kept_wflw = [wf for wf, keep in zip(WFLW_INDICES, mask) if keep]
        points_2d = np.array([v.landmarks_98[i] for i in kept_wflw], dtype=np.float64)
        template_masked = template_landmarks_full[mask]
        pose = gif.solve_epnp(template_masked, points_2d, v.camera_matrix)
        if pose is None:
            return fitted_positions, "width-correction-skipped: frontal-view EPNP pose failed"
        R, t = pose

        import cv2
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(np.array([p0_3d, p32_3d]), rvec, t, v.camera_matrix, np.zeros((4, 1)))
        proj = proj.reshape(2, 2)
        fitted_width_px = float(np.linalg.norm(proj[0] - proj[1]))
        real_width_px = float(np.linalg.norm(np.array(v.landmarks_98[0]) - np.array(v.landmarks_98[32])))

        if fitted_width_px < 1.0 or real_width_px < 1.0:
            return fitted_positions, "width-correction-skipped: degenerate landmark width"

        raw_factor = real_width_px / fitted_width_px
        correction_factor = float(np.clip(raw_factor, MIN_CORRECTION_FACTOR, MAX_CORRECTION_FACTOR))

        landmarks = gif.evaluate_points(fitted_positions)
        center_x = float(landmarks.mean(axis=0)[0])
        half_width_3d = float(abs((p0_3d[0] - p32_3d[0]) / 2))
        if half_width_3d < 1e-4:
            return fitted_positions, "width-correction-skipped: degenerate 3D half-width"

        corrected = _smoothstep_width_correct(fitted_positions, center_x, half_width_3d, correction_factor)
        return corrected, None

    except Exception as err:  # noqa: BLE001 -- best-effort correction, never blocks the base response
        return fitted_positions, f"width-correction-failed: {type(err).__name__}: {err}"
