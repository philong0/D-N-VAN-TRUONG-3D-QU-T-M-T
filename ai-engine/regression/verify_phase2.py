"""
Phase 2 regression gate — checks the eye sub-pipeline (gnm_eye_render.py) on
the same `post_colors.npy` produced by capture_compare_phase1.py for the 5
real patients (Phase 2's apply_eye_render runs unconditionally inside
bake_vertex_colors, so it's already present in that output).

PASS criteria (per the approved plan):
  - 100% of sclera/pupil/iris vertices get a color in a defined plausible
    range (sclera near-white, pupil near-black, iris in a plausible
    brown/hazel band or the real per-patient sampled mean).
  - 0% "skin-beige leak" — i.e. no sclera/pupil/iris vertex left at
    SENTINEL_UNCOVERED (the generic skin fallback), which the old
    D-eyefallback measured at 63-85% before this phase.
"""
import json
import sys
from pathlib import Path

import numpy as np

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ENGINE_DIR))
from gnm_eye_render import SCLERA_COLOR, PUPIL_COLOR, _load_eye_region_masks  # noqa: E402

from capture_baseline import PATIENTS  # noqa: E402

COMPARE_DIR = Path(__file__).resolve().parent / "compare"
SENTINEL_UNCOVERED = np.array([224, 172, 143], dtype=np.float64)  # same constant as gnm_texture_bake.py

# Plausible bounds, generous but real: sclera should read as off-white/light,
# pupil near-black, iris anywhere in a broad brown/hazel/green/blue-gray band
# (this project has no ground truth per-patient eye color to check against,
# so the check is "not skin-beige and not obviously wrong", not exact hue).
SCLERA_MIN_LUM = 180  # mean RGB well above skin tone
PUPIL_MAX_LUM = 60    # mean RGB well below skin tone


def main():
    masks = _load_eye_region_masks()
    sclera, iris, pupil = masks["sclera"], masks["iris"], masks["pupil"]

    all_pass = True
    report = []
    for patient_id in PATIENTS:
        d = COMPARE_DIR / patient_id
        colors = np.load(d / "post_colors.npy").astype(np.float64)

        sclera_lum = colors[sclera].mean(axis=1)
        pupil_lum = colors[pupil].mean(axis=1)

        sclera_ok = (sclera_lum >= SCLERA_MIN_LUM).mean()
        pupil_ok = (pupil_lum <= PUPIL_MAX_LUM).mean()

        # exact-match check: sclera/pupil are unconditionally overridden by
        # apply_eye_render, so they must be EXACTLY the procedural constant.
        sclera_exact = np.allclose(colors[sclera], SCLERA_COLOR, atol=1e-3)
        pupil_exact = np.allclose(colors[pupil], PUPIL_COLOR, atol=1e-3)

        # "beige leak" check: no eyeball vertex should equal SENTINEL_UNCOVERED
        beige_leak = 0
        for mask in (sclera, iris, pupil):
            beige_leak += int(np.all(np.isclose(colors[mask], SENTINEL_UNCOVERED, atol=1.0), axis=1).sum())

        # iris-minus-pupil: pupil is a real nested subset of iris (see
        # gnm_eye_render.py) and is deliberately darker than the surrounding
        # iris ring, so uniformity is only expected OUTSIDE the pupil subset.
        iris_only = iris & ~pupil
        iris_colors = colors[iris_only]
        iris_uniform = np.allclose(iris_colors, iris_colors[0], atol=1e-3)  # single mean color, by design

        patient_pass = sclera_exact and pupil_exact and beige_leak == 0 and iris_uniform
        if not patient_pass:
            all_pass = False

        row = {
            "patient_id": patient_id,
            "sclera_bright_fraction": float(sclera_ok),
            "pupil_dark_fraction": float(pupil_ok),
            "sclera_exact_match": bool(sclera_exact),
            "pupil_exact_match": bool(pupil_exact),
            "iris_color_used": iris_colors[0].tolist(),
            "iris_uniform": bool(iris_uniform),
            "beige_leak_vertex_count": beige_leak,
            "gate": "PASS" if patient_pass else "FAIL",
        }
        report.append(row)
        print(
            f"{patient_id}: sclera_exact={sclera_exact} pupil_exact={pupil_exact} "
            f"iris_color={iris_colors[0].round(1).tolist()} beige_leak={beige_leak} [{'PASS' if patient_pass else 'FAIL'}]"
        )

    out = {"patients": report, "overall": "PASS" if all_pass else "FAIL"}
    (Path(__file__).resolve().parent / "phase2_verify_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nOVERALL PHASE 2 EYE GATE: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
