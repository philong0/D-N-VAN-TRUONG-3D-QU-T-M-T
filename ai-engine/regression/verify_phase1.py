"""
Phase 1 regression gate — diffs Phase 0 baseline vs Phase 1 capture for the
same 5 real patients. Hard gate (per the approved plan's PASS criteria):
face_mask positions/colors must be EXACTLY unchanged. Soft check: shell
(non-face_mask, non-ear) color variance should not increase.
"""
import json
import sys
from pathlib import Path

import numpy as np

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ENGINE_DIR))
from gnm_vertex_trust import compute_face_mask  # noqa: E402

from capture_baseline import PATIENTS  # noqa: E402

BASELINE_DIR = Path(__file__).resolve().parent / "baseline"
PHASE1_DIR = Path(__file__).resolve().parent / "phase1"


def _load_vertex_group(name: str) -> np.ndarray:
    import zipfile
    asset = AI_ENGINE_DIR.parent / "public" / "models" / "gnm" / "gnm_head_v3.npz"
    with zipfile.ZipFile(asset) as zf:
        with zf.open("vertex_group_names.npy") as f:
            names = list(np.load(f, allow_pickle=True))
        with zf.open("vertex_groups.npy") as f:
            groups = np.load(f, allow_pickle=True)
    return groups[names.index(name)] > 0.5


def main():
    face_mask = compute_face_mask(17821)
    ears = _load_vertex_group("ears")
    shell = ~face_mask  # everything outside the protected core, including ears

    all_pass = True
    report = []

    for patient_id in PATIENTS:
        b_dir = BASELINE_DIR / patient_id
        p_dir = PHASE1_DIR / patient_id
        if not b_dir.exists() or not p_dir.exists():
            print(f"[SKIP] {patient_id}: missing baseline or phase1 dir")
            all_pass = False
            continue

        pos_b = np.load(b_dir / "fitted_positions.npy")
        pos_p = np.load(p_dir / "fitted_positions.npy")
        col_b = np.load(b_dir / "vertex_colors.npy")
        col_p = np.load(p_dir / "vertex_colors.npy")

        pos_diff_face = np.abs(pos_b[face_mask] - pos_p[face_mask]).max()
        col_diff_face = np.abs(col_b[face_mask] - col_p[face_mask]).max()

        pos_diff_shell = np.abs(pos_b[shell] - pos_p[shell]).max()
        pos_diff_shell_mean = np.abs(pos_b[shell] - pos_p[shell]).mean()

        std_b_shell = col_b[shell].std()
        std_p_shell = col_p[shell].std()
        std_b_ears = col_b[ears].std()
        std_p_ears = col_p[ears].std()

        # Hard gate: face_mask must be EXACTLY unchanged (allow only float32
        # round-trip epsilon from the .npy save/load, not real numerical drift)
        HARD_EPS = 1e-5
        gate_pos = pos_diff_face <= HARD_EPS
        gate_col = col_diff_face <= HARD_EPS
        patient_pass = gate_pos and gate_col

        if not patient_pass:
            all_pass = False

        row = {
            "patient_id": patient_id,
            "face_mask_max_pos_diff": float(pos_diff_face),
            "face_mask_max_color_diff": float(col_diff_face),
            "shell_max_pos_diff_m": float(pos_diff_shell),
            "shell_mean_pos_diff_m": float(pos_diff_shell_mean),
            "shell_color_std_baseline": float(std_b_shell),
            "shell_color_std_phase1": float(std_p_shell),
            "shell_color_std_delta": float(std_p_shell - std_b_shell),
            "ears_color_std_baseline": float(std_b_ears),
            "ears_color_std_phase1": float(std_p_ears),
            "ears_color_std_delta": float(std_p_ears - std_b_ears),
            "face_mask_gate": "PASS" if patient_pass else "FAIL",
        }
        report.append(row)
        print(
            f"{patient_id}: face_mask pos_diff={pos_diff_face:.2e} color_diff={col_diff_face:.2e} "
            f"[{'PASS' if patient_pass else 'FAIL'}] | shell pos moved mean={pos_diff_shell_mean*1000:.2f}mm max={pos_diff_shell*1000:.2f}mm | "
            f"shell color std {std_b_shell:.2f}->{std_p_shell:.2f} | ears color std {std_b_ears:.2f}->{std_p_ears:.2f}"
        )

    out = {"patients": report, "overall": "PASS" if all_pass else "FAIL"}
    (Path(__file__).resolve().parent / "phase1_verify_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nOVERALL PHASE 1 FACE_MASK GATE: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
