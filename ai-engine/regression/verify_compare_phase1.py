"""
Authoritative Phase 1 regression gate — diffs `pre` vs `post` from
capture_compare_phase1.py (same-process, same fit, no dense-nose subprocess
confound). Hard gate: face_mask positions AND colors must be EXACTLY
unchanged. Soft check: shell/ears color variance should not increase.
"""
import json
import sys
from pathlib import Path

import numpy as np

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ENGINE_DIR))
from gnm_vertex_trust import compute_face_mask  # noqa: E402

from capture_baseline import PATIENTS  # noqa: E402

COMPARE_DIR = Path(__file__).resolve().parent / "compare"


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
    shell = ~face_mask

    all_pass = True
    report = []
    HARD_EPS = 1e-5

    for patient_id in PATIENTS:
        d = COMPARE_DIR / patient_id
        if not d.exists():
            print(f"[SKIP] {patient_id}: missing compare dir")
            all_pass = False
            continue

        pre_pos = np.load(d / "pre_positions.npy")
        post_pos = np.load(d / "post_positions.npy")
        pre_col = np.load(d / "pre_colors.npy")
        post_col = np.load(d / "post_colors.npy")

        pos_diff_face = float(np.abs(pre_pos[face_mask] - post_pos[face_mask]).max())
        col_diff_face = float(np.abs(pre_col[face_mask] - post_col[face_mask]).max())

        pos_diff_shell_mean = float(np.abs(pre_pos[shell] - post_pos[shell]).mean()) * 1000
        pos_diff_shell_max = float(np.abs(pre_pos[shell] - post_pos[shell]).max()) * 1000

        std_pre_shell, std_post_shell = float(pre_col[shell].std()), float(post_col[shell].std())
        std_pre_ears, std_post_ears = float(pre_col[ears].std()), float(post_col[ears].std())

        gate = (pos_diff_face <= HARD_EPS) and (col_diff_face <= HARD_EPS)
        if not gate:
            all_pass = False

        report.append({
            "patient_id": patient_id,
            "face_mask_max_pos_diff": pos_diff_face,
            "face_mask_max_color_diff": col_diff_face,
            "shell_mean_pos_move_mm": pos_diff_shell_mean,
            "shell_max_pos_move_mm": pos_diff_shell_max,
            "shell_color_std_pre": std_pre_shell,
            "shell_color_std_post": std_post_shell,
            "ears_color_std_pre": std_pre_ears,
            "ears_color_std_post": std_post_ears,
            "gate": "PASS" if gate else "FAIL",
        })
        print(
            f"{patient_id}: face_mask pos_diff={pos_diff_face:.2e} color_diff={col_diff_face:.2e} [{'PASS' if gate else 'FAIL'}] | "
            f"shell moved mean={pos_diff_shell_mean:.2f}mm max={pos_diff_shell_max:.2f}mm | "
            f"shell color std {std_pre_shell:.2f}->{std_post_shell:.2f} | ears color std {std_pre_ears:.2f}->{std_post_ears:.2f}"
        )

    out = {"patients": report, "overall": "PASS" if all_pass else "FAIL"}
    (Path(__file__).resolve().parent / "phase1_compare_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nOVERALL PHASE 1 FACE_MASK GATE (confound-free): {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
