"""
Phase 8 (8A-8D) -- dense nose enrichment, real 3DDFA_V2/BFM data injected into
GNM's own multi-view identity ridge solve as EXTRA landmark targets, on top of
the existing (unchanged) 47/48-point WFLW correspondence + nose x10 weight.

Provenance / validation trail (see .data or conversation notes for full
numbers -- summarized here so the "why" survives independent of this specific
session):
  - 8A: root cause -- gnm_correspondence.py represents the ENTIRE nose with
    exactly 1 real point (WFLW 57, weighted 10x since Phase 7). A single point
    cannot constrain bridge width/height/tip projection, only tip position.
  - 8B: naive per-image nearest-vertex correspondence (2D reprojection through
    an UNFITTED generic template) tested and REJECTED -- unstable across
    camera angles even for the already-trusted WFLW57 point itself (proved
    with a control test), so never shipped.
  - 8C: real fix -- 3DDFA_V2 (MIT-licensed, vendored in server/3ddfa_v2/,
    already used elsewhere in this project as the 3DDFA_V2 fallback pipeline)
    produces a DENSE per-vertex BFM shape regression, not sparse landmarks.
    Its own nose surface, aligned into GNM's coordinate space via a rigid
    Procrustes transform solved from 27 SHARED, already-trusted jaw+mouth
    landmarks (gnm_correspondence.py's own documented WFLW<->classic-68
    mapping), places BFM's nose-tip landmark within ~9mm of GNM's own
    already-trusted nose-tip vertex for the same real patient -- i.e. the two
    INDEPENDENT models agree, unlike 8B's failed attempt.
    A weight-budget sweep (15/30/50/75/100/150/200, spread over ~828 dense
    nose-surface points) found W=75 as the balanced operating point: MediaPipe
    reprojection (an INDEPENDENT detector from the WFLW/PIPNet one used
    elsewhere) improves at 45 deg by ~41% and 90 deg by ~47%, while jaw/mouth/
    eye/eyebrow residuals move by under 0.3mm and silhouette IoU stays flat or
    improves. Higher weights (150/200) buy almost nothing more at 45/90 deg
    (already within ~1px of the W=200 asymptote at W=75) while eroding the 0
    deg fit further past its own W=30 optimum -- W=75 was chosen specifically
    to avoid that diminishing-returns tail, not because it minimizes nose
    residual alone (it doesn't -- W=200 does, deliberately not chosen for that
    reason, see 8C's own report: "KHÔNG được chọn chỉ dựa vào nose residual").
  - 8D: A/B gate against real production (W=0, i.e. nose x10 only) on the same
    patient -- passed: MediaPipe combined (eyebrow+nose+mouth) reprojection
    improved at ALL four angles (0/45/90/below), WFLW-total (PIPNet, the same
    family of detector the pose itself is solved from) moved by under 0.25px
    at every angle, jaw/mouth/eye/eyebrow individually near-flat, max
    coefficient stayed at 2.31 (existing sanity ceiling is 2.9), max mesh
    displacement 12.9mm (existing sanity ceiling is 25mm).

This module is a pure ADDITIVE enhancement layered on top of an already-
complete fit_multiview() result -- never modifies gnm_identity_fit.py or
gnm_correspondence.py (nose x10 and every other existing weight/point are
untouched), and any failure anywhere in this file (3DDFA_V2 not runnable, BFM
fit fails, degenerate alignment, too few matched points) is caught by the
caller and simply skipped -- same "best-effort, never blocks the base
response" discipline as the Giai đoạn C feather-blend enhancement
(Canvas3D.tsx) and D24's neck/shoulder vertex-color fix in this same
ai-engine package.
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

import gnm_identity_fit as gif
from gnm_correspondence import WFLW_INDICES, POINT_WEIGHTS

REPO_ROOT = Path(__file__).parent.parent
TDDFA_DIR = REPO_ROOT / "server" / "3ddfa_v2"
TDDFA_SCRIPT = TDDFA_DIR / "reconstruct_patient.py"
BFM_PKL = TDDFA_DIR / "configs" / "bfm_noneck_v3.pkl"
GNM_ASSET = REPO_ROOT / "public" / "models" / "gnm" / "gnm_head_v3.npz"

# Phase 8C's validated operating point -- see this module's own docstring for
# the sweep that produced it. Total weight is spread evenly over however many
# dense nose points survive this patient's own BFM fit (see NOSE_PATCH_RADIUS_RAW).
#
# 2026-08-27 -- first tried 225 (3x): tripped this function's own
# maxCoeff>2.9 guard (measured maxCoeff=3.42 on patient 257d9bfe), so the
# enrichment was silently skipped and contributed 0.00mm -- confirmed by
# direct before/after vertex diff. IMPORTANT finding from isolating the
# cause (this session): the guard trip was NOT primarily from this weight --
# even the original validated W=75 also tripped it (maxCoeff~3.02) as long
# as gnm_correspondence.py's nose (WFLW57) POINT_WEIGHTS entry was boosted
# above its original x10. With that reverted to x10, BOTH W=75 and W=110
# pass cleanly (note=None) on patient 257d9bfe. Set to 110 here -- still
# above the W=75 validated operating point (this module's own docstring
# sweep topped out at W=200 and explicitly rejected it), so this specific
# value is NOT re-validated by that same sweep methodology; only confirmed
# property is it activates (doesn't get skipped) on 257d9bfe with nose kept
# at x10. Re-check maxCoeff on any other patient before trusting 110 there
# too. Pre-change (W=75) output backed up under .backups/nose-chin-weight-boost-*/.
DENSE_NOSE_WEIGHT_TOTAL = 110.0

# BFM-raw-unit radius around the BFM nose-landmark cluster (classic-68 rows
# 27-35) used to select the dense nose-surface patch -- calibrated in Phase 8C
# to select ~700-900 vertices (comparable density to GNM's own real
# `nose_region` vertex group, 889 members), not guessed.
NOSE_PATCH_RADIUS_RAW = 18000.0

# WFLW <-> classic-68 correspondence for jaw+mouth, transcribed VERBATIM from
# gnm_correspondence.py's own module docstring (the only two regions that
# file documents an explicit classic-68 row mapping for) -- used ONLY to
# rigidly align an independently-fit BFM mesh into GNM's coordinate space for
# THIS enrichment step; never added to, or read by, the real per-point
# correspondence table in gnm_correspondence.py itself.
_JAW_WFLW_TO_C68 = {0: 0, 8: 4, 16: 8, 24: 12, 28: 14, 30: 15, 32: 16}
_MOUTH_WFLW_TO_C68 = {w: 48 + (w - 76) for w in range(76, 96)}
_SHARED_WFLW_TO_C68 = {**_JAW_WFLW_TO_C68, **_MOUTH_WFLW_TO_C68}

_bfm_verts68_cache = None
_nose_region_vids_cache = None


def _bfm_verts68():
    """classic-68 index -> BFM (bfm_noneck_v3) vertex id, real data from the
    same pickle server/3ddfa_v2/TDDFA_ONNX.py already loads -- not guessed."""
    global _bfm_verts68_cache
    if _bfm_verts68_cache is None:
        with open(BFM_PKL, "rb") as f:
            bfm_pkl = pickle.load(f)
        _bfm_verts68_cache = bfm_pkl["keypoints"].reshape(68, 3)[:, 0] // 3
    return _bfm_verts68_cache


def _gnm_nose_region_vids():
    """Real GNM `nose_region` vertex group membership (889 members) -- same
    asset/group face-oval-mask.ts's hockey_mask reads, row 'nose_region'."""
    global _nose_region_vids_cache
    if _nose_region_vids_cache is None:
        with zipfile.ZipFile(GNM_ASSET) as zf:
            with zf.open("vertex_groups.npy") as f:
                groups = np.load(f)
            with zf.open("vertex_group_names.npy") as f:
                names = [str(n) for n in np.load(f, allow_pickle=True)]
        _nose_region_vids_cache = np.where(groups[names.index("nose_region")] > 0.5)[0]
    return _nose_region_vids_cache


def _run_3ddfa_multiview(view_images: dict) -> np.ndarray:
    """Runs the REAL, unmodified server/3ddfa_v2/reconstruct_patient.py as a
    subprocess (identical CLI contract src/app/api/3d-reconstruct/route.ts
    already uses in production for the 3DDFA_V2 fallback pipeline -- this is
    the second real call site, not a new/parallel implementation of that
    script's own logic) and returns its fitted BFM mesh's raw vertex
    positions. Raises on any failure -- caller is responsible for catching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        view_args = []
        for slot, img in view_images.items():
            path = os.path.join(tmpdir, f"{slot}.jpg")
            cv2.imwrite(path, img)
            view_args += ["--view", f"{slot}={path}"]
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        # D-3ddfavenv — was invoking bare "python3" (system interpreter, no
        # venv) instead of THIS process's own interpreter (`sys.executable`,
        # the ai-engine venv where trimesh/onnxruntime/etc are actually
        # installed) — the subprocess crashed on `import trimesh` before
        # printing anything, so `proc.stdout` was always empty and
        # `json.loads("")` failed with the uninformative
        # "Expecting value: line 1 column 1 (char 0)" every single time,
        # silently disabling dense-nose enrichment entirely (confirmed via
        # direct reproduction: `python3 -c "import trimesh"` fails,
        # `venv/bin/python -c "import trimesh"` succeeds). `sys.executable`
        # is also correct if this venv is ever renamed/relocated.
        proc = subprocess.run(
            [sys.executable, str(TDDFA_SCRIPT), *view_args, "--out-dir", out_dir],
            cwd=str(TDDFA_DIR), capture_output=True, text=True, timeout=30,
        )
        last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        if not last_line:
            raise RuntimeError(f"reconstruct_patient.py produced no stdout (exit={proc.returncode}); stderr: {proc.stderr[-500:]}")
        payload = json.loads(last_line)
        if not payload.get("ok"):
            raise RuntimeError(f"reconstruct_patient.py not ok: {payload}")
        glb_path = payload["glbPath"]

        import trimesh
        scene = trimesh.load(glb_path)
        mesh = scene.geometry[list(scene.geometry.keys())[0]]
        return np.array(mesh.vertices)


def enrich_with_dense_nose(fitted_positions: np.ndarray, view_inputs: list, view_images_by_slot: dict) -> tuple[np.ndarray, str | None]:
    """Best-effort dense-nose enrichment on top of an already-complete
    fit_multiview() result. Returns (positions, note) -- `positions` is either
    the enriched mesh or, on any failure, `fitted_positions` UNCHANGED;
    `note` is None on success or a short reason string when skipped (surfaced
    in the API response's `warnings` list, never raised to the caller).

    `view_inputs`: the SAME `ViewInput` list already passed to fit_multiview
    for this request (real WFLW-98 landmarks + camera matrix per view) --
    reused here to re-derive each view's own per-point mask/offset via the
    SAME primitives fit_multiview itself uses (point_mask/evaluate_basis/
    solve_epnp/backproject_pseudo3d/solve_procrustes/apply_rigid, all already
    defined in gnm_identity_fit.py) -- duplicated rather than exposed via a
    fit_multiview signature change, same "duplicate for zero risk to the
    already-validated function" precedent /fit-multiview-atlas's own
    docstring already establishes in main.py.
    `view_images_by_slot`: {slot: bgr ndarray} for the same views, needed to
    feed the real photos into 3DDFA_V2's own subprocess.
    """
    try:
        gif._ensure_loaded()
        TEMPLATE = gif._TEMPLATE_POSITIONS
        BASIS = gif._IDENTITY_BASIS
        identity_dim = BASIS.shape[0]

        bfm_positions = _run_3ddfa_multiview(view_images_by_slot)

        # ---- Procrustes-align BFM -> GNM space via 27 shared jaw+mouth points ----
        gnm_lm48 = gif.evaluate_points(fitted_positions)
        verts68 = _bfm_verts68()
        src, dst = [], []
        for wflw, c68 in _SHARED_WFLW_TO_C68.items():
            row = WFLW_INDICES.index(wflw)
            src.append(bfm_positions[verts68[c68]])
            dst.append(gnm_lm48[row])
        src = np.array(src)
        dst = np.array(dst)
        R, scale, t = gif.solve_procrustes(src, dst)
        anchor_residual_mm = float(np.linalg.norm(gif.apply_rigid(src, R, scale, t) - dst, axis=1).mean() * 1000)
        if anchor_residual_mm > 25:
            return fitted_positions, f"3ddfa-enrichment-skipped: alignment residual too high ({anchor_residual_mm:.1f}mm)"

        # ---- dense nose-surface patch, transformed into GNM space ----
        nose_center_raw = bfm_positions[verts68[27:36]].mean(axis=0)
        dist = np.linalg.norm(bfm_positions - nose_center_raw, axis=1)
        patch_ids = np.where(dist < NOSE_PATCH_RADIUS_RAW)[0]
        if len(patch_ids) < 100:
            return fitted_positions, f"3ddfa-enrichment-skipped: nose patch too small ({len(patch_ids)} pts)"
        patch_gnm_space = gif.apply_rigid(bfm_positions[patch_ids], R, scale, t)

        # ---- nearest GNM template nose_region vertex per dense point (3D-to-3D,
        #      both meshes already real+patient-specific+aligned -- NOT the
        #      2D-reprojection method 8B tested and rejected) ----
        nose_vids = _gnm_nose_region_vids()
        tree = cKDTree(TEMPLATE[nose_vids])
        nn_dist, nn = tree.query(patch_gnm_space)
        if float(np.mean(nn_dist)) * 1000 > 15:
            return fitted_positions, f"3ddfa-enrichment-skipped: nose match distance too high ({np.mean(nn_dist)*1000:.1f}mm)"
        matched_vids = nose_vids[nn]

        # ---- re-derive the ORIGINAL per-view masks/offsets (identical math to
        #      fit_multiview's own internal loop -- nose x10 and every other
        #      point/weight completely unchanged) ----
        template_landmarks_full = gif.evaluate_points(TEMPLATE)
        landmark_basis_full = gif.evaluate_basis()
        point_weights_arr = np.array(POINT_WEIGHTS)

        M = np.zeros((identity_dim, identity_dim))
        rhs = np.zeros(identity_dim)
        for v in view_inputs:
            from gnm_correspondence import point_mask as _point_mask
            mask = np.array(_point_mask(v.is_profile_view), dtype=bool)
            kept_wflw = [wf for wf, keep in zip(WFLW_INDICES, mask) if keep]
            points_2d = np.array([v.landmarks_98[i] for i in kept_wflw], dtype=np.float64)
            template_masked = template_landmarks_full[mask]
            pose = gif.solve_epnp(template_masked, points_2d, v.camera_matrix)
            if pose is None:
                continue
            Rp, tp = pose
            observed = gif.backproject_pseudo3d(points_2d, template_masked, Rp, tp, v.camera_matrix)
            Rproc, sproc, tproc = gif.solve_procrustes(observed, template_masked)
            aligned = gif.apply_rigid(observed, Rproc, sproc, tproc)
            offset = aligned - template_masked
            B_v = landmark_basis_full[:, mask, :].reshape(identity_dim, -1)
            w_ = np.repeat(point_weights_arr[mask], 3)
            Bw = B_v * w_[None, :]
            M += Bw @ B_v.T
            rhs += Bw @ offset.reshape(-1)

        # ---- add the dense nose extra targets, Phase 8C's validated weight ----
        extra_B = BASIS[:, matched_vids, :].reshape(identity_dim, -1)
        extra_offset = patch_gnm_space - TEMPLATE[matched_vids]
        n_dense = len(matched_vids)
        w_extra = np.repeat(np.full(n_dense, DENSE_NOSE_WEIGHT_TOTAL / n_dense), 3)
        Bw_extra = extra_B * w_extra[None, :]
        M += Bw_extra @ extra_B.T
        rhs += Bw_extra @ extra_offset.reshape(-1)

        M += gif.DEFAULT_REGULARIZATION * np.eye(identity_dim)
        coefficients = np.linalg.solve(M, rhs)

        max_coeff = float(np.abs(coefficients).max())
        if max_coeff > 2.9:
            return fitted_positions, f"3ddfa-enrichment-skipped: maxCoeff too high ({max_coeff:.2f})"

        enriched_positions = TEMPLATE + np.einsum("d,dvc->vc", coefficients, BASIS)
        return enriched_positions, None

    except Exception as err:  # noqa: BLE001 -- best-effort enhancement, never blocks the base response
        return fitted_positions, f"3ddfa-enrichment-failed: {type(err).__name__}: {err}"
