"""
Phase 2 — eye sub-pipeline, decoupled from the generic vertex-color bake.

Architecture principle (reverse-engineered from Crisalix, see file/var.docx's
`eyelids_manager.js`): the eye region is never left to the same raw-photo
bake as skin — Crisalix ships a dedicated per-patient eyelid mask + a bank
of pre-rendered eyelid texture variants, blended independently of the face
texture. This project has no equivalent per-patient eye photo bank to draw
on, so the equivalent-in-principle (not in code) choice here is: render
sclera/iris/pupil PROCEDURALLY, always, decoupled from whatever the generic
occlusion/ICM bake happened to sample for those specific vertices — never a
per-vertex photo-bake for the eyeball surface, covered or not.

Why "covered or not" (this is the actual Phase 2 change vs. the pre-existing
D-eyefallback block in gnm_texture_bake.py, which this module supersedes):
D-eyefallback already measured, on a real patient, that even eyeball
vertices bake_vertex_colors DOES mark as "covered" are noisy — std ~30-58
per channel, contaminated by grazing-angle eyelid-margin pixels bleeding in
at ordinary 0/45/90/below photo angles, since most of a rest-pose eyeball
sphere is genuinely self-occluded by the eyelid regardless of coverage
percentage. D-eyefallback only overrode the UNCOVERED vertices (63-85% of
sclera/pupil on the one measured patient) with a flat fallback color,
leaving the remaining "covered" fraction as raw (noisy) baked pixels — this
module removes that split entirely: sclera and pupil are ALWAYS the same
flat procedural color (no per-vertex photo data is trustworthy there at
all, covered or not), and iris is ALWAYS the same single mean color derived
from whatever real, uncontaminated iris coverage exists for this patient
(never per-vertex — a real iris is close to a uniform hue across its visible
area, so a single mean is a truthful summary, not a fabrication) — falling
back to a natural, clearly-generic brown only when this patient has zero
real iris coverage at all (never fabricated to mimic a specific unseen eye
color).
"""
import zipfile
from pathlib import Path

import numpy as np

GNM_ASSET = Path(__file__).parent.parent / "public" / "models" / "gnm" / "gnm_head_v3.npz"

# Same row indices already validated/used by gnm_texture_bake.py's own
# D-eyefallback (SCLERA_ROW/IRIS_ROW/PUPIL_ROW) — re-derived from the same
# source asset here rather than imported, to keep this module import-cycle-
# free (gnm_texture_bake.py calls into this module, so this module must not
# import gnm_texture_bake at module scope — same discipline already used by
# gnm_vertex_trust.py).
SCLERA_ROW = 19
IRIS_ROW = 20
PUPIL_ROW = 21
EYE_SOCKETS_ROW = 16
LEFT_ORBITAL_ROW = 32
RIGHT_ORBITAL_ROW = 33

SENTINEL_UNCOVERED = np.array([224, 172, 143], dtype=np.float64)

# Same two constants gnm_texture_bake.py's own D-eyefallback already
# validated (off-white, not blown-out 255; near-black, not pure 0) — reused
# verbatim since Phase 2 doesn't change WHAT the fallback colors should be,
# only WHEN they apply (always, not just when uncovered).
SCLERA_COLOR = np.array([232, 227, 220], dtype=np.float64)
PUPIL_COLOR = np.array([25, 22, 20], dtype=np.float64)
# Real iris coverage (per D-eyefallback's own measurement) is the ONE
# eyeball sub-region where real photo data is usable often enough (64% real
# coverage on the measured patient) to be worth sampling — this is only the
# LAST-RESORT default for a patient with literally zero real iris coverage
# at any angle: a clearly-generic mid-brown, not tuned to mimic any specific
# real eye color.
IRIS_DEFAULT_COLOR = np.array([120, 85, 60], dtype=np.float64)

_eye_masks_cache: dict[str, np.ndarray] | None = None
_mesh_adj_cache: list[set[int]] | None = None


def _load_eye_data() -> tuple[dict[str, np.ndarray], list[set[int]]]:
    global _eye_masks_cache, _mesh_adj_cache
    if _eye_masks_cache is None or _mesh_adj_cache is None:
        with zipfile.ZipFile(GNM_ASSET) as zf:
            with zf.open("vertex_groups.npy") as f:
                groups = np.load(f)
            with zf.open("triangles.npy") as f:
                triangles = np.load(f)
        _eye_masks_cache = {
            "sclera": groups[SCLERA_ROW] > 0.5,
            "iris": groups[IRIS_ROW] > 0.5,
            "pupil": groups[PUPIL_ROW] > 0.5,
            "eye_sockets": groups[EYE_SOCKETS_ROW] > 0.5,
            "orbital": (groups[LEFT_ORBITAL_ROW] > 0.5) | (groups[RIGHT_ORBITAL_ROW] > 0.5),
        }
        V = groups.shape[1]
        adj: list[set[int]] = [set() for _ in range(V)]
        for a, b, c in triangles:
            adj[a].update((b, c))
            adj[b].update((a, c))
            adj[c].update((a, b))
        _mesh_adj_cache = adj
    return _eye_masks_cache, _mesh_adj_cache


def _load_eye_region_masks() -> dict[str, np.ndarray]:
    masks, _ = _load_eye_data()
    return masks


def apply_eye_render(vertex_color: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """`vertex_color` (V,3): bake_vertex_colors' own output (already includes
    D24/trust-smoothing if the caller ran those first).
    1. Propagates verified patient skin color from surrounding orbital skin to
       uncovered eye_sockets vertices (eyelid margin) instead of leaving them at
       SENTINEL_UNCOVERED (#e0ac8f).
    2. Overrides eyeball rows (sclera/iris/pupil) with clean procedural colors.
    Returns a new array; does not mutate the input in place."""
    masks, adj = _load_eye_data()
    out = vertex_color.copy()

    # Step 1 — Eyelid skin fallback: eye_sockets vertices that missed direct camera
    # coverage receive real patient skin color propagated from adjacent orbital skin
    # (left_orbital_region / right_orbital_region / covered socket skin).
    # Eyeball vertices are strictly excluded from this propagation.
    sockets_mask = masks["eye_sockets"]
    eyeball_mask = masks["sclera"] | masks["iris"] | masks["pupil"]
    is_sentinel = np.all(np.isclose(out, SENTINEL_UNCOVERED, atol=1.0), axis=1)

    uncovered_sockets = sockets_mask & (is_sentinel | (~covered))
    if uncovered_sockets.any():
        resolved = (~is_sentinel) & covered & (~eyeball_mask)
        for _ in range(6):
            updates: dict[int, np.ndarray] = {}
            for vid in np.where(uncovered_sockets & (~resolved))[0]:
                nbr_colors = [out[n] for n in adj[vid] if resolved[n] and not eyeball_mask[n]]
                if nbr_colors:
                    updates[vid] = np.mean(nbr_colors, axis=0)
            if not updates:
                break
            for vid, c in updates.items():
                out[vid] = c
                resolved[vid] = True

    # Step 2 — Eyeball procedural rendering (sclera/iris/pupil)
    out[masks["sclera"]] = SCLERA_COLOR

    iris_mask = masks["iris"]
    iris_covered = iris_mask & covered
    if iris_covered.any():
        iris_color = vertex_color[iris_covered].mean(axis=0)
    else:
        iris_color = IRIS_DEFAULT_COLOR
    out[iris_mask] = iris_color

    # `pupils` is a NESTED SUBSET of `irises` (real GNM vertex-group data —
    # every pupil vertex is also an iris vertex, anatomically correct: a
    # pupil sits inside the iris). Pupil MUST be applied AFTER iris.
    out[masks["pupil"]] = PUPIL_COLOR

    return out
