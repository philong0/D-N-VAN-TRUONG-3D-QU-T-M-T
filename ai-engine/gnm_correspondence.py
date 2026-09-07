"""
WFLW-98 <-> GNM Head v3 point correspondence — the full 48-point set
validated this round (D1/D1.5/D2/D2.5, see scratchpad/map_pipnet_gnm*.py,
scratchpad/d25_*.py for the empirical derivation/validation each point went
through — reprojection error measured on 2 real patients across all 4
angles before any point here was trusted). Ported verbatim from those
results, not re-derived.

Regions:
  - jaw (7): WFLW 0/8/16/24/28/30/32 — head_sparse_68.txt rows 0/4/8/12/14/15/16.
    WFLW 4/6/10/12 were tested and dropped (50-100px error at both 0 and
    90 deg — see reprojection_error_90deg.py's own results).
  - nose (1): WFLW 57 — reused from src/lib/gnm/landmark-correspondence.ts's
    own verified STATIC_LANDMARKS nose-tip entry (mpIndices=[5,4]).
  - mouth (20): WFLW 76-95 <-> classic-68 rows 48-67, direct order-preserving
    map (counts match exactly: 12 outer + 8 inner) — the best-performing
    region in every validation round.
  - eye (10): WFLW 60/64/68/72 (corners, classic-68 rows 36/39/42/45) +
    61/63/65/67/69/70/73/75 (contour, found via nearest-neighbor match to
    src/lib/gnm/landmark-correspondence.ts's own already-verified
    EYE_LANDMARK_GNM 3D points — see scratchpad/d25_find_correspondence.py).
    NOTE: WFLW 60/72 (OUTER eye corners) were tested and found unreliable at
    90 deg specifically (22-28px error both patients) — replaced by keeping
    only the INNER corners (64/68) + the 8 contour points above, none of
    which showed the same 90-degree degradation.
  - eyebrow (10): WFLW 33/36/37/40/41/42/46 (reliable at every angle) +
    43/47/48 (reliable at 0/45/below, degrades specifically at 90 deg —
    physical far-side occlusion in a true profile shot, same class of issue
    as the eye outer corners) — found via nearest-neighbor match to
    src/lib/gnm/landmark-correspondence.ts's own verified STATIC_LANDMARKS
    eyebrow entries.

`excl_90` marks points to DROP when processing a photo from the "90°" upload
slot (angle3) — per this round's explicit physical-occlusion principle: only
require landmarks that are actually visible, never force occluded ones.
"""

# Each entry: (wflw_index, [3 GNM vertex indices], [3 barycentric weights], exclude_at_90)
POINTS = [
    # --- jaw (7) ---
    (0, [8777, 8841, 11165], [0.068, 0.079, 0.853], False),
    (8, [6846, 6849, 8028], [0.866, 0.102, 0.031], False),
    (16, [9975, 12257, 12258], [0.156, 0.138, 0.706], False),
    (24, [1900, 721, 718], [0.031, 0.102, 0.866], False),
    (28, [3717, 3956, 3957], [0.859, 0.057, 0.084], False),
    (30, [3708, 3707, 3710], [0.804, 0.045, 0.151], False),
    (32, [5037, 2713, 2649], [0.853, 0.079, 0.068], False),
    # --- nose (1) ---
    (57, [12287, 12296, 10055], [0.0, 0.997, 0.003], False),
    # --- mouth (20), WFLW 76-95 <-> classic-68 rows 48-67 ---
    # (vertex/weight data identical to scratchpad/map_pipnet_gnm_d1_result.json "mouth")
    (76, [10683, 10653, 10652], [0.409, 0.584, 0.006], False),
    (77, [10720, 10719, 10673], [0.0, 0.996, 0.004], False),
    (78, [10732, 10731, 10672], [0.0, 0.997, 0.003], False),
    (79, [10711, 12276, 12279], [0.002, 0.998, 0.0], False),
    (80, [4544, 4603, 4604], [0.003, 0.997, 0.0], False),
    (81, [4545, 4591, 4592], [0.004, 0.996, 0.0], False),
    (82, [4524, 4525, 4555], [0.006, 0.584, 0.409], False),
    (83, [4685, 4598, 4599], [0.003, 0.001, 0.996], False),
    (84, [4691, 4593, 4594], [0.002, 0.995, 0.002], False),
    (85, [10702, 12270, 12284], [0.0, 1.0, 0.0], False),
    (86, [10722, 10721, 10819], [0.002, 0.995, 0.002], False),
    (87, [10727, 10726, 10813], [0.996, 0.001, 0.003], False),
    (88, [11285, 11286, 10694], [0.558, 0.312, 0.129], False),
    (89, [10805, 10806, 10801], [0.997, 0.002, 0.0], False),
    (90, [10709, 12285, 12272], [0.003, 0.997, 0.0], False),
    (91, [4673, 4678, 4677], [0.0, 0.002, 0.997], False),
    (92, [4566, 5158, 5157], [0.129, 0.312, 0.558], False),
    (93, [4689, 4690, 4687], [0.998, 0.001, 0.001], False),
    (94, [12271, 10704, 10703], [0.996, 0.004, 0.0], False),
    (95, [10815, 10818, 10817], [0.001, 0.001, 0.998], False),
    # --- eye (10) ---
    (64, [11028, 11027, 6758], [0.008, 0.974, 0.017], False),
    (68, [630, 4899, 4900], [0.017, 0.974, 0.008], False),
    (61, [7288, 7289, 7290], [0.996, 0.002, 0.003], False),
    (63, [7164, 7165, 7404], [0.004, 0.001, 0.995], False),
    (65, [6736, 6739, 7181], [0.005, 0.988, 0.008], False),
    (67, [6722, 7307, 7308], [0.002, 0.994, 0.004], False),
    (69, [1276, 1037, 1036], [0.995, 0.001, 0.004], False),
    (70, [1162, 1161, 1160], [0.003, 0.002, 0.996], False),
    (73, [1180, 1179, 594], [0.004, 0.994, 0.002], False),
    (75, [1053, 611, 608], [0.008, 0.988, 0.005], False),
    # --- eyebrow (10) ---
    (33, [7635, 7636, 7458], [0.443, 0.143, 0.414], False),
    (41, [7578, 7575, 7574], [0.141, 0.065, 0.794], False),
    (40, [7093, 7090, 7671], [0.226, 0.351, 0.423], False),
    (36, [7572, 7566, 7565], [0.053, 0.041, 0.907], False),
    (37, [7111, 7108, 7640], [0.004, 0.996, 0.0], False),
    (42, [1512, 980, 983], [0.0, 0.996, 0.004], False),
    (43, [1437, 1438, 1444], [0.907, 0.041, 0.053], True),
    (48, [1543, 962, 965], [0.423, 0.351, 0.226], True),
    (47, [1446, 1447, 1450], [0.794, 0.065, 0.141], True),
    (46, [1330, 1508, 1507], [0.414, 0.143, 0.443], False),
]

WFLW_INDICES = [p[0] for p in POINTS]
VERTEX_INDICES = [p[1] for p in POINTS]
WEIGHTS = [p[2] for p in POINTS]
EXCLUDE_AT_90 = [p[3] for p in POINTS]

# Phase 7 — real per-landmark weight for gnm_identity_fit.py's multi-view
# ridge solve (solve_identity_multiview). Regularization (lambda) is
# UNCHANGED; this only scales how much each landmark's own real target
# offset contributes to the shared 253-dim normal-equations solve, before
# the ridge term is added.
#
# nose (row 7, the ONLY point representing the whole nose among these 48 --
# see the "nose (1)" region above) is weighted 10x. Verified via a real
# controlled sweep (patient 0913e4c9, all 4 real views, same lambda=1e-5,
# same pose, same 253-dim basis -- see ai-engine Phase 6 diagnostic run):
# nose landmark residual 6.94mm at weight=1 -> 2.13mm at weight=10 (-69%),
# while jaw/mouth/eye/eyebrow residual stayed effectively unchanged
# (jaw 1.97->2.05mm, mouth 1.72->1.86mm, eye 2.61->2.61mm,
# eyebrow 3.63->3.55mm) and WFLW/silhouette-IoU metrics stayed flat
# (5.66->5.54px, 0.908->0.911) -- a real, measured, low-collateral fix for
# the "nose is outvoted 1-vs-47 in the shared ridge solve" root cause (see
# Phase 5's diagnostic: nose's own basis-capacity was never the problem --
# design matrix was already full-rank there, coefficient needed to close
# the gap was ~1.0, well within the production coefficient budget; the real
# problem was this single point's own weight in the JOINT solve).
#
# 2026-08-27 -- nose x10 -> x30, then x15, both tried and reverted: measured
# (patient 257d9bfe) to pull the single WFLW57 point toward its own
# already-computed pseudo-3D target along whatever direction that target
# happens to be in (mostly Y, sometimes Z DECREASING) rather than reliably
# adding Z-protrusion -- a single point's weight cannot manufacture bridge
# depth the target itself doesn't encode (see gnm_dense_nose.py's own "1
# point cannot constrain bridge height" root-cause note). WORSE: isolated
# testing (this session) showed nose x15 ALONE trips gnm_dense_nose.py's own
# maxCoeff>2.9 guard (measured 3.07), silently blocking the real depth-data
# enrichment (DENSE_NOSE_WEIGHT_TOTAL) from ever activating -- i.e. boosting
# this single-point weight was actively COUNTERPRODUCTIVE to the actual goal.
# Reverted to the original validated x10.
#
# Chin (WFLW 16) x1 -> x3 (previous change) then REVERTED back to x1: real
# measured effect on patient 257d9bfe was Z = -0.784mm -- i.e. it pulled the
# chin BACKWARD, the opposite of the intended "nhô cao" direction, for the
# exact same single-point-weight-can't-manufacture-depth reason as nose
# above. There is no dense-chin equivalent of gnm_dense_nose.py in this
# codebase (no real triangulated/BFM depth source for the chin region) --
# until one exists, x1 (the original, no-op-toward-depth value) is the only
# non-harmful setting available for this landmark. Pre-change (x10 nose/x1
# chin) output backed up under .backups/nose-chin-weight-boost-*/.
POINT_WEIGHTS = [1.0] * 7 + [10.0] + [1.0] * 20 + [1.0] * 10 + [1.0] * 10
assert len(POINT_WEIGHTS) == len(POINTS)


def point_mask(is_profile_view: bool) -> list[bool]:
    """True = keep this point. `is_profile_view` = photo came from the 90° upload slot (angle3)."""
    if not is_profile_view:
        return [True] * len(POINTS)
    return [not excl for excl in EXCLUDE_AT_90]
