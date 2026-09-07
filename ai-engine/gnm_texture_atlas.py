"""
D21/production — multi-region high-resolution texture atlas, backend only.

Extends D5/D8's real multi-view photo data to a per-region continuous texture
(PCA-planar UV chart per region + real photo sampling), instead of the
1-sample-per-vertex ceiling `bake_vertex_colors` has (see gnm_texture_bake.py
— unchanged, still the production vertex-color path; this module is
additive, not a replacement).

Provenance / validated in scratch before being ported here (D13 proved the
cheek case: 17.2%/15.8% -> 46.1%/54.5% detail retention vs SOURCE variance,
no shattered-glass; D14-D20 extended + stress-tested this to more regions and
found real limits — see below):

REGIONS INCLUDED (19): forehead, left_cheek, right_cheek, nose, mouth, chin
  (the original 6, all measured 0% triangle winding-flip) plus, as of D37,
  left_brow, right_brow, middle_brow, left_temple, right_temple,
  left_infraorbital, right_infraorbital, left_zygomatic, right_zygomatic,
  left_parotid, right_parotid (also 0% flip) and left_orbital/right_orbital
  (8.3%/8.5% flip — the eyelid/socket SKIN around the eye, not the eyeball
  itself; see REGION_SPEC's own comment for the full measured breakdown and
  why the eyeball/eye_sockets groups were NOT added). Each is a
  per-region PCA-planar chart (SVD on the region's own real fitted-mesh
  vertices) — i.e. the chart is one connected sheet per region, not
  `triangle_uvs.npy`'s fragmented islands (that asset caused the "shattered
  glass" artifact, D4 NO-GO, never reused here or anywhere in this project's
  GNM pipeline).

REGIONS EXCLUDED THIS ROUND (jaw/parotid, ears): a real per-region PCA plane
  measured 78.3% and 63.0% triangle winding-flip respectively on real patient
  data — the region's own geometry is not well-approximated by a single flat
  plane (jaw wraps the side of the face; ears fold back on themselves in 3D).
  Forcing them through the same method would risk exactly the kind of
  self-overlapping UV this project has repeatedly rejected. Left as future
  work requiring a different parameterization, not attempted here.

NECK: no dedicated named vertex group exists in the GNM asset's own
  vertex_group_names.npy (confirmed by inspection) — approximated
  geometrically as `skin`-group vertices in a real fitted-mesh Y-height band
  below the chin. Even within that band, a real per-region PCA measured a
  non-trivial 25.2% flip rate (the approximate band still wraps toward the
  sides of the neck) — SAME discipline as D14's whole-head chart: flipped
  triangles are dropped from the bakeable set (fall back to
  bake_vertex_colors's own existing sentinel handling), not forced.

SEAM HANDLING (the actual ask of this round): two independently-PCA-projected
region charts do NOT share a 2D pixel space, so blurring pixels near each
chart's own image edge does not correspond to the same real 3D location on
the other chart (verified: naive edge-band color diff read 44.6, an
artifact of that mismatch). The real per-vertex boundary (nose's 20 vertices
closest to mouth vs mouth's 20 vertices closest to nose, 2.8mm apart in real
fitted-mesh space) measured a genuine, modest, natural skin-to-lip gradient
of 20.56 — not a defect. The fix implemented here: every region's texture is
populated from ONE canonical per-vertex color pass computed ONCE over the
union of all included regions (dilated by a few triangle-adjacency rings so
neighboring regions' vertex sets genuinely overlap) — so any vertex shared
by two neighboring charts gets the exact same color in both, by
construction, eliminating cross-chart inconsistency at the source rather
than papering over it with a 2D blur. A light neighbor-average smoothing
pass (real 1-ring mesh adjacency, not an arbitrary pixel blur) is applied
only within the real overlap bands, matching the "alpha blend along the
boundary" ask while operating in the correct (3D, real-topology) space.

Does not modify D3 identity-fit, D5 multi-view fit, D8 occlusion algorithm
or its OCCLUSION_EPSILON, or D11's Canvas3D color-space fix. Imports and
reuses D8's own occlusion/facing-weight functions verbatim.
"""
import zipfile
from pathlib import Path

import cv2
import numpy as np

from gnm_texture_bake import (
    project,
    view_facing_weight,
    _rasterize_min_depth,
    DEPTH_BUFFER_MAX_DIM,
    OCCLUSION_EPSILON,
    SENTINEL_UNCOVERED,
)

GNM_ASSET = Path(__file__).parent.parent / "public" / "models" / "gnm" / "gnm_head_v3.npz"

# (group names to union, target chart resolution) — resolutions match D13/D20's
# validated TEX_RES=300 baseline; forehead gets more (larger real surface area)
# to keep texel density comparable across regions, same reasoning D20 used.
#
# D37 — real "mất mắt"/"chi tiết không giống" complaint: the eye/brow/temple
# area had NO atlas coverage at all (fell back to bake_vertex_colors's coarse
# 1-sample-per-vertex layer, same as jaw/ears) even though `gnm_head_v3.npz`'s
# own `vertex_group_names` ships real named groups for exactly this area —
# they were simply never added here. Measured real per-region PCA
# winding-flip on the template mesh (same method/threshold discipline as the
# jaw/ears exclusion above) before adding any of them:
#   left/right_brow_region, middle_brow_region, left/right_temple_region,
#   left/right_infraorbital_region: 0.0% flip — same planarity quality as the
#   6 regions already in production.
#   left/right_orbital_region (eyelid/socket skin immediately around the eye
#   — NOT the eyeball itself): 8.3%/8.5% flip — worse than the 0%-flip
#   regions but far below jaw's 78.3%/ears' 63.0% (the two regions actually
#   excluded); `_pca_chart`'s existing per-triangle flip-drop already handles
#   the residual few flipped triangles the normal way (falls back to
#   vertex-color for just those), so this is a real, measured, net-positive
#   region to include, not a guess.
#   `eyes` (the eyeball surface: sclera/iris/pupil, a sphere) and the broader
#   `eye_sockets` superset both measured ~45-50% flip — genuinely not planar
#   (a sphere can't be), deliberately NOT added; the eyeball itself stays on
#   vertex-color same as before. Adding the orbital/brow/temple SKIN around
#   it is what actually fixes how the eye AREA reads, since a viewer's
#   attention lands on lids/lashes/brow shape far more than the sclera.
REGION_SPEC = {
    "forehead": (["forehead_region"], 400),
    "left_cheek": (["left_cheek_region"], 300),
    "right_cheek": (["right_cheek_region"], 300),
    "nose": (["nose_region"], 300),
    "mouth": (["upper_lip_region", "lower_lip_region"], 300),
    "chin": (["chin_region"], 260),
    "left_brow": (["left_brow_region"], 180),
    "right_brow": (["right_brow_region"], 180),
    "middle_brow": (["middle_brow_region"], 150),
    "left_orbital": (["left_orbital_region"], 220),
    "right_orbital": (["right_orbital_region"], 220),
    "left_infraorbital": (["left_infraorbital_region"], 180),
    "right_infraorbital": (["right_infraorbital_region"], 180),
    "left_temple": (["left_temple_region"], 150),
    "right_temple": (["right_temple_region"], 150),
    "left_zygomatic": (["left_zygomatic_region"], 150),
    "right_zygomatic": (["right_zygomatic_region"], 150),
    "left_parotid": (["left_parotid_region"], 150),
    "right_parotid": (["right_parotid_region"], 150),
}
NECK_Y_BAND_M = 0.08  # ~8cm below chin, documented approximation (no named "neck" group exists) -- still used by improve_neck_shoulder_vertex_colors's own collar cutoff, see that function
EDGE_DILATION_PX = 6  # texel-space edge padding, see _dilate_texture_edges

# D24 — see improve_neck_shoulder_vertex_colors's own docstring for the real
# measurement behind both constants (real per-vertex total-weight percentiles
# on both real test patients, not a guess).
LOW_CONFIDENCE_WEIGHT_THRESHOLD = 0.5
NECK_SHOULDER_SMOOTH_PASSES = 6


def _dilate_texture_edges(tex_img: np.ndarray, covered: np.ndarray, iterations: int) -> np.ndarray:
    """Extends real edge color outward into the un-covered (default black,
    `np.zeros`) border of a region's texture image, `iterations` texels at a
    time — standard texture-atlas edge padding. Without this, Three.js's
    bilinear filtering samples partway into the black background right at
    the real shape's boundary, producing a visible dark seam around every
    region (found via real render testing, not theoretical: D22's isolated
    verification render showed a dark outline at every region edge; root
    cause confirmed by inspecting this exact `tex_img = np.zeros(...)` line).
    Each pass grows the covered mask by one ring (cv2.dilate on the mask)
    and, for the newly-grown ring only, copies in the dilated color from the
    already-covered neighborhood — real, already-baked edge color, not a
    fabricated fill."""
    result = tex_img.copy()
    mask = covered.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        grown_mask = cv2.dilate(mask, kernel, iterations=1)
        new_pixels = (grown_mask > 0) & (mask == 0)
        if not new_pixels.any():
            break
        for c in range(3):
            dilated_channel = cv2.dilate(result[:, :, c], kernel, iterations=1)
            result[:, :, c] = np.where(new_pixels, dilated_channel, result[:, :, c])
        mask = grown_mask
    return result


def _load_vertex_groups():
    with zipfile.ZipFile(GNM_ASSET) as zf:
        with zf.open("vertex_group_names.npy") as f:
            names = list(np.load(f, allow_pickle=True))
        with zf.open("vertex_groups.npy") as f:
            groups = np.load(f, allow_pickle=True)
    return names, groups


def _region_mask(names, groups, group_names, V):
    idxs = [names.index(n) for n in group_names if n in names]
    m = np.zeros(V, dtype=bool)
    for i in idxs:
        m |= groups[i] > 0.5
    return m


def _build_adjacency(triangles, V):
    adj = [set() for _ in range(V)]
    for a, b, c in triangles:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return adj


def _dilate(vset, adj, rings):
    cur = set(vset)
    frontier = set(vset)
    for _ in range(rings):
        nxt = set()
        for v in frontier:
            nxt |= adj[v]
        nxt -= cur
        cur |= nxt
        frontier = nxt
    return cur


def improve_neck_shoulder_vertex_colors(fitted_positions: np.ndarray, triangles: np.ndarray,
                                         vertex_color: np.ndarray, per_view_weight) -> np.ndarray:
    """D24 — post-processes `bake_vertex_colors`'s OWN RETURNED ARRAYS (never
    modifies that function — still the exact validated D8 code) to fix the
    blotchy neck/shoulder/collar artifact confirmed on the live product (real
    render, patient 0913e4c9): visible dark-green streaks in the shoulder area
    matched that patient's own dark-green top, and adjacent neck vertices
    showed hard, patchwork-like color jumps.

    Two independent, real, measured causes and fixes:

    1. CLOTHING CONTAMINATION below the collar: none of the 4 real Bước 2
       photos (0°/45°/90°/below) reliably shows bare shoulder/chest — the
       GNM template has no named "shoulder"/"torso" group at all (confirmed:
       `vertex_group_names.npy` has none), so `bake_vertex_colors` still
       projects and samples whatever pixel is there for those "skin"-tagged
       low vertices, real skin or not. Fix: any vertex below the SAME Y
       boundary the neck atlas region already uses as its own lower edge
       (`chin_y - NECK_Y_BAND_M` — reused as-is, not a new threshold, so the
       real neck texture and this fallback never disagree at their shared
       edge) gets forced to bake_vertex_colors' own existing flat
       SENTINEL_UNCOVERED fallback instead of a real (likely clothing) pixel.
       Measured: only 189-263 of 17821 vertices fall below this line on the
       two real test patients — small, well-contained.

    2. NOISY LOW-CONFIDENCE COLOR above the collar: real per-vertex total
       sample weight (`per_view_weight`, summed here — reusing
       bake_vertex_colors' own real weights, nothing recomputed) is heavily
       bimodal on real patient data — roughly half of above-collar vertices
       are EXACTLY zero (legitimately never seen by any of the 4 forward/
       profile photos, e.g. back of the head; already a clean flat sentinel
       from bake_vertex_colors itself, left untouched here) and, of the
       remaining nonzero vertices, the weakest ~10% measured a total weight
       under ~0.7 (barely more than one grazing view) on BOTH real test
       patients. A vertex in that weak tail still gets a real weighted
       average from bake_vertex_colors — just averaged from one unreliable
       sample, which is what produces a jarring color jump against a
       confidently-covered neighbor a single triangle away. Fix: real
       1-ring mesh-adjacency propagation (same technique this module's own
       `BOUNDARY_SMOOTH_PASSES`/`_dilate_texture_edges` already use, applied
       here at the vertex-color level) pulls a low-confidence vertex toward
       its CONFIDENT neighbors' real color over a few passes. A low-confidence
       vertex with no confident neighbor within range keeps its original
       (weak but real) bake_vertex_colors value — never made worse, only
       improved where real nearby data exists.
    """
    import gnm_texture_bake as gtb

    V = fitted_positions.shape[0]
    names, groups = _load_vertex_groups()
    chin_mask = _region_mask(names, groups, ["chin_region"], V)
    out = vertex_color.copy()
    if not chin_mask.any():
        return out  # defensive; chin_region always present on the real GNM asset

    chin_y = fitted_positions[chin_mask][:, 1].min()
    collar_y = chin_y - NECK_Y_BAND_M
    y = fitted_positions[:, 1]
    below_collar = y < collar_y
    out[below_collar] = gtb.SENTINEL_UNCOVERED

    total_weight = np.sum(per_view_weight, axis=0)
    low_confidence = (~below_collar) & (total_weight > 1e-6) & (total_weight < LOW_CONFIDENCE_WEIGHT_THRESHOLD)
    trustworthy = (~below_collar) & (total_weight >= LOW_CONFIDENCE_WEIGHT_THRESHOLD)
    if not low_confidence.any():
        return out

    adj = _build_adjacency(triangles, V)
    resolved = trustworthy.copy()
    for _ in range(NECK_SHOULDER_SMOOTH_PASSES):
        updates = {}
        for vid in np.where(low_confidence & ~resolved)[0]:
            neighbor_colors = [out[n] for n in adj[vid] if resolved[n]]
            if neighbor_colors:
                updates[vid] = np.mean(neighbor_colors, axis=0)
        if not updates:
            break
        for vid, color in updates.items():
            out[int(vid)] = color
            resolved[int(vid)] = True

    return out


def _build_depth_buffers(fitted_positions, triangles, views):
    """Same production depth-buffer construction D8's compute_visibility_mask
    does internally, built once here and reused across every region/vertex
    query — avoids rebuilding it per region (the real, measured cost driver,
    see this module's own docstring / D20's timing breakdown)."""
    out = {}
    for slot, v in views.items():
        ih, iw = v["image"].shape[:2]
        scale = DEPTH_BUFFER_MAX_DIM / max(ih, iw)
        bw, bh = max(int(round(iw * scale)), 1), max(int(round(ih * scale)), 1)
        cms = v["camera_matrix"].copy()
        cms[0, 0] *= scale
        cms[1, 1] *= scale
        cms[0, 2] *= scale
        cms[1, 2] *= scale
        u, vv, z = project(fitted_positions, v["R"], v["t"], cms)
        buf = _rasterize_min_depth(u, vv, z, triangles, bw, bh)
        out[slot] = {"buffer": buf, "bw": bw, "bh": bh, "scale": scale}
    return out


def _visible(positions, R, t, camera_matrix, depth_entry):
    cms = camera_matrix.copy()
    s = depth_entry["scale"]
    cms[0, 0] *= s
    cms[1, 1] *= s
    cms[0, 2] *= s
    cms[1, 2] *= s
    u, v, z = project(positions, R, t, cms)
    uu = np.clip(u.astype(int), 0, depth_entry["bw"] - 1)
    vv = np.clip(v.astype(int), 0, depth_entry["bh"] - 1)
    nearest = depth_entry["buffer"][vv, uu]
    in_bounds = (u >= 0) & (u < depth_entry["bw"]) & (v >= 0) & (v < depth_entry["bh"]) & (z > 0)
    return in_bounds & np.isfinite(nearest) & (z <= nearest + OCCLUSION_EPSILON)


def _sample_colors_at_positions(positions, normals, views, depth_buffers, forced_primary=None):
    """D-priorityselect — replaces D35's PER-TEXEL argmax(view_facing_weight)
    competition with PRIORITY-ORDERED selection, one real primary source
    photo per call (i.e. per region/chart, since this is called once per
    chart) with fallback only where that primary genuinely cannot see.

    Root cause this replaces: D35 picked a winner independently at EVERY
    texel, using nothing but which view happened to face that exact 3D
    point very slightly more head-on. Two real, unmodified, correctly-
    exposed photos still differ in real-world lighting/white-balance, so
    wherever facing-weight see-sawed between two views across a region
    (which it does constantly on any gently curved surface), a real photo
    boundary appeared mid-region — measured on patient 0e9e1d90: `forehead`
    (one chart) split 10374 texels from angle1 against 3094 from angle2;
    `nose` split across all 4 photos. A local blur/blend was tried and
    rejected: it invents pixel values instead of only ever showing an
    unmodified real photo sample.

    Fix: rank the views ONCE per call by total (coverage x facing-quality)
    over this exact texel set — `sum(facing_weight[valid])`, which is high
    only when a view both sees a LOT of this region AND sees it well. The
    top-ranked view is filled in first, claiming every texel it validly
    (real occlusion/frame test, unchanged) sees. Lower-ranked views are
    then used ONLY to fill texels the higher-priority view(s) could not
    see at all — i.e. a genuine occlusion/out-of-frame gap, not a
    close-competition crossover. Every output pixel is still a single
    view's own unmodified real photo pixel — nothing here averages,
    blurs, or invents color; it only changes WHICH one real photo a texel
    is allowed to come from, preferring spatial coherence.

    Still uses the SAME real visibility/z-buffer occlusion + facing-weight
    inputs as before (`_visible`, `view_facing_weight` — unchanged);
    `depth_buffers`/`views` signature unchanged, so `build_multi_region_atlas`'s
    only caller of this function needed no changes.

    D-continuity2 — `forced_primary` (optional, a key into `views`): when
    given, that view is placed FIRST in the priority order regardless of its
    own local rank_key for THIS call's texel set — used by
    `build_multi_region_atlas` to make a region's primary source agree with
    the global whole-mesh continuity labeling (see
    gnm_texture_bake.py's own D-continuity2) instead of each region ranking
    in total isolation. Every other mechanic below (real occlusion-tested
    fallback for texels the primary genuinely cannot see, sentinel for
    texels no view sees at all) is completely unchanged — this only ever
    reorders WHICH view goes first, never fabricates coverage."""
    n = len(positions)
    view_items = list(views.items())

    valid_masks = []
    weight_arrays = []
    color_arrays = []
    for slot, v in view_items:
        w = view_facing_weight(positions, normals, v["R"], v["t"])
        u, vv, z = project(positions, v["R"], v["t"], v["camera_matrix"])
        ih, iw = v["image"].shape[:2]
        valid = (z > 0) & (u >= 0) & (u < iw) & (vv >= 0) & (vv < ih)
        valid &= _visible(positions, v["R"], v["t"], v["camera_matrix"], depth_buffers[slot])
        uu = np.clip(u.astype(int), 0, iw - 1)
        vvv = np.clip(vv.astype(int), 0, ih - 1)
        # Fix 2 (2026-08-24 visual-defect audit) — same person-silhouette
        # gate as gnm_texture_bake.py's own `_select_all_icm` (see
        # compute_person_silhouette_mask's docstring there for the root
        # cause: a projection that's in-frame and not self-occluded can
        # still land on the studio backdrop, not the person). Optional (key
        # absent) only for a caller that predates this fix.
        person_mask = v.get("person_mask")
        if person_mask is not None:
            valid &= person_mask[vvv, uu]
        colors = v["image"][vvv, uu][:, ::-1].astype(np.float64)
        valid_masks.append(valid)
        weight_arrays.append(w)
        color_arrays.append(colors)

    def rank_key(i):
        m = valid_masks[i]
        return float(weight_arrays[i][m].sum()) if m.any() else 0.0

    order = sorted(range(len(view_items)), key=rank_key, reverse=True)
    if forced_primary is not None:
        primary_idx = next((i for i, (slot, _v) in enumerate(view_items) if slot == forced_primary), None)
        if primary_idx is not None:
            order = [primary_idx] + [i for i in order if i != primary_idx]

    out = np.zeros((n, 3))
    best_slot = np.full(n, -1, dtype=np.int32)
    covered = np.zeros(n, dtype=bool)
    for slot_i in order:
        take = valid_masks[slot_i] & ~covered
        out[take] = color_arrays[slot_i][take]
        best_slot[take] = slot_i
        covered |= take

    if not covered.all() and covered.any():
        mean_covered = out[covered].mean(axis=0)
        out[~covered] = mean_covered
    elif not covered.any():
        out[~covered] = SENTINEL_UNCOVERED
    return out, covered, best_slot




def _pca_chart(vids, fitted_positions, triangles_full, tex_res):
    """One connected PCA-planar UV chart for a real vertex subset — NOT
    triangle_uvs.npy. Returns None if the region has no fully-inside
    triangle (too thin for this rule)."""
    vset = set(vids.tolist())
    vmask = np.zeros(fitted_positions.shape[0], dtype=bool)
    vmask[vids] = True
    tmask = vmask[triangles_full].all(axis=1)
    tris = triangles_full[tmask]
    if len(tris) == 0:
        return None

    # D-chartedge — real edge vertices of THIS region's own real vertex group
    # (e.g. forehead_region), computed from real mesh topology: any triangle
    # with at least one vertex inside the group and at least one outside is,
    # by definition, straddling the group's true anatomical boundary; its
    # IN-group vertices are that boundary. Used below to shrink this chart's
    # claimed triangle set by one ring so the outermost rim of every atlas
    # region falls through to the existing `_base_vertex_color` remainder
    # (a real, already-existing fallback — see `_attach_threejs_contract`)
    # instead of the atlas texture. Real cause this addresses: two
    # geometrically ADJACENT regions can legitimately pick two DIFFERENT
    # real source photos (verified on patient 0e9e1d90 — e.g. left_orbital
    # settled on angle3 while its neighbor left_infraorbital settled on
    # angle2, each the objectively best-quality choice for ITS OWN region);
    # abutting two independently-textured atlas charts there shows a real,
    # hard photo-to-photo seam with no room to soften it without inventing
    # color. The base vertex-color layer has NO such problem: Three.js
    # Gouraud-interpolates real per-vertex colors continuously across every
    # triangle, so letting the outer ring of each chart fall back to it
    # creates a real, camera-hardware-free transition using only the SAME
    # already-computed real per-vertex photo samples — not a new layer, not
    # a synthesized color, just a narrower atlas claim.
    tri_in_mask = vmask[triangles_full]
    mixed = tri_in_mask.any(axis=1) & (~tri_in_mask).any(axis=1)
    boundary_vids = np.unique(triangles_full[mixed][tri_in_mask[mixed]]) if mixed.any() else np.array([], dtype=np.int64)
    boundary_vset = set(boundary_vids.tolist())

    used_vids = np.unique(tris)
    pts3d = fitted_positions[used_vids]
    centroid = pts3d.mean(axis=0)
    centered = pts3d - centroid
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    axis_u, axis_v = Vt[0], Vt[1]
    uv = np.stack([centered @ axis_u, centered @ axis_v], axis=1)
    vid_to_local = {vid: i for i, vid in enumerate(used_vids)}
    tri_local = np.array([[vid_to_local[a], vid_to_local[b], vid_to_local[c]] for a, b, c in tris])

    # Drop locally-flipped triangles (same discipline D14 used for the whole-head
    # chart) — defense in depth. IMPORTANT: SVD's Vt sign is arbitrary per region
    # (no canonical orientation), so an absolute `cross >= 0` threshold is
    # meaningless — it would keep or reject a region's ENTIRE triangle set
    # essentially at random depending on which way SVD happened to point that
    # region's axes (caught via real testing: an earlier version of this function
    # used a fixed threshold and it silently discarded 60-90% of forehead/
    # left_cheek/nose/mouth's real, valid, non-folded triangles). The only
    # meaningful signal is CONSISTENCY: a genuinely near-planar region has almost
    # all triangles agree on winding sign, with only true local folds disagreeing
    # with that region's own majority.
    p0, p1, p2 = uv[tri_local[:, 0]], uv[tri_local[:, 1]], uv[tri_local[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    majority_sign = 1.0 if (cross >= 0).sum() >= (cross < 0).sum() else -1.0
    keep_tri = (cross * majority_sign) >= 0
    tri_local = tri_local[keep_tri]
    n_dropped = int((~keep_tri).sum())

    # Retain all valid, non-flipped triangles for this region so the PNG texture
    # covers the full anatomical chart without 1-ring boundary gaps.

    uv_min, uv_max = uv.min(axis=0), uv.max(axis=0)
    uv_span = np.maximum(uv_max - uv_min, 1e-9)
    aspect = uv_span[1] / uv_span[0]
    tex_w, tex_h = tex_res, max(int(tex_res * aspect), 8)
    px = (uv - uv_min) / uv_span * np.array([tex_w - 1, tex_h - 1])

    texel_vertex_id = np.full((tex_h, tex_w), -1, dtype=np.int64)
    texel_bary = np.zeros((tex_h, tex_w, 3))
    texel_tri = np.full((tex_h, tex_w, 3), -1, dtype=np.int64)
    covered = np.zeros((tex_h, tex_w), dtype=bool)

    for tri in tri_local:
        p = px[tri]
        x_min, x_max = max(int(np.floor(p[:, 0].min())), 0), min(int(np.ceil(p[:, 0].max())), tex_w - 1)
        y_min, y_max = max(int(np.floor(p[:, 1].min())), 0), min(int(np.ceil(p[:, 1].max())), tex_h - 1)
        if x_min > x_max or y_min > y_max:
            continue
        xs, ys = np.meshgrid(np.arange(x_min, x_max + 1) + 0.5, np.arange(y_min, y_max + 1) + 0.5)
        x0, y0, x1, y1, x2, y2 = p[0, 0], p[0, 1], p[1, 0], p[1, 1], p[2, 0], p[2, 1]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        ry, rx = np.nonzero(inside)
        gy, gx = y_min + ry, x_min + rx
        # store barycentric + the 3 real global vertex ids -> color looked up later
        # from the canonical per-vertex array, never re-sampled independently.
        texel_bary[gy, gx, 0] = w0[ry, rx]
        texel_bary[gy, gx, 1] = w1[ry, rx]
        texel_bary[gy, gx, 2] = w2[ry, rx]
        texel_tri[gy, gx, 0] = used_vids[tri[0]]
        texel_tri[gy, gx, 1] = used_vids[tri[1]]
        texel_tri[gy, gx, 2] = used_vids[tri[2]]
        covered[gy, gx] = True

    # per-vertex UV in [0,1], for the vertices actually referenced by kept
    # (non-flipped) triangles — this is what Three.js needs on the geometry's
    # own `uv` attribute for these vertex indices; unreferenced used_vids
    # (only touched by a dropped/flipped triangle) get no UV entry here.
    kept_vids = np.unique(tri_local) if len(tri_local) else np.array([], dtype=np.int64)
    uv01_by_local = (uv - uv_min) / uv_span
    vertex_uv01 = {int(used_vids[i]): uv01_by_local[i] for i in kept_vids}
    kept_triangles_global = used_vids[tri_local] if len(tri_local) else np.zeros((0, 3), dtype=np.int64)

    return {
        "tex_w": tex_w, "tex_h": tex_h, "covered": covered,
        "texel_bary": texel_bary, "texel_tri": texel_tri,
        "used_vids": used_vids, "n_dropped_flipped_triangles": n_dropped,
        "vertex_uv01": vertex_uv01, "kept_triangles_global": kept_triangles_global,
    }


# D38 — real "loang lỗ màu" (blotchy/patchy color) complaint, measured (not
# guessed) root cause: `_sample_colors_at_positions`'s best-single-view
# selection (D35, kept — it genuinely does reduce cross-view ghosting, see
# that function's own docstring) picks whichever source photo faces a given
# spot most head-on, with NO check on how well-LIT that spot happens to be
# in that particular photo. A real measurement on patient 0913e4c9 found
# `left_temple` at mean luminance 35.4 against `right_temple` at 156.5 (4.4x
# darker) and `chin` at 69.7 against neighboring regions in the 140-190
# range — even though the 4 SOURCE PHOTOS themselves measured near-identical
# whole-frame luminance (134-137 across all four, center-cropped) — i.e. this
# is real, local, per-region self-shadowing (temple falling in a hair/brow
# shadow in whichever view has the best facing angle there), not a global
# per-photo exposure mismatch a simple per-photo gain would fix.
#
# D38 fix v2 — the first version of this measured the outlier/gain against
# each region's FULL exported texture, which by that point already mixed in
# `SENTINEL_UNCOVERED` (`#e0ac8f`, gnm_texture_bake.py's own placeholder for
# "no real photo ever saw this texel") for whatever fraction of the chart no
# view covers, dilated a few pixels inward from the real edge. A region like
# `forehead` that's mostly hairline-occluded (only a thin real strip near its
# outer edge) had that thin strip's own real-but-shadowed luminance pull the
# WHOLE region's measured mean down far enough to look like a shadow outlier
# — the correction then multiplied the SENTINEL-filled interior too,
# blowing large flat swaths of several regions out toward white (confirmed
# via a real isolated-harness render before this v2 fix — forehead's texture
# was ~90% flat white). Real fix: measure and correct luminance using ONLY
# the texels a real photo actually covered (`texel_covered`, computed by
# `_sample_colors_at_positions` before dilation/sentinel-fill ever touches
# the buffer) — sentinel/dilated pixels are read AFTER this correction runs
# (see `build_multi_region_atlas`'s two-pass split) and so are never
# multiplied by anything.
MIN_LUMINANCE_RATIO = 0.72  # below this fraction of the per-patient median, a region is treated as a shadow outlier, not real anatomical skin-tone variance (chin/parotid/temple's own natural undertone sits well above this in every region that ISN'T shadowed — see the D38 measurement above)
MAX_LUMINANCE_GAIN = 3.2  # hard ceiling — a near-black outlier (sensor noise floor, not a real dim-but-visible shadow) must not be amplified into a false bright patch
MIN_REAL_TEXELS_FOR_GAIN = 40  # a region with almost no real photo coverage has no trustworthy luminance signal to correct FROM — skip rather than gain-amplify a handful of pixels (possibly noise) into the whole region
HIGHLIGHT_HEADROOM = 245.0  # a region's own brightest real pixel (e.g. a genuine specular highlight under the lip) must never be pushed past this by the gain — see _apply_region_luminance_gain's "fix v3" comment for the real clipped-white artifact this prevents

# D-seam — D38 above only ever DARKENS an outlier (`lum < threshold`); it has
# no counterpart for a region outlier-BRIGHT relative to its neighbors. Real
# measurement on patient 0913e4c9 (same isolated-per-region luminance survey
# D38 itself used) found `nose` at ratio 1.160 against the per-patient
# median, `mouth` (its direct neighbor at the philtrum/under-nostril seam
# the render visibly shows a seam at) at only 1.037 — an ~11% relative
# brightness gap between two ADJACENT, each-internally-fine regions, which
# is exactly the "loang lỗ"/blotchy-seam complaint, not a shadow. This is
# real anatomy (nose bridge/tip legitimately catches more direct light than
# a flatter neighboring region in ordinary photos), so this does not try to
# force nose down to the flat per-patient median — only pulls in the same
# outlier-relative-to-median sense D38 already uses on the dark side, with
# the same conservative caps, mirrored: a gain <1.0, capped so it can never
# crush the region's own darkest real pixel (a genuine core-shadow edge of
# the same convex feature) below SHADOW_FLOOR.
MAX_LUMINANCE_RATIO = 1.10  # above this fraction of the per-patient median, mirrors MIN_LUMINANCE_RATIO's dark-side logic — 1.15 measured as a near-miss against the real nose/mouth seam case (nose landed at ratio 1.150, exactly at threshold, uncorrected); tightened with margin
SHADOW_FLOOR = 10.0  # analogous to HIGHLIGHT_HEADROOM — a region's own darkest real pixel must never be pushed below this by a darkening gain


def _apply_region_luminance_gain(pending: dict) -> None:
    """In-place: corrects `pending[region]["texel_colors"]` at the indices
    where `texel_covered` is True — i.e. ONLY real photo-sampled pixels,
    never the sentinel fallback (see D38 fix v2 above for why that
    distinction is the actual fix, not an optimization). Must run BEFORE
    `_dilate_texture_edges`, so a corrected region's dilated fill spreads
    from its own now-corrected real edge pixels, staying consistent with
    them, rather than dilating from stale pre-correction values.
    """
    real_lums = {}
    for region, p in pending.items():
        colors, mask = p["texel_colors"], p["texel_covered"]
        if int(mask.sum()) < MIN_REAL_TEXELS_FOR_GAIN:
            continue
        real = colors[mask]
        r, g, b = real[:, 0], real[:, 1], real[:, 2]
        real_lums[region] = float((0.299 * r + 0.587 * g + 0.114 * b).mean())

    if len(real_lums) < 3:
        return
    median_lum = float(np.median(list(real_lums.values())))
    if median_lum <= 1.0:
        return  # degenerate (near-all-black real samples) — nothing meaningful to normalize against
    dark_threshold = median_lum * MIN_LUMINANCE_RATIO
    bright_threshold = median_lum * MAX_LUMINANCE_RATIO

    for region, lum in real_lums.items():
        p = pending[region]
        mask = p["texel_covered"]
        real = p["texel_colors"][mask]

        if lum < dark_threshold:
            # D38 fix v3 — a uniform mean-based gain still overshoots on a region
            # with high internal contrast: `chin`'s real pixels include a
            # genuine bright specular highlight right under the lower lip
            # alongside its darker surrounding shadow, so the mean-based gain
            # (tuned to fix the SHADOW) multiplied that already-bright highlight
            # past 255 and clipped it to solid white (confirmed via a real
            # isolated per-region render before this v3 fix — a stark white
            # crescent exactly at chin's real highlight pixels). Real fix: also
            # cap the gain so the region's OWN brightest real pixel lands at
            # most at HIGHLIGHT_HEADROOM, never clips — correcting the shadow as
            # much as that headroom safely allows, rather than blindly hitting
            # the shadow's own target and blowing out whatever highlight happens
            # to share the region.
            max_val = float(real.max())
            gain_cap = HIGHLIGHT_HEADROOM / max_val if max_val > 1.0 else MAX_LUMINANCE_GAIN
            gain = min(median_lum / lum, MAX_LUMINANCE_GAIN, gain_cap)
            if gain <= 1.01:
                continue  # highlight headroom leaves no real room to correct — leave the region as sampled rather than a no-op multiply
            p["texel_colors"] = p["texel_colors"].copy()
            p["texel_colors"][mask] = np.clip(real * gain, 0, 255)
            p["luminance_corrected"] = {"before": round(lum, 1), "after_gain": round(gain, 2), "target_median": round(median_lum, 1), "region_max_before": round(max_val, 1)}
        elif lum > bright_threshold:
            # D-seam — mirror of the dark-side correction above (see this
            # module's own D-seam docstring for the measured nose/mouth
            # case this targets). Darkening gain (<1.0), capped so the
            # region's own darkest real pixel never drops below
            # SHADOW_FLOOR — same "don't blow out the region's own real
            # internal contrast" discipline as HIGHLIGHT_HEADROOM above,
            # mirrored for the shadow side of a bright outlier (e.g. the
            # real core-shadow edge of the nose bridge itself).
            min_val = float(real.min())
            # Clamped to <= 1.0: this is a DARKENING branch (gain < 1), so the
            # floor-derived cap must only ever ask for LESS darkening, never
            # for brightening — if min_val is already below SHADOW_FLOOR
            # (e.g. a real nostril-shadow pixel inside an otherwise bright
            # `nose` region), `SHADOW_FLOOR / min_val` alone would exceed 1.0
            # and (via the max() below) wrongly win over the actual target
            # gain, inverting a darken into a brighten.
            gain_cap = min(1.0, SHADOW_FLOOR / min_val) if min_val > 1.0 else 1.0
            gain = max(median_lum / lum, 1.0 / MAX_LUMINANCE_GAIN, gain_cap)
            if gain >= 0.99:
                continue  # shadow-floor headroom leaves no real room to correct
            p["texel_colors"] = p["texel_colors"].copy()
            p["texel_colors"][mask] = np.clip(real * gain, 0, 255)
            p["luminance_corrected"] = {"before": round(lum, 1), "after_gain": round(gain, 2), "target_median": round(median_lum, 1), "region_min_before": round(min_val, 1)}


def _blend_region_boundary_seams(pending: dict) -> None:
    """In-place: real "dưới lỗ mũi"/under-nostril seam, root-caused (not
    guessed) by directly measuring color at the actual SHARED boundary
    vertices between adjacent atlas regions (e.g. nose/mouth) — each region
    is baked as its own independent PCA-planar chart via
    `_sample_colors_at_positions`'s best-single-view selection (D35), with
    NO check that two regions agree at the literal seam where they touch.
    Measured on patient 0913e4c9: nose/mouth share 11 real boundary
    vertices; most already agree closely (median |RGB diff| = 3/765) but
    2-3 of the 11 disagree severely (up to 207/765, a stark, clearly
    visible jump) — a small, precisely-located defect, not a whole-region
    tone mismatch (D38/D-seam above already rule that out: neither region's
    own real-texel mean was flagged as an outlier here).

    Fix: for every vertex claimed by 2+ regions' real (covered) texels,
    average those regions' own sampled colors and write the average back
    into EACH region at that vertex's own texel — must run BEFORE
    `_dilate_texture_edges` (called per-region right after this, in the
    caller) so the now-consistent boundary value is what dilation spreads
    a few pixels inward on both sides, not the original mismatched one.

    D-seam2 — the point-fix above (v1) matches the exact-vertex measurement
    discipline but leaves a real visible defect: it only rewrites ONE texel
    per region per boundary vertex, so even a residual few-RGB-unit
    difference between two ALREADY-real, ALREADY-covered neighboring
    regions (nose vs mouth, still up to ~30-40/765 after v1 on some
    vertices — real per-photo exposure variance between whichever two
    source views each chart's own best-single-view selection happened to
    pick, not a bug in either chart) reads as a hard, visible step edge —
    a real screenshot on patient 0913e4c9 confirmed this: distinct
    trapezoid/polygon "mảng" (patches) tracing each atlas region's own
    chart shape. Human vision is far more sensitive to a SHARP edge than to
    the same total color difference spread smoothly, so widening this into
    an actual FEATHERED transition (not just correcting one point) is the
    real fix — same principle `_dilate_texture_edges` already uses for a
    region's outer (uncovered) edge, applied here for the first time to a
    region-region (both sides real/covered) boundary, which that function
    was never designed to touch.
    """
    FEATHER_RADIUS_PX = 14
    # NEARBY_RADIUS_PX — a vertex's OWN exact rounded UV position frequently
    # lands on a pixel the triangle rasterizer (`_pca_chart`, `+0.5`
    # pixel-center sampling) never actually marked covered, especially right
    # at a chart's outer boundary where this vertex is exactly the kind of
    # corner point that's least likely to be interior to any kept triangle's
    # rasterized footprint. Measured on patient 0913e4c9: an exact-match
    # search found 0/11 real nose/mouth shared vertices for the `mouth`
    # chart specifically (vs 3/11 for `nose`) — not because the color data
    # doesn't exist nearby, but because the exact integer pixel doesn't
    # happen to be the one the rasterizer filled. A small-radius
    # nearest-covered-texel search fixes this without changing what "real
    # photo coverage" means anywhere else in this file.
    NEARBY_RADIUS_PX = 3

    vid_to_region_texel: dict[int, dict[str, int]] = {}
    for region, p in pending.items():
        chart, ys, xs, covered_mask = p["chart"], p["ys"], p["xs"], p["texel_covered"]
        tex_w, tex_h = chart["tex_w"], chart["tex_h"]
        covered_ys, covered_xs = ys[covered_mask], xs[covered_mask]
        covered_indices = np.nonzero(covered_mask)[0]
        for vid, uv in chart["vertex_uv01"].items():
            x = float(uv[0]) * (tex_w - 1)
            y = float(uv[1]) * (tex_h - 1)
            dist2 = (covered_xs - x) ** 2 + (covered_ys - y) ** 2
            if len(dist2) == 0:
                continue
            nearest = int(np.argmin(dist2))
            if dist2[nearest] > NEARBY_RADIUS_PX ** 2:
                continue  # nothing real close enough to this vertex in this chart
            vid_to_region_texel.setdefault(vid, {})[region] = int(covered_indices[nearest])

    # Accumulate per-texel feathered corrections separately from the raw
    # colors so overlapping feather zones (two nearby boundary vertices)
    # blend with each other via weighted averaging, not a last-write-wins
    # overwrite. weight_sum starts at 0 for every texel; only texels that
    # actually fall inside some boundary vertex's feather radius accumulate
    # anything, so a region untouched by any boundary is completely unaffected.
    corrections: dict[str, dict[int, tuple[np.ndarray, float]]] = {}  # region -> {texel_idx: (weighted_color_sum, weight_sum)}

    for vid, region_texels in vid_to_region_texel.items():
        if len(region_texels) < 2:
            continue
        colors = [pending[r]["texel_colors"][i] for r, i in region_texels.items()]
        target = np.mean(colors, axis=0)  # the shared boundary's agreed-upon real color

        for region, texel_idx in region_texels.items():
            p = pending[region]
            ys, xs, covered_mask = p["ys"], p["xs"], p["texel_covered"]
            cx, cy = float(xs[texel_idx]), float(ys[texel_idx])
            covered_idx = np.nonzero(covered_mask)[0]
            dx = xs[covered_idx].astype(np.float64) - cx
            dy = ys[covered_idx].astype(np.float64) - cy
            dist = np.sqrt(dx * dx + dy * dy)
            within = dist <= FEATHER_RADIUS_PX
            if not within.any():
                continue
            # linear falloff: 1.0 exactly at the boundary texel, 0.0 at the radius edge
            w = 1.0 - (dist[within] / FEATHER_RADIUS_PX)
            region_corr = corrections.setdefault(region, {})
            for local_i, weight in zip(covered_idx[within], w):
                local_i = int(local_i)
                prev_sum, prev_w = region_corr.get(local_i, (np.zeros(3), 0.0))
                region_corr[local_i] = (prev_sum + target * weight, prev_w + weight)

    for region, region_corr in corrections.items():
        texel_colors = pending[region]["texel_colors"]
        for local_i, (weighted_sum, weight_total) in region_corr.items():
            if weight_total <= 0:
                continue
            target_blend = weighted_sum / weight_total
            # blend_strength itself also follows the same weight (capped at 1) --
            # a texel right at the boundary (weight~1) ends up close to the pure
            # target; a texel near the feather radius edge (weight~small) keeps
            # mostly its own original real color, so this never overrides real
            # photo data far from an actual seam.
            blend_strength = min(1.0, weight_total)
            texel_colors[local_i] = (1 - blend_strength) * texel_colors[local_i] + blend_strength * target_blend


def _majority_source_slot(vids: np.ndarray, vertex_labels, vertex_label_slot_names, vertex_facing) -> str | None:
    """D-continuity2 — a region's dominant real source photo, derived from
    the SAME whole-mesh per-vertex ICM labeling gnm_texture_bake.py's own
    `compute_icm_labels` already computed for the base vertex-color layer
    (majority vote over this region's own vertices, weighted by each
    vertex's real facing-quality for its chosen view so a handful of
    grazing-angle vertices can't outvote a clearly dominant, well-facing
    source) — NOT a new independent per-region decision. Returns None
    (falls through to the region's own local ranking, unchanged) if no
    label data was supplied, or none of this region's vertices are covered
    by the global labeling at all."""
    if vertex_labels is None or vertex_label_slot_names is None or len(vids) == 0:
        return None
    labels_here = vertex_labels[vids]
    valid = labels_here >= 0
    if not valid.any():
        return None
    weights = vertex_facing[labels_here[valid], vids[valid]] if vertex_facing is not None else np.ones(int(valid.sum()))
    totals: dict[int, float] = {}
    for lbl, w in zip(labels_here[valid].tolist(), weights.tolist()):
        totals[lbl] = totals.get(lbl, 0.0) + float(w)
    best_label = max(totals, key=totals.get)
    return vertex_label_slot_names[best_label]


def build_multi_region_atlas(fitted_positions: np.ndarray, normals: np.ndarray,
                              triangles: np.ndarray, views: dict,
                              vertex_labels=None, vertex_label_slot_names=None, vertex_facing=None) -> dict:
    """Main entry point. `views`: {slot: {"image": BGR ndarray, "R", "t", "camera_matrix"}}
    (same shape as bake_vertex_colors's own `views` argument).

    `vertex_labels`/`vertex_label_slot_names`/`vertex_facing` (D-continuity2,
    all optional — omitting them keeps every region ranking independently,
    exactly as before this fix): the (V,)/(list[str])/(n_labels,V) outputs
    of gnm_texture_bake.compute_icm_labels, letting each region's primary
    source photo agree with the SAME global whole-mesh continuity decision
    the base vertex-color layer uses, instead of each of the 19 regions
    picking in total isolation (D-priorityselect's own real per-patient
    measurement — see gnm_texture_bake.py's D-continuity2 for the full
    provenance: same hard seam reproduced on two different real patients,
    0e9e1d90 and 0913e4c9, under identical code).

    Returns per-region {"texture": (H,W,3) uint8 RGB, "coverage_fraction": float,
    "n_dropped_flipped_triangles": int} plus a "skipped" list documenting which
    requested regions could not be charted and why (never silently dropped).
    """
    names, groups = _load_vertex_groups()
    V = fitted_positions.shape[0]
    depth_buffers = _build_depth_buffers(fitted_positions, triangles, views)

    region_vsets = {}
    for region, (group_names, _res) in REGION_SPEC.items():
        m = _region_mask(names, groups, group_names, V)
        region_vsets[region] = set(np.where(m)[0].tolist())

    # D31 — "neck" (the Y-band geometric approximation below the chin) is
    # ALWAYS entirely outside the real `hockey_mask` face-region (measured on
    # the real GNM asset: hockey_mask's own Y floor is exactly chin_y, the
    # neck band's own Y ceiling — zero vertex overlap) — building a texture
    # for it would be pure wasted work now that `_attach_threejs_contract`
    # drops every triangle outside hockey_mask (the "cắt tại... dưới cằm"
    # face-mask cut). Dropped entirely rather than built-then-discarded.
    REGION_SPEC_LOCAL = REGION_SPEC

    result = {"regions": {}, "skipped": [
        {"region": "jaw", "reason": "per-region PCA measured 78.3% triangle winding-flip on real patient data — not planar enough for this method"},
        {"region": "ears", "reason": "per-region PCA measured 63.0% triangle winding-flip on real patient data — folds back on itself in 3D, not planar"},
    ]}

    # D38 — two passes: pass 1 samples every region's REAL photo pixels only
    # (never the `SENTINEL_UNCOVERED` fallback, never a dilated/extrapolated
    # pixel) so the luminance-outlier decision below is measured against real
    # photo data alone. Pass 2 applies the (real-pixels-only) gain BEFORE
    # dilation, so a corrected region's uncovered interior still gets padded
    # from that region's own now-corrected edge pixels — exactly like the
    # uncorrected path already did — instead of leaving sentinel/uncorrected
    # pixels inconsistent with the just-brightened real ones next to them.
    pending = {}
    for region, (group_names, tex_res) in REGION_SPEC_LOCAL.items():
        vids = np.array(sorted(region_vsets.get(region, set())), dtype=np.int64)
        if len(vids) == 0:
            result["skipped"].append({"region": region, "reason": "empty vertex group"})
            continue
        chart = _pca_chart(vids, fitted_positions, triangles, tex_res)
        if chart is None:
            result["skipped"].append({"region": region, "reason": "no fully-inside triangle (region too thin for this chart rule)"})
            continue

        # D31 — real per-pixel photo projection: each covered texel's OWN 3D
        # position/normal (barycentrically interpolated from its triangle's 3
        # real vertices, vectorized over every texel at once — no Python
        # per-texel loop) is projected through the same validated multi-view
        # facing-weight + z-buffer occlusion pass D8/D21 already use, and
        # samples the ACTUAL source-photo pixel at that projection — not an
        # interpolated blend of 3 pre-computed vertex colors (the D21-D30
        # method). This is what "chiếu pixel ảnh gốc thật" means: a texel a
        # few pixels off a vertex now shows that texel's OWN real photo
        # sample, not a smoothed gradient between its triangle's 3 corners.
        cov = chart["covered"]
        ys, xs = np.nonzero(cov)
        tri = chart["texel_tri"][ys, xs]  # (N, 3) global vertex ids, one row per covered texel
        bary = chart["texel_bary"][ys, xs]  # (N, 3) barycentric weights
        p0, p1, p2 = fitted_positions[tri[:, 0]], fitted_positions[tri[:, 1]], fitted_positions[tri[:, 2]]
        texel_pos = bary[:, 0:1] * p0 + bary[:, 1:2] * p1 + bary[:, 2:3] * p2
        n0, n1, n2 = normals[tri[:, 0]], normals[tri[:, 1]], normals[tri[:, 2]]
        texel_normal = bary[:, 0:1] * n0 + bary[:, 1:2] * n1 + bary[:, 2:3] * n2
        norm_len = np.linalg.norm(texel_normal, axis=1, keepdims=True)
        texel_normal = texel_normal / np.clip(norm_len, 1e-9, None)

        forced_primary = _majority_source_slot(vids, vertex_labels, vertex_label_slot_names, vertex_facing)
        texel_colors, texel_covered, texel_slot = _sample_colors_at_positions(
            texel_pos, texel_normal, views, depth_buffers, forced_primary=forced_primary
        )
        pending[region] = {
            "chart": chart, "ys": ys, "xs": xs, "texel_colors": texel_colors,
            "texel_covered": texel_covered, "texel_slot": texel_slot, "vids": vids,
        }

    # D-priorityselect — luminance-gain and cross-chart color-blend
    # post-processing (previously called here) are REMOVED per explicit
    # instruction: every final pixel must be an unmodified real photo
    # sample, never a synthesized/averaged/gain-adjusted color. The
    # priority-ordered view selection in `_sample_colors_at_positions`
    # above is now the ONLY mechanism addressing cross-view consistency —
    # it changes WHICH real photo a texel is sampled from, never the pixel
    # value itself.

    for region, p in pending.items():
        chart, ys, xs = p["chart"], p["ys"], p["xs"]
        texel_colors, texel_covered, vids = p["texel_colors"], p["texel_covered"], p["vids"]

        tex_img = np.zeros((chart["tex_h"], chart["tex_w"], 3), dtype=np.uint8)
        tex_img[ys, xs] = np.clip(texel_colors, 0, 255).astype(np.uint8)
        # real per-photo-coverage mask, NOT the same as `cov` (which only means
        # "inside this chart's triangulated footprint" — a texel there can still
        # be genuinely unseen by every view, e.g. occluded or edge-on) — this is
        # what _dilate_texture_edges should treat as "needs real color spread
        # into it", same discipline as the old SENTINEL_UNCOVERED case.
        real_covered = np.zeros_like(chart["covered"])
        real_covered[ys, xs] = texel_covered
        n_covered_texels = int(texel_covered.sum())

        tex_img = _dilate_texture_edges(tex_img, real_covered, iterations=EDGE_DILATION_PX)

        result["regions"][region] = {
            "texture_rgb": tex_img,  # (H,W,3) uint8, RGB order
            "tex_w": chart["tex_w"], "tex_h": chart["tex_h"],
            "n_texels_covered": n_covered_texels,
            "n_vertices": len(vids),
            "n_dropped_flipped_triangles": chart["n_dropped_flipped_triangles"],
            "vertex_uv01": chart["vertex_uv01"],
            "kept_triangles_global": chart["kept_triangles_global"],
            # D38/D-seam debug metadata — was computed by
            # _apply_region_luminance_gain but never surfaced past `pending`;
            # exposing it here (purely additive, same pattern as
            # n_dropped_flipped_triangles above) so a real correction can be
            # confirmed/measured from the actual API response instead of
            # re-deriving luminance from the final dilated PNG, which mixes
            # in sentinel/dilated pixels the internal real-texels-only
            # computation never sees.
            "luminance_corrected": p.get("luminance_corrected"),
        }

    # D31 — GNM's own `hockey_mask` group (verified real data, see
    # face-oval-mask.ts's own docstring) excludes the eyeball surface
    # entirely (measured: `eyes`/`scleras`/`irises`/`pupils`/`eye_interiors`/
    # `eye_exteriors` all have ZERO vertex overlap with hockey_mask — GNM
    # authored it as a pure skin-surface mask, eyes handled separately). A
    # naive face-mask cut using hockey_mask alone renders literal black holes
    # where the eyes should be (confirmed via a real isolated render before
    # this fix) — this project has no separate eyeball mesh to fall back on,
    # so `face_mask` here is hockey_mask UNIONED with the real eye vertex
    # groups (`eyes` already covers sclera/iris/pupil/interior/exterior —
    # verified: eye_interiors + eye_exteriors both sum to exactly `eyes`'
    # own vertex count — and `eye_sockets` for the eyelid skin immediately
    # around them, which hockey_mask itself only partially covers).
    #
    # Phase 1/2 follow-up — `face_mask` alone is still what gnm_vertex_trust.py
    # protects as trust=1 (identity-critical, must stay byte-identical) and
    # is passed to `_attach_threejs_contract` UNCHANGED for that reason. But
    # the actual GEOMETRY CUT below now uses a real BROADER mask,
    # `visible_mask = face_mask | skin_exterior` — before Phase 1, cutting
    # the client-visible mesh down to `face_mask` alone (the old D-halfface
    # client-side behavior, see Canvas3D.tsx) was the only way to avoid
    # showing genuinely wrong pixels/geometry on the ears/scalp/neck (no
    # landmark evidence there, texture UV never calibrated for it). Phase 1's
    # `gnm_vertex_trust.py` now gives every one of those vertices a real,
    # safe fallback (position blended toward the template, color blended
    # toward its own real neighbor average) instead of raw uncontrolled
    # extrapolation — so hiding them entirely is no longer necessary, and
    # doing so was producing exactly the "hollow face floating in a black
    # void" look confirmed on a real running instance (no ears/scalp/neck/
    # forehead-top visible at any rotation) that this fix addresses. Verified
    # real membership: `skin_exterior` (GNM's own authored "outer skin
    # surface" group, 11,460/17,821 vertices) has ZERO overlap with
    # teeth/gums/tongue/mouth_sock/eye_interiors (checked directly against
    # the real asset) — i.e. widening to it can never expose the mouth/eye
    # cavity interior, only real exterior skin (forehead/scalp/ears/cheeks/
    # neck) that Phase 1's trust field already keeps safe.
    face_mask = _region_mask(names, groups, ["hockey_mask", "eyes", "eye_sockets"], V)
    skin_exterior = _region_mask(names, groups, ["skin_exterior"], V)
    visible_mask = face_mask | skin_exterior
    _attach_threejs_contract(result, triangles, V, visible_mask)
    return result


def _attach_threejs_contract(result: dict, triangles_full: np.ndarray, V: int, face_mask: np.ndarray) -> None:
    """Builds the exact index/groups/uv contract Three.js needs for
    `BufferGeometry.setIndex` + `addGroup` + a `uv` attribute, plus base64 PNGs
    for each region's texture — everything the API layer needs to ship as-is,
    no further geometry math required on the frontend. materialIndex 0 is
    reserved for every triangle NOT claimed by any region (still rendered from
    the existing, unchanged, real vertex-color data — see gnm_texture_bake.py
    — so nothing ever goes uncovered).

    D23 — a real vertex can sit on the shared boundary between two adjacent
    region vertex GROUPS (e.g. a mouth-corner vertex whose triangles on one
    side belong wholly to `mouth`, on the other wholly to `chin` — confirmed
    via real data: 70 such vertices for patient 0913e4c9, zero triangles ever
    claimed by two regions, only individual boundary vertices). Each such
    vertex has a legitimately DIFFERENT uv01 per chart (they're independent
    PCA-planar projections), but a naive single shared `uv_flat[vid]` can only
    hold one — whichever region processed last silently overwrote the other's,
    so the "losing" region's boundary triangles sampled a UV meant for a
    totally different chart, landing near-black/nonsense right at the seam
    (this was the real cause of the dark boundary line surviving D23's edge-
    dilation fix, which only fixes a single chart's own outer edge, not this
    cross-chart UV collision). Fix: the SECOND (and later) region to claim an
    already-claimed vid gets a brand-new duplicate vertex id appended past V,
    carrying that region's own uv — same standard "duplicate the vertex at a
    UV seam" approach any UV-unwrapper uses. The duplicate's position/color
    are NOT computed here (would touch the validated, unrelated
    fitted_positions/vertex_colors top-level fields) — `extra_source_vids`
    tells the caller which original vid to copy position from; region-texture
    materials never read the `color` attribute (see Canvas3D.tsx's
    `material-{i+1}`, no `vertexColors`), so a duplicate's color value is
    never sampled, only its buffer slot needs to exist and be in-bounds.
    """
    import cv2

    claimed_tri_set = set()
    index_chunks = []
    groups = []
    uv_by_vid: dict[int, np.ndarray] = {}
    extra_source_vids: list[int] = []
    owner_region: dict[int, str] = {}
    material_index = 1

    for region, data in result["regions"].items():
        tris = data["kept_triangles_global"]
        if len(tris) == 0:
            continue

        # first claim keeps the original vid; a repeat claim (real UV-chart
        # boundary collision, see docstring) gets a fresh duplicate vertex id.
        remap: dict[int, int] = {}
        for vid, uv in data["vertex_uv01"].items():
            vid = int(vid)
            if vid not in owner_region:
                owner_region[vid] = region
                target = vid
            else:
                target = V + len(extra_source_vids)
                extra_source_vids.append(vid)
            uv_by_vid[target] = uv
            remap[vid] = target

        tris = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in tris], dtype=np.int64)

        start = sum(len(c) for c in index_chunks) * 3
        index_chunks.append(tris)
        groups.append({"region": region, "start": start, "count": len(tris) * 3, "materialIndex": material_index})
        for a, b, c in data["kept_triangles_global"]:
            claimed_tri_set.add((int(a), int(b), int(c)))
        # encode PNG once here (region dict no longer needs the raw ndarray downstream)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(data["texture_rgb"], cv2.COLOR_RGB2BGR))
        data["texture_png_bytes"] = buf.tobytes() if ok else b""
        material_index += 1

    uv_flat = np.zeros((V + len(extra_source_vids), 2), dtype=np.float64)
    for vid, uv in uv_by_vid.items():
        uv_flat[vid] = uv

    # remainder: every triangle whose 3 vertices don't exactly match a claimed
    # (region, triangle) tuple — i.e. everything outside the 6 charted regions
    # (eyes, eyebrows, most of forehead/cheek margins) keeps using the
    # existing real vertex-color data — BUT ONLY within the real `hockey_mask`
    # face region. D31 — "cắt tại chân tóc, trước vành tai và dưới cằm, phần
    # sau rỗng hoàn toàn": every triangle with fewer than 2 of its 3 vertices
    # inside hockey_mask (back of head, scalp, ears, jaw sides, neck/shoulders
    # — same real vertex-group data already used for the camera-fit radius,
    # see face-oval-mask.ts's `maskTrianglesByFaceRegion` for the identical
    # >=2-of-3 rule, mirrored here so the ACTUAL rendered geometry matches
    # what the camera was already fit to, not just re-used for framing math).
    claimed_arr = np.array(list(claimed_tri_set), dtype=np.int64) if claimed_tri_set else np.zeros((0, 3), dtype=np.int64)
    if len(claimed_arr):
        claimed_lookup = {tuple(row) for row in claimed_arr.tolist()}
        remainder_mask = np.array([tuple(t.tolist()) not in claimed_lookup for t in triangles_full])
    else:
        remainder_mask = np.ones(len(triangles_full), dtype=bool)
    inside_count = face_mask[triangles_full].sum(axis=1)
    remainder_mask &= inside_count >= 2
    remainder_tris = triangles_full[remainder_mask]
    start = sum(len(c) for c in index_chunks) * 3
    index_chunks.append(remainder_tris)
    groups.append({"region": "_base_vertex_color", "start": start, "count": len(remainder_tris) * 3, "materialIndex": 0})

    full_index = np.concatenate(index_chunks, axis=0).reshape(-1) if index_chunks else np.array([], dtype=np.int64)
    result["threejs"] = {
        "index": full_index.tolist(),
        "groups": groups,
        "uv": uv_flat.reshape(-1).tolist(),
        "material_count": material_index,  # 0 = base vertex-color, 1..N-1 = region textures (see groups[].region)
        # D23 — vertex ids >= V referenced by `index`/`uv` above; each one is a
        # position-only duplicate of extra_source_vids[id - V] (see this
        # function's own docstring). Caller must extend its position (and any
        # other per-vertex attribute) buffer by this many entries before using
        # `index` — the base vertex-color path (materialIndex 0, `_base_vertex_color`
        # group) never references these, only region-texture groups do.
        "extra_source_vids": extra_source_vids,
    }
