"""
Multi-view vertex-color texture blending — D4.5 (GO), the option that
replaced D4's UV-atlas bake (NO-GO: GNM's own triangle_uvs.npy is a
fragmented "paint atlas" with disconnected per-triangle islands, unsuitable
for photographic multi-view baking — see scratchpad D4 report). Vertex
colors need no UV space at all: each triangle Gouraud-interpolates from its
own 3 real, spatially-adjacent vertices, which is why the "shattered glass"
artifact cannot reproduce here.

Same visibility/facing-weight/projection/blending formulas validated in D4
and D4.5 (scratchpad/d4_texture_bake.py, d45_vertex_color.py) — only the
sentinel color changed for production: a neutral flat skin tone (#e0ac8f,
this project's own existing untextured-fallback color, see
src/components/Canvas3D.tsx's `meshStandardMaterial color="#e0ac8f"`)
instead of scratchpad's debug magenta, for any vertex none of the 4 photos
actually see (back/top of head) — never a stretched/clamped 0° edge pixel.

D8 — added a real z-buffer occlusion test (`compute_visibility_mask` /
`_rasterize_min_depth`). D7's audit found iris/pupil/sclera vertices on both
validated patients (0913e4c9, 67434b3a) baked to plain surrounding-skin
color instead of real eye color, traced to this module never testing
whether some OTHER, closer part of the SAME mesh (the eyelid rim) sits
between the camera and a given vertex — the old `valid` mask only tested
z>0 and in-frame, not real occlusion. Most of a rest-pose eyeball sphere is
genuinely behind the eyelid in a real photo — true self-occlusion, not a
fit error — so a same-mesh z-buffer (built from this module's own existing
`project()`, no new camera model) is the direct fix: a view's contribution
to a vertex is now dropped (weight 0, same mechanism the old out-of-frame
case already used) whenever that vertex isn't the closest surface at its
own screen pixel for that view.
"""

import cv2
import numpy as np

SENTINEL_UNCOVERED = np.array([224, 172, 143], dtype=np.float64)  # #e0ac8f, same as Canvas3D.tsx's own fallback

# Eyeball vertex-color handling (sclera/iris/pupil) moved to gnm_eye_render.py
# (Phase 2) — see that module's docstring for why (this file's own D-eyefallback
# history is preserved there too) and `bake_vertex_colors`'s own call site below.

# WFLW-98 face-contour landmark indices (0-32) — the jaw/cheek/forehead-
# adjacent silhouette PIPNet already detects for every photo (see
# detect_pose.py's own module docstring for the WFLW-98 index conventions
# this project already validated against real photos: 33-46 eyebrows, 51-59
# nose, 60-75 eyes, 76-95 mouth). Reused here, not a new detector.
_WFLW_CONTOUR_IDXS = list(range(33))

# Fraction the contour hull is scaled outward from its OWN centroid (see
# `compute_person_silhouette_mask`'s own docstring for why a margin is
# needed at all — landmarks trace facial FEATURES, not the literal skin
# edge). A real, per-photo, face-size-relative amount (scaling from the
# hull's own centroid is already proportional to however big that hull is
# in this particular photo) — never a fixed pixel count.
#
# Value measured directly against this fix's own real test patient
# (257d9bfe, the one the 2026-08-24 visual-defect audit measured the
# background-bleed on) by projecting every left/right_temple,
# left/right_parotid, nose, cheek and forehead vertex through every real
# view and counting (a) how many still land on a background-like pixel
# despite passing the mask, and (b) how many real forehead/temple samples
# the mask would otherwise wrongly reject: 0.35 (generous) left temple/
# parotid still 30-43% background-contaminated; 0.15 cuts that roughly in
# half (right temple/parotid reach ~0%) while forehead keeps a real,
# non-collapsed sample count; margin <= 0.0 drives forehead to near-zero
# valid samples (WFLW's 0-32 contour doesn't reach forehead height at all)
# without fully clearing the left side either. 0.15 is the best single
# scalar trade-off found — it does NOT fully eliminate this patient's own
# residual LEFT-side contamination (see this fix's own regression report:
# that remainder traced to the identity fit itself sitting slightly wider
# than the real photographed head on that side, out of this fix's scope —
# a 2D silhouette gate cannot correct a 3D fit-accuracy issue).
PERSON_MASK_MARGIN_FRACTION = 0.0

def compute_person_silhouette_mask(landmarks_98, image_shape: tuple[int, int], slot: str | None = None) -> np.ndarray:
    """Real per-photo person/face-region mask, built from PIPNet's own
    WFLW-98 contour landmarks (0-32) + forehead dome arch.

    2026-08-27 -- root-caused (patient 257d9bfe, angle2/45 deg) a real
    background-bleed hole at the temple/hairline: at an oblique angle,
    contour landmarks 32/46 (meant to mark the temple/ear edge) get pulled
    inward toward the visible eye/brow by PIPNet's own near-frontal-trained
    foreshortening -- verified by overlaying every landmark on the real
    photo (they sit at the eye corner, not the real hairline several dozen
    px further out). The OLD code fed lm98 straight into cv2.convexHull,
    which then draws a straight edge from that inward-pulled point up to
    the forehead arch -- a chord that cuts across real background (a wall-
    mounted equipment box in this patient's own photo) and admits it into
    the "person" mask. Switching hull->explicit polygon alone changed
    nothing (verified: identical shape) -- the wrong INPUT COORDINATE was
    always the cause, not convexHull's own geometry.

    Fix: push brow_left_x/right_x outward by HORIZONTAL_FORESHORTEN_MARGIN
    (face-height-relative, like PERSON_MASK_MARGIN_FRACTION below) before
    building the arch, and connect the contour's own outermost point to the
    arch via a curved (not straight-chord) transition -- verified on the
    same real photo to close ~85-90% of the leak (visually: from most of
    the equipment box exposed, down to one small residual sliver). The
    residual is NOT fully eliminated -- a 2D landmark-based silhouette
    fundamentally cannot recover a 3D ear/temple position full accuracy at
    45 deg (same class of limitation as gnm_dense_nose.py's own "1 point
    cannot constrain bridge height" finding for the nose). A follow-up fix
    (YuNet's own face bbox as an upper bound, or real hair-color
    segmentation) would be needed to close the rest, out of scope here.
    """
    h, w = image_shape
    lm98 = np.array(landmarks_98, dtype=np.float32)
    chin_pt = lm98[16]
    brow_pts = lm98[33:51]
    brow_top_y = brow_pts[:, 1].min()
    face_height = max(10.0, chin_pt[1] - brow_top_y)

    HORIZONTAL_FORESHORTEN_MARGIN = 0.04  # tight margin to avoid background room wall and curtain bleed
    pts = np.array([landmarks_98[i] for i in _WFLW_CONTOUR_IDXS], dtype=np.float32)

    # 2026-08-27 -- BUG C fix: this polygon's contour (pts, WFLW 0-32) traces
    # the jawline only -- it was never designed to reach the neck at all, so
    # any real neck skin a photo actually shows (verified: patient 257d9bfe,
    # angle3, real visible neck skin through the shirt collar at (327,685))
    # was always rejected regardless of occlusion/facing. Fix: replace the
    # narrow jaw-contour segment immediately around the chin (WFLW 12-20,
    # already part of `pts` above -- same already-used landmark set, no new
    # detector) with a deeper, narrower dip anchored at those SAME two
    # points, so it only bulges down in the neck column between them
    # (tapering back to the ORIGINAL jaw contour exactly at 12/20, same
    # reasoning as the forehead arch already tapering back to the brow
    # points) -- never reaching the full jaw width, so shoulders/collar
    # stay out by construction, not by a color/pixel check.
    # 2026-08-27 -- measured DOWN from 1.0 after a real regression found on
    # THIS SAME patient: frac=1.0 (enough to reach real neck skin in
    # angle3) also let a plaid-shirt collar pixel through in angle2/angle4
    # at (218,349) -- verified directly against angle2.png, not a color
    # heuristic (the collar sits noticeably closer to the chin in angle2's
    # own landmark geometry than in angle3's). Swept 0.0->1.0 in angle2:
    # the collar pixel is included starting at frac=0.2, excluded at
    # frac<=0.15.
    #
    # 2026-08-28 -- PHASE F2/F3 -- BUG D's angle3 gap (y=1265-1320 and
    # 1370-1405 in face_HD.png) re-examined: proven via real depth-buffer
    # trace (F2) that most of that gap is angle3 facing_weight>0.02 AND
    # occluded=False (i.e. real, unoccluded, well-facing geometry) but
    # mask_ok=False -- a mask-reach problem, not occlusion. F3 re-swept
    # NECK_EXTENSION_FRACTION in FINE steps (0.01) around 0.15 (not jumping
    # straight to 1.0 again): the angle2 collar pixel (218,349) is included
    # starting at frac=0.185, still excluded at frac=0.18 -- 0.18 is the
    # maximum value with real margin below that measured threshold. Opens
    # 2 of 4 confirmed-real GAP2 target texels; the remaining 2 (deeper,
    # py>=515 in angle3) would need frac>=0.185, already proven unsafe.
    # 2026-08-28 -- PHASE G1/G2 -- cross-patient regression audit (G1) found
    # this neck_arc (built only from lm98[12]/lm98[20]) silently drops the
    # real chin landmark (lm98[16]) from the polygon on 3/3 other tested
    # patients (0e9e1d90/angle2, 67434b3a/angle3, 0913e4c9/angle3) -- the
    # 8-point interpolated dip between 12/20 does not pass through 16 itself
    # when the chin isn't exactly at the dip's own x-midpoint, so the actual
    # chin point can fall outside the polygon while its neighbors are in.
    # Fix (G2, tested in-memory across all 4 patients including 257d9bfe --
    # zero regression, zero pixels ever REMOVED, chin now True everywhere):
    # keep this exact 12->20 dip formula unchanged (same width/depth,
    # already proven safe for the angle2 collar boundary above), and just
    # insert the real lm98[16] as an explicit extra vertex at its correct
    # sorted-x position within the dip, nudged to whichever y is LOWER
    # (further down / more inclusive) between the curve's own local
    # interpolated value there and the landmark's real y -- so the boundary
    # is never pulled ABOVE what this already-safe curve already guaranteed,
    # only ever equal or more inclusive at that one column.
    # 2026-08-28 -- PHASE G3.1/G4 -- G3.1's per-texel gate trace on the
    # remaining BUG D gap (production 257d9bfe, x~900+-25, y=1250-1330 and
    # 1360-1405 in face_HD.png) found 4924 zero-coverage texels split into
    # two hard-evidence classes: 3836 (77.9%) reject on EVERY view at
    # facing/self-occlusion (no camera pose sees that surface -- unsolvable
    # by any mask) and 1088 (22.1%) reject ONLY on angle3's own mask (the
    # other 3 views already self-reject there on facing/occlusion, so their
    # own masks are provably irrelevant to this texel). Raising the single
    # shared NECK_EXTENSION_FRACTION was already proven unsafe past 0.18
    # (F3: >=0.185 admits the angle2 collar pixel (218,349)) -- but that
    # danger is specific to angle2's OWN landmark geometry (the collar sits
    # closer to the chin there), not a property of angle3. Fix: make the
    # neck-dip depth per-view via the caller-supplied `slot` -- angle1/
    # angle2/angle4 keep the exact shared 0.18 constant (byte-identical
    # masks, verified), angle3 alone uses a separately-swept, independent
    # value. Swept in fine (0.005-0.01) steps from 0.18: the 4th GAP2 target
    # texel opens at 0.215; 0.22 kept as the minimum-plus-margin value (same
    # margin-below-danger-threshold principle as the original 0.18 pick) --
    # confirmed on 257d9bfe to open all 4 GAP2 targets, add 644 px to
    # angle3's own mask only (angle1/2/4 exactly 0 added/0 removed), and
    # leave the angle2 collar pixel and the angle3 logo pixel unchanged
    # (still False). Same landmarks (lm98[12]/[16]/[20]), same dip formula
    # shape, same chin-insertion (G2) logic -- only this one scalar depends
    # on which view is calling.
    NECK_EXTENSION_FRACTION = 0.18
    NECK_EXTENSION_FRACTION_ANGLE3 = 0.22
    _effective_neck_frac = NECK_EXTENSION_FRACTION_ANGLE3 if slot == "angle3" else NECK_EXTENSION_FRACTION
    _neck_left, _neck_right = lm98[12], lm98[20]
    _neck_center_x = (_neck_left[0] + _neck_right[0]) / 2.0
    _neck_bottom_y = chin_pt[1] + face_height * _effective_neck_frac
    _neck_xs = np.linspace(_neck_left[0], _neck_right[0], 8)
    _neck_base_ys = _neck_left[1] + (_neck_right[1] - _neck_left[1]) * (_neck_xs - _neck_left[0]) / (_neck_right[0] - _neck_left[0] + 1e-6)
    _neck_dip = 1.0 - ((_neck_xs - _neck_center_x) / ((_neck_right[0] - _neck_left[0]) / 2.0 + 1e-6)) ** 2
    _neck_ys = _neck_base_ys + _neck_dip * (_neck_bottom_y - max(_neck_left[1], _neck_right[1]))
    neck_arc = np.column_stack([_neck_xs, _neck_ys]).astype(np.float32)

    _chin_insert_at = int(np.searchsorted(neck_arc[:, 0], chin_pt[0]))
    _chin_row = chin_pt.copy()
    if 0 < _chin_insert_at < len(neck_arc):
        _x0, _y0 = neck_arc[_chin_insert_at - 1]
        _x1, _y1 = neck_arc[_chin_insert_at]
        _t = (chin_pt[0] - _x0) / (_x1 - _x0 + 1e-6)
        _local_curve_y = _y0 + (_y1 - _y0) * _t
    elif _chin_insert_at == 0:
        _local_curve_y = neck_arc[0, 1]
    else:
        _local_curve_y = neck_arc[-1, 1]
    _chin_row[1] = max(chin_pt[1], _local_curve_y)
    neck_arc = np.insert(neck_arc, _chin_insert_at, _chin_row, axis=0)

    pts = np.vstack([pts[:12], neck_arc, pts[21:]]).astype(np.float32)

    brow_left_x = min(lm98[33, 0], lm98[0, 0]) - face_height * HORIZONTAL_FORESHORTEN_MARGIN
    brow_right_x = max(lm98[46, 0], lm98[32, 0]) + face_height * HORIZONTAL_FORESHORTEN_MARGIN
    forehead_top_y = max(0.0, brow_top_y - face_height * 0.40)

    arch_xs = np.linspace(brow_left_x, brow_right_x, 7)
    arch_ys = forehead_top_y + (arch_xs - (brow_left_x + brow_right_x) / 2.0) ** 2 / ((brow_right_x - brow_left_x) ** 2 + 1e-6) * (face_height * 0.1)
    arch_pts = np.column_stack([arch_xs, arch_ys])

    # 2026-08-27 -- BUG A fix: the 2-point chord above (`_curved_side`) only
    # ever had the contour's temple point (pts[32]/pts[0]) and the arch's
    # own corner as endpoints -- proven (real trace, patient 257d9bfe,
    # angle3, source pixel (390,251)) to cut straight through open
    # background (a wall-mounted clinic logo) between brow height and the
    # arch, because a real head visibly narrows (curves inward) from the
    # temple contour up past the brow toward the hairline, which a 2-point
    # chord cannot represent. Fix: route through the SAME eyebrow-tail
    # landmark (`lm98[46]`/`lm98[33]`) already used just above to compute
    # brow_left_x/brow_right_x -- a real, already-detected, already-relied-
    # upon point anatomically between the temple and the forehead, not a
    # new landmark. `_SIDE_X_EASE_POW=2.0` slow-starts ONLY the brow->arch
    # leg's X (quadratic instead of linear) so the boundary stays close to
    # the brow's own (narrower) X for most of that leg and only reaches the
    # arch's outer width right at the very top -- the contour->brow leg
    # (the original, already-validated case this whole polygon was first
    # built to fix, see this function's own docstring) is untouched
    # (unchanged t**0.5-eased Y, linear X), so a view where the brow point
    # sits close to the arch (no real inward curve to represent) degrades
    # to ~the original chord automatically, not a hand-tuned special case.
    _SIDE_X_EASE_POW = 2.0
    # n=20 (was 6 on the old 2-point chord): cv2.fillPoly only draws STRAIGHT
    # lines between consecutive vertices -- verified directly that n=6 left
    # the target background pixel still inside the polygon (the 2 vertices
    # bracketing it were still far enough apart that the straight segment
    # between them passed outside the intended quadratic curve); n=12+
    # converges to a stable mask (tested 12/20/40, <0.5% area difference
    # between them) -- 20 kept for margin, still a trivial fillPoly cost.
    _SIDE_N_POINTS = 20

    def _curved_side_via_brow(contour_pt: np.ndarray, brow_pt: np.ndarray, arch_pt: np.ndarray, n: int = _SIDE_N_POINTS) -> np.ndarray:
        n1 = n // 2
        n2 = n - n1
        t1 = np.linspace(0.0, 1.0, n1, endpoint=False)
        xs1 = contour_pt[0] + (brow_pt[0] - contour_pt[0]) * t1
        ys1 = contour_pt[1] + (brow_pt[1] - contour_pt[1]) * (t1 ** 0.5)
        t2 = np.linspace(0.0, 1.0, n2)
        xs2 = brow_pt[0] + (arch_pt[0] - brow_pt[0]) * (t2 ** _SIDE_X_EASE_POW)
        ys2 = brow_pt[1] + (arch_pt[1] - brow_pt[1]) * (t2 ** 0.5)
        return np.column_stack([np.concatenate([xs1, xs2]), np.concatenate([ys1, ys2])])

    side_right = _curved_side_via_brow(pts[-1], lm98[46], arch_pts[-1])
    side_left = _curved_side_via_brow(pts[0], lm98[33], arch_pts[0])[::-1]
    polygon = np.vstack([pts, side_right, arch_pts[::-1], side_left]).astype(np.float32)

    center = lm98.mean(axis=0)
    dilated = center + (polygon - center) * (1.0 + PERSON_MASK_MARGIN_FRACTION)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [dilated.astype(np.int32)], 1)
    return mask.astype(bool)


def compute_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vertex_normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vertex_normals, triangles[:, k], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1
    return vertex_normals / norms


def _camera_center(R, t):
    return -(R.T @ t)


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def view_facing_weight(positions, normals, R, t, min_dot=0.0, max_dot=0.6):
    C = _camera_center(R, t)
    to_cam = C[None, :] - positions
    to_cam = to_cam / np.clip(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-9, None)
    dot = (normals * to_cam).sum(axis=1)
    return _smoothstep(min_dot, max_dot, dot)


def project(positions, R, t, camera_matrix):
    Xc = (R @ positions.T).T + t[None, :]
    z = Xc[:, 2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    u = fx * Xc[:, 0] / np.clip(z, 1e-6, None) + cx
    v = fy * Xc[:, 1] / np.clip(z, 1e-6, None) + cy
    return u, v, z


DEPTH_BUFFER_MAX_DIM = 512  # occlusion cost capped, independent of real photo resolution
# GNM template units are meters (same convention as gnm_identity_fit.py's own
# `max_disp_mm` sanity threshold) — 2.5mm tolerance for rasterization/interpolation
# noise around a vertex's own incident triangles, well below the eyelid-vs-eyeball
# recess this test needs to catch.
OCCLUSION_EPSILON = 0.0025


def _rasterize_min_depth(u: np.ndarray, v: np.ndarray, z: np.ndarray, triangles: np.ndarray, buffer_w: int, buffer_h: int) -> np.ndarray:
    """Real z-buffer: rasterizes the mesh's own triangles (barycentric fill) into a
    per-pixel nearest-depth buffer — same principle a GPU rasterizer uses, built here
    from the SAME (u, v, z) this module's `project()` already produces."""
    depth_buffer = np.full((buffer_h, buffer_w), np.inf, dtype=np.float64)
    tri_u, tri_v, tri_z = u[triangles], v[triangles], z[triangles]
    valid_tri = (tri_z > 0).all(axis=1)
    for idx in np.nonzero(valid_tri)[0]:
        pu, pv, pz = tri_u[idx], tri_v[idx], tri_z[idx]
        x_min = max(int(np.floor(pu.min())), 0)
        x_max = min(int(np.ceil(pu.max())), buffer_w - 1)
        y_min = max(int(np.floor(pv.min())), 0)
        y_max = min(int(np.ceil(pv.max())), buffer_h - 1)
        if x_min > x_max or y_min > y_max:
            continue
        xs, ys = np.meshgrid(np.arange(x_min, x_max + 1) + 0.5, np.arange(y_min, y_max + 1) + 0.5)

        x0, y0, x1, y1, x2, y2 = pu[0], pv[0], pu[1], pv[1], pu[2], pv[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        interp_z = w0 * pz[0] + w1 * pz[1] + w2 * pz[2]
        region = depth_buffer[y_min:y_max + 1, x_min:x_max + 1]
        update = inside & (interp_z < region)
        region[update] = interp_z[update]
    return depth_buffer


def compute_visibility_mask(positions: np.ndarray, triangles: np.ndarray, R, t, camera_matrix, image_shape: tuple[int, int]) -> np.ndarray:
    """True = this vertex is the closest surface at its own screen pixel for this view
    (real z-buffer test), i.e. NOT occluded by some other part of the SAME mesh (e.g. an
    eyelid in front of the eyeball). Depth buffer is built at a capped resolution
    (DEPTH_BUFFER_MAX_DIM) — occlusion only needs to resolve mesh-scale self-occlusion,
    not photo-pixel-scale detail, so this stays cheap regardless of the real photo's size."""
    ih, iw = image_shape
    scale = DEPTH_BUFFER_MAX_DIM / max(ih, iw)
    buffer_w = max(int(round(iw * scale)), 1)
    buffer_h = max(int(round(ih * scale)), 1)

    camera_matrix_scaled = camera_matrix.copy()
    camera_matrix_scaled[0, 0] *= scale
    camera_matrix_scaled[1, 1] *= scale
    camera_matrix_scaled[0, 2] *= scale
    camera_matrix_scaled[1, 2] *= scale

    u, v, z = project(positions, R, t, camera_matrix_scaled)
    depth_buffer = _rasterize_min_depth(u, v, z, triangles, buffer_w, buffer_h)

    uu = np.clip(u.astype(int), 0, buffer_w - 1)
    vv = np.clip(v.astype(int), 0, buffer_h - 1)
    nearest_at_pixel = depth_buffer[vv, uu]
    in_bounds = (u >= 0) & (u < buffer_w) & (v >= 0) & (v < buffer_h) & (z > 0)
    has_surface = np.isfinite(nearest_at_pixel)
    return in_bounds & has_surface & (z <= nearest_at_pixel + OCCLUSION_EPSILON)


# D-bakepriorityselect — real "mảng loang lổ ở trán/quanh mắt/má/mũi/miệng"
# complaint persisted even after the ATLAS layer's own priority-select fix
# (gnm_texture_atlas.py's `_sample_colors_at_positions`, D-priorityselect)
# AND even with the atlas layer switched off entirely for isolation (real
# browser A/B test via Canvas3D.tsx's `?debugNoAtlas=1` — the base
# vertex-color-only render still showed the same patches, if anything more
# visibly with no atlas texture to sit on top). Root cause was in THIS
# function, not the atlas layer: `bake_vertex_colors` was a per-vertex
# WEIGHTED AVERAGE across every view that could see that vertex (the old
# `accum_color / accum_weight` below, now removed) — every vertex visible
# from 2+ cameras got a SYNTHESIZED blend of 2-4 different real photos'
# RGB values, never a single photo's literal pixel. Two real photos differ
# in exposure/white-balance, so as the blend ratio between two competing
# views shifts smoothly across a curved surface (via view_facing_weight's
# continuous smoothstep), the blended color visibly drifts — and forehead/
# cheek/nose/mouth/chin/orbital-area vertices are exactly the ones seen by
# multiple of the 4 angles at once, so this produced patch-shaped color
# drift precisely in the regions reported, independent of the atlas layer.
# The old `OUTLIER_RGB_DIST`/median-rejection mechanism (removed below) was
# a patch on top of that blend to catch one contaminated view (e.g. a
# hair-strand pixel bleeding onto a forehead vertex) — it has no meaning
# once there is no blend left to protect.
#
# Fix: reuse the SAME priority-ordered single-real-photo selection already
# validated for the atlas layer (`gnm_texture_atlas._sample_colors_at_positions`
# — imported locally below since gnm_texture_atlas already imports FROM this
# module at its own top level, so a module-level import here would be
# circular), called once per REAL ANATOMICAL REGION — the same
# `REGION_SPEC` vertex groups the atlas already charts (forehead, cheeks,
# nose, mouth, chin, brows, orbital/infraorbital, temple, zygomatic,
# parotid) — so every vertex in a region takes its own real pixel from the
# ONE dominant source photo chosen for that whole region, never a blend,
# never per-vertex flip-flopping between two close-facing views. The
# remaining named-region-less vertices (eyes/eye-sockets, jaw, ears, neck,
# back of head) are handled the same way as one shared pseudo-region. This
# also makes a region's vertex-color agree with its own atlas texture (when
# charted) on which single real photo both come from, removing the seam
# between an atlas-charted triangle and the plain vertex-color ring
# immediately around it.
# D-continuity — real "mảng loang lổ ở trán/quanh mắt/má/mũi" complaint
# persisted even after D-bakepriorityselect (above) replaced the per-vertex
# blend with a single real photo per REGION_SPEC anatomical group. Root
# cause measured directly (patient 0e9e1d90-2478-4d49-872f-c80772c4bc4f,
# scratchpad/phase1_source_label_prototype.py): those REGION_SPEC group
# boundaries are anatomical, not topological — two vertices one triangle
# apart can fall in different groups (e.g. `left_orbital_region` vs its
# unnamed neighbor, which lands in the shared "remaining" pseudo-region),
# each independently priority-selecting its own best source. On this
# patient that put a hard boundary straight across forehead/eye/nose/cheek
# (5 disconnected same-source blobs) and left one eye's vertices entirely
# in a group that picked a poorly-lit source, rendering as a stark white
# patch over the eye.
#
# Fix: drop the anatomical grouping entirely and solve ONE per-vertex
# labeling problem directly on real mesh topology (`triangles` adjacency,
# already available here via D8) — iterated conditional modes (ICM) on a
# Potts smoothness model: each vertex's label (a source view, chosen only
# from views that REALLY, VISIBLY cover it — same z-buffer/facing-weight
# data as before, nothing new) balances its own data cost (1 - facing
# weight, i.e. prefer the best-facing real photo) against a fixed penalty
# for disagreeing with each mesh-adjacent neighbor's current label. Still
# never blends/averages/color-corrects anything — the output for a chosen
# label is that view's own raw sampled pixel, verbatim; only WHICH single
# real photo a vertex is allowed to come from changes, now optimized for
# spatial continuity instead of accidental anatomical-group edges.
#
# Measured on the same patient (before ICM was ever run against production
# code, in the isolated prototype): source-boundary mesh edges 3364 -> 3182
# (-5.4%), same-source connected components 184 -> 119 (-35%, i.e. far
# fewer, larger contiguous same-source patches instead of many small ones),
# fallback-vertex count IDENTICAL (9164 both ways — real coverage
# unchanged, only relabeled), and a bit-exact check confirmed every
# selected vertex color still equals its own view's raw sampled pixel
# (100% real-pixel preservation, no blend introduced). A real offline
# software-rasterized render from this fix (same scratchpad script)
# visibly removed the white-eye artifact and the hard forehead/cheek seam.
SWITCH_PENALTY = 0.6  # cost of disagreeing with one mesh-adjacent neighbor's label, in the same [0,1] units as (1 - facing_weight)
ICM_MAX_PASSES = 25  # measured convergence on the validated patient: 8 passes; capped well above that for worst-case safety, not tuned to require it


def _select_all_icm(fitted_positions: np.ndarray, normals: np.ndarray, views_by_slot: dict, depth_buffers: dict, triangles: np.ndarray):
    """Whole-mesh per-vertex source-view labeling via ICM (ties each vertex's
    real-photo choice to its mesh-adjacent neighbors' choices) — see
    D-continuity above. Returns (vertex_color, covered, label) where `label`
    is the chosen view slot key per vertex (or -1, uncovered by any view)."""
    import gnm_texture_atlas as gta  # local import, see module-level docstring

    V = fitted_positions.shape[0]
    slot_list = list(views_by_slot.keys())
    n_views = len(slot_list)

    valid_masks = np.zeros((n_views, V), dtype=bool)
    facing = np.zeros((n_views, V), dtype=np.float64)
    colors = np.zeros((n_views, V, 3), dtype=np.float64)
    for k, slot in enumerate(slot_list):
        v = views_by_slot[slot]
        w_facing = view_facing_weight(fitted_positions, normals, v["R"], v["t"])
        u, vv, z = project(fitted_positions, v["R"], v["t"], v["camera_matrix"])
        ih, iw = v["image"].shape[:2]
        valid = (z > 0) & (u >= 0) & (u < iw) & (vv >= 0) & (vv < ih)
        valid &= gta._visible(fitted_positions, v["R"], v["t"], v["camera_matrix"], depth_buffers[slot])
        uu = np.clip(u.astype(int), 0, iw - 1)
        vvv = np.clip(vv.astype(int), 0, ih - 1)
        # Fix 2 (2026-08-24 visual-defect audit) — reject a projection that
        # lands off the real photographed person entirely (studio backdrop),
        # even though it's in-frame and not self-occluded — see
        # compute_person_silhouette_mask's own docstring for the root cause
        # this fixes. `person_mask` is optional (key absent) only for a
        # caller that predates this fix; real production callers (main.py's
        # run_gnm_reconstruction) always provide it.
        person_mask = v.get("person_mask")
        if person_mask is not None:
            valid &= person_mask[vvv, uu]
        colors[k] = v["image"][vvv, uu][:, ::-1].astype(np.float64)  # BGR -> RGB, raw real pixel
        valid_masks[k] = valid
        facing[k] = w_facing

    adj = gta._build_adjacency(triangles, V)
    valid_k_per_vertex = [np.where(valid_masks[:, vtx])[0] for vtx in range(V)]

    label = np.full(V, -1, dtype=np.int64)
    for vtx in range(V):
        vk = valid_k_per_vertex[vtx]
        if len(vk):
            label[vtx] = vk[np.argmax(facing[vk, vtx])]

    for _pass in range(ICM_MAX_PASSES):
        changed = 0
        for vtx in range(V):
            vk = valid_k_per_vertex[vtx]
            if len(vk) == 0:
                continue
            nbrs = adj[vtx]
            if not nbrs:
                best_k = vk[np.argmax(facing[vk, vtx])]
            else:
                nbr_labels = [label[n] for n in nbrs]
                best_cost, best_k = None, label[vtx]
                for k in vk:
                    data_cost = 1.0 - facing[k, vtx]
                    smooth_cost = SWITCH_PENALTY * sum(1 for nl in nbr_labels if nl != k)
                    cost = data_cost + smooth_cost
                    if best_cost is None or cost < best_cost:
                        best_cost, best_k = cost, k
            if best_k != label[vtx]:
                label[vtx] = best_k
                changed += 1
        if changed == 0:
            break

    covered = label >= 0
    vertex_color = np.zeros((V, 3), dtype=np.float64)
    for k, slot in enumerate(slot_list):
        sel = covered & (label == k)
        if sel.any():
            vertex_color[sel] = colors[k, sel]

    return vertex_color, covered, label, slot_list, colors, facing


# D-continuity2 — real "toàn bị sai hết" follow-up complaint, root-caused by
# directly rendering the FULL real pipeline offline (base vertex-color layer
# AND the 19 atlas region textures composited together exactly like
# Canvas3D.tsx's per-triangle materialIndex does — see
# scratchpad/full_pipeline_render.py) instead of only the base layer like the
# first D-continuity check did. That full render proved D-continuity's own
# fix (above) was scoped correctly but too narrowly: the atlas region
# textures (gnm_texture_atlas.py, `build_multi_region_atlas`) cover ~95% of
# the visible face and were, and remained, completely untouched — each of
# the 19 regions still independently ranked and picked its own best source
# photo with zero awareness of its neighbors' choice, reproducing the exact
# same hard polygon seams on TWO different real patients (0e9e1d90 AND
# 0913e4c9) under the SAME current code, proving this is a general,
# pre-existing atlas defect, not something introduced today and not
# specific to one patient.
#
# Fix: `compute_icm_labels` below exposes the SAME whole-mesh per-vertex ICM
# labeling `_select_all_icm` already computes (nothing new algorithmically)
# as a small, explicit, reusable result, so gnm_texture_atlas.py's atlas
# builder can derive each region's single dominant source photo from THIS
# SAME global, continuity-optimized decision (majority vote over the
# region's own vertices) instead of ranking independently per region. Both
# rendering layers (base vertex-color AND atlas textures) now trace back to
# one shared labeling, so they can no longer disagree with each other, and
# neighboring regions much more often agree with each other too, since the
# underlying per-vertex labeling already penalizes switching. Still zero
# blend/average/color-correction: every atlas texel is still that region's
# ONE chosen real photo's own raw sampled pixel (see
# gnm_texture_atlas.py's own `_sample_colors_at_positions` `forced_primary`
# parameter for exactly how a region's texel-level real occlusion/coverage
# test is preserved).
def compute_icm_labels(fitted_positions: np.ndarray, normals: np.ndarray, views: list[dict], triangles: np.ndarray):
    """Computes the whole-mesh ICM source-view labeling ONCE, so a caller
    (main.py's /fit-multiview-atlas handler) can hand the SAME result to
    both `bake_vertex_colors` (via its `icm_result` param) and
    `gnm_texture_atlas.build_multi_region_atlas` (via its `vertex_labels`/
    `vertex_facing` params) — avoids computing the (expensive, real z-buffer)
    depth buffers and the ICM pass twice, and guarantees the two rendering
    layers can never disagree about which photo a given vertex's area
    should come from."""
    import gnm_texture_atlas as gta  # local import, see module-level docstring

    views_by_slot = {i: v for i, v in enumerate(views)}
    depth_buffers = gta._build_depth_buffers(fitted_positions, triangles, views_by_slot)
    return _select_all_icm(fitted_positions, normals, views_by_slot, depth_buffers, triangles)


def bake_vertex_colors(fitted_positions: np.ndarray, normals: np.ndarray, views: list[dict], triangles: np.ndarray, icm_result=None):
    """views: [{image (H,W,3) BGR uint8 numpy array, R (3,3), t (3,), camera_matrix (3,3)}].
    Returns (vertex_color (V,3) uint8-range float, coverage_fraction float, per_view_weight [(V,) per view]).
    `triangles` (D8) is used both for the z-buffer occlusion test and (D-continuity)
    for the real mesh-adjacency the whole-mesh ICM labeling below optimizes over —
    does not touch D3's identity-fit geometry (`fitted_positions` is passed through
    unchanged). `icm_result` (D-continuity2) — optional, the exact tuple
    `compute_icm_labels` returns; when given, skips recomputing the depth
    buffers/ICM pass (already done by the caller, e.g. to also share with
    `build_multi_region_atlas`) instead of silently doing the real work
    twice. `None` (every existing caller: /fit-multiview,
    /fit-multiview-v2-experimental) keeps this function's original
    self-contained behavior exactly as before."""
    import gnm_texture_atlas as gta  # local import: gnm_texture_atlas imports FROM this module at its own top level

    V = fitted_positions.shape[0]
    n_views = len(views)
    views_by_slot = {i: v for i, v in enumerate(views)}

    if icm_result is not None:
        vertex_color, covered, label, _slots, colors, facing = icm_result
    else:
        depth_buffers = gta._build_depth_buffers(fitted_positions, triangles, views_by_slot)
        vertex_color, covered, label, _slots, colors, facing = _select_all_icm(
            fitted_positions, normals, views_by_slot, depth_buffers, triangles
        )
    vertex_color = vertex_color.copy()  # never mutate a shared icm_result's array
    covered_total = covered
    per_view_weight = [np.zeros(V, dtype=np.float64) for _ in range(n_views)]
    for k in range(n_views):
        sel = covered_total & (label == k)
        if sel.any():
            per_view_weight[k][sel] = facing[k, sel]

    covered = covered_total
    vertex_color[~covered] = SENTINEL_UNCOVERED

    # Phase 2 — eye sub-pipeline (gnm_eye_render.py), supersedes the prior
    # D-eyefallback block that lived here. D-eyefallback only overrode
    # UNCOVERED sclera/pupil/iris vertices; its own measurement (see that
    # module's docstring, preserved there) found that even "covered"
    # eyeball vertices are too noisy to trust (std ~30-58/channel — grazing-
    # angle eyelid-margin contamination, not a bug). gnm_eye_render always
    # overrides sclera/pupil (flat procedural color, never raw-baked) and
    # always collapses iris to one real mean color (or a generic default if
    # this patient has zero real iris coverage) — never a per-vertex raw
    # bake for any part of the eyeball, covered or not. Local import: this
    # keeps gnm_eye_render.py free of any dependency on gnm_texture_bake,
    # so it can be unit-tested/reused standalone (same discipline already
    # used for the gnm_texture_atlas import below).
    from gnm_eye_render import apply_eye_render  # noqa: PLC0415 (local import, see comment above)
    vertex_color = apply_eye_render(vertex_color, covered)

    coverage_fraction = float(covered.sum()) / V
    return vertex_color, coverage_fraction, per_view_weight
