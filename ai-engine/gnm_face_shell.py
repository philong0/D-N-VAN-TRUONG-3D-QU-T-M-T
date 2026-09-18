"""
Crisalix-compatible Face Shell Topology & Unified HD UV Texture Generator.

This module extracts the dedicated Face Shell (~10,390 vertices, ~20,410 triangles)
from GNM Head v3, covering:
  - Anterior Front Scalp (natural hairline dome)
  - Temples & Zygomatics (full face width)
  - Both Ears (full ear anatomy)
  - Eyes & Eye Sockets
  - Cheeks, Nose, Mouth, Chin
  - Submental / Upper Neck Transition

It establishes a canonical, continuous UV parameterization and bakes all 4 multi-view
photos (0°, 45°, 90°, below-chin) directly into ONE unified 2K GPU texture (`face_HD.png`),
completely replacing per-vertex Gouraud colors and 19-patch PCA atlas approximations.
"""

from collections import deque
from pathlib import Path
import cv2
import numpy as np

from gnm_vertex_trust import compute_vertex_trust
from gnm_texture_bake import _smoothstep
import gnm_pyrender_occlusion as pyrender_occlusion

GNM_ASSET = Path(__file__).parent.parent / "public" / "models" / "gnm" / "gnm_head_v3.npz"

_shell_cache = None
_shell_eye_submasks_cache = None


def _get_shell_eye_submasks(shell_vids):
    """Phase 4 — real eyeball sub-group membership (`eyes`/`sclera`/`iris`/
    `pupil`/`eye_exteriors`) for THIS shell's own vertices, cached
    module-wide (same discipline as `_shell_cache` above) so
    `bake_unified_face_texture`'s draw-order fix doesn't re-read the 53MB
    GNM asset off disk on every call."""
    global _shell_eye_submasks_cache
    if _shell_eye_submasks_cache is None:
        gnm = np.load(GNM_ASSET, allow_pickle=True)
        gnames = [str(n) for n in gnm["vertex_group_names"]]
        vgroups = gnm["vertex_groups"]
        _shell_eye_submasks_cache = {
            name: (vgroups[gnames.index(name)] > 0.5)[shell_vids]
            for name in ("eyes", "scleras", "irises", "pupils", "eye_exteriors")
        }
    return _shell_eye_submasks_cache


def get_face_shell_topology():
    """
    Returns (shell_vids, triangles, uvs, template_positions):
      - shell_vids: (10390,) int array of vertex indices in GNM 17,821
      - triangles: (20410, 3) int32 triangle index array (0-indexed to shell_vids)
      - uvs: (10390, 2) float32 UV coordinates in [0, 1]
      - template_positions: (10390, 3) float32 template positions
    """
    global _shell_cache
    if _shell_cache is not None:
        return _shell_cache

    gnm = np.load(GNM_ASSET, allow_pickle=True)
    vgroups = gnm["vertex_groups"]
    gnames = [str(n) for n in gnm["vertex_group_names"]]
    all_triangles = gnm["triangles"]
    tpl_pos = gnm["template_vertex_positions"]

    skin_ext = vgroups[gnames.index("skin_exterior")] > 0.5
    hockey = vgroups[gnames.index("hockey_mask")] > 0.5
    eye_sockets = vgroups[gnames.index("eye_sockets")] > 0.5

    # Pure aesthetic anatomical face mask with continuous closed eye sockets:
    shell_mask = hockey | eye_sockets




    tri_inside = shell_mask[all_triangles[:, 0]] & shell_mask[all_triangles[:, 1]] & shell_mask[all_triangles[:, 2]]
    shell_triangles = all_triangles[tri_inside]
    all_tri_uvs = gnm["triangle_uvs"]
    shell_tri_uvs = all_tri_uvs[tri_inside]

    shell_vids_list = []
    shell_uvs_list = []
    vert_map = {}
    remapped_tris = []

    for t_idx in range(len(shell_triangles)):
        tri_remap = []
        for k in range(3):
            orig_vid = shell_triangles[t_idx, k]
            uv = shell_tri_uvs[t_idx, k]
            key = (orig_vid, round(float(uv[0]), 5), round(float(uv[1]), 5))
            if key not in vert_map:
                idx = len(shell_vids_list)
                vert_map[key] = idx
                shell_vids_list.append(orig_vid)
                shell_uvs_list.append(uv)
            tri_remap.append(vert_map[key])
        remapped_tris.append(tri_remap)

    shell_vids = np.array(shell_vids_list, dtype=np.int32)
    shell_uvs = np.array(shell_uvs_list, dtype=np.float32)
    remapped_triangles = np.array(remapped_tris, dtype=np.int32)
    shell_pos = tpl_pos[shell_vids]

    # Anatomical Nostril Capping: close open nostril loops with natural interior floors
    from collections import Counter
    import networkx as nx
    edge_counts = Counter([tuple(sorted(e)) for tri in remapped_triangles for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]])
    boundary_edges = [e for e, c in edge_counts.items() if c == 1]
    bg = nx.Graph()
    for u, v in boundary_edges:
        bg.add_edge(u, v)
    cycles = list(nx.cycle_basis(bg))

    extra_vids_list = []
    extra_uvs_list = []
    extra_tris_list = []
    
    for cyc in cycles:
        c_orig_vids = shell_vids[cyc]
        c_pos = tpl_pos[c_orig_vids]
        centroid = c_pos.mean(axis=0)
        # Nostril opening loop (y in [0.24, 0.28], |x| < 0.02, 10-35 vertices)
        if 0.24 <= centroid[1] <= 0.28 and abs(centroid[0]) < 0.02 and len(cyc) < 36:
            sub_g = bg.subgraph(cyc)
            ordered_cyc = [cyc[0]]
            while len(ordered_cyc) < len(cyc):
                curr = ordered_cyc[-1]
                neighbors = [n for n in sub_g.neighbors(curr) if n not in ordered_cyc]
                if not neighbors:
                    break
                ordered_cyc.append(neighbors[0])
            center_vid_idx = len(shell_vids) + len(extra_vids_list)
            center_uv = shell_uvs[ordered_cyc].mean(axis=0)
            extra_vids_list.append(shell_vids[ordered_cyc[0]])
            extra_uvs_list.append(center_uv)
            N = len(ordered_cyc)
            for i in range(N):
                v0 = ordered_cyc[i]
                v1 = ordered_cyc[(i + 1) % N]
                extra_tris_list.append([v0, v1, center_vid_idx])

    if extra_vids_list:
        shell_vids = np.concatenate([shell_vids, np.array(extra_vids_list, dtype=np.int32)])
        shell_uvs = np.concatenate([shell_uvs, np.array(extra_uvs_list, dtype=np.float32)])
        remapped_triangles = np.vstack([remapped_triangles, np.array(extra_tris_list, dtype=np.int32)])
        shell_pos = tpl_pos[shell_vids]

    _shell_cache = (shell_vids, remapped_triangles, shell_uvs, shell_pos.astype(np.float32))
    return _shell_cache


def multiband_blend_views(view_imgs: list[np.ndarray], view_masks: list[np.ndarray], levels: int = 4) -> np.ndarray:
    """Blends multiple UV-projected view images using Laplacian pyramid multi-band blending."""
    n_views = len(view_imgs)
    if n_views == 1:
        return view_imgs[0]

    pyr_imgs = []
    pyr_masks = []
    for k in range(n_views):
        g_img = [view_imgs[k].astype(np.float32)]
        g_mask = [view_masks[k].astype(np.float32)]
        for _ in range(levels):
            g_img.append(cv2.pyrDown(g_img[-1]))
            g_mask.append(cv2.pyrDown(g_mask[-1]))
        pyr_imgs.append(g_img)
        pyr_masks.append(g_mask)

    pyr_laps = []
    for k in range(n_views):
        l_img = []
        for l in range(levels):
            h, w = pyr_imgs[k][l].shape[:2]
            up = cv2.pyrUp(pyr_imgs[k][l + 1], dstsize=(w, h))
            lap = pyr_imgs[k][l] - up
            l_img.append(lap)
        l_img.append(pyr_imgs[k][levels])
        pyr_laps.append(l_img)

    # Detail levels (0..levels-1): winner-take-all per pixel (the view with
    # the highest mask weight at THIS level's own resolution), not a
    # weighted sum. Averaging two views' own high-frequency detail (pores,
    # eyebrow hairs, skin micro-texture) — two REAL, independently-exposed
    # photos of the same surface, never pixel-perfectly aligned — produces a
    # soft double-exposure/ghost wherever two views' weights are close, even
    # when both are individually valid samples (a blending-choice artifact,
    # distinct from the occlusion fix above). Argmax keeps every output
    # detail pixel a single real photo's own unmodified value.
    blended_pyr = []
    for l in range(levels):
        # Soft multi-band blending across Laplacian pyramid bands:
        # Avoids hard argmax seams between views that cause rectangular patches on cheeks.
        m_stack = np.stack([pyr_masks[k][l] for k in range(n_views)], axis=0) # (K, H, W)
        m_sum = np.sum(m_stack, axis=0, keepdims=True)
        m_norm = np.where(m_sum > 1e-6, m_stack / np.clip(m_sum, 1e-6, None), 1.0 / n_views)
        l_blend = np.sum(np.stack([pyr_laps[k][l] for k in range(n_views)], axis=0) * m_norm[:, :, :, None], axis=0)
        blended_pyr.append(l_blend)

    mask_sum_base_raw = sum(pyr_masks[k][levels] for k in range(n_views))
    mask_sum_base = np.clip(mask_sum_base_raw, 1e-6, None)
    base_blend = np.zeros_like(pyr_laps[0][levels])
    for k in range(n_views):
        norm_w_base = (pyr_masks[k][levels] / mask_sum_base)[:, :, None]
        base_blend += pyr_laps[k][levels] * norm_w_base

    # Task 6 audit fix (2026-08-27) -- proven root cause (real trace on
    # patient 257d9bfe: left_ear/right_ear both measured
    # mask_sum_base_raw == 0.0 EXACTLY at this base level, vs
    # right_cheek's 0.00302743). When no view's weight survives even a
    # full 4-level Gaussian pyramid blur, norm_w_base above is a
    # degenerate 0-numerator-over-epsilon-floor for EVERY view, so each
    # view's own already-valid pyr_imgs[k][levels] fallback mean color is
    # multiplied by 0 and discarded, yielding a literal [0,0,0] instead
    # of any real color. Fix scoped ONLY to pixels where
    # mask_sum_base_raw is at/under the same epsilon already used to
    # floor the denominator above -- every other pixel's `base_blend`
    # (computed above) is untouched, same weighted formula, same values.
    MASK_SUM_BASE_EPSILON = 1e-6
    zero_weight_base = mask_sum_base_raw <= MASK_SUM_BASE_EPSILON
    if zero_weight_base.any():
        unweighted_mean_base = sum(pyr_imgs[k][levels] for k in range(n_views)) / n_views
        base_blend = np.where(zero_weight_base[:, :, None], unweighted_mean_base, base_blend)

    blended_pyr.append(base_blend)

    recon = blended_pyr[levels]
    for l in range(levels - 1, -1, -1):
        h, w = blended_pyr[l].shape[:2]
        recon = cv2.pyrUp(recon, dstsize=(w, h)) + blended_pyr[l]

    return np.clip(recon, 0.0, 255.0).astype(np.float32)


def bake_unified_face_texture(fitted_positions_17821: np.ndarray, normals_17821: np.ndarray, views: list[dict], tex_size: int = 2048) -> tuple[np.ndarray, bytes, dict]:
    """Bakes clean, artifact-free 2K face texture without eye circles, cutouts, or background logo bleed."""
    photo_slots = [v.get("slot", f"view_{i}") for i, v in enumerate(views)]
    print(f"Baking from {len(views)} photos: {photo_slots}", flush=True)

    shell_vids, triangles, uvs, _ = get_face_shell_topology()
    shell_pos = fitted_positions_17821[shell_vids]
    shell_normals = normals_17821[shell_vids]
    shell_trust = compute_vertex_trust()[shell_vids]

    # 2026-09-07 audit note — first attempted this fix by targeting the
    # `scleras`/`irises`/`pupils` GNM vertex groups (`_get_shell_eye_submasks`,
    # matching what `gnm_eye_render.py` already does for the OTHER, separate
    # vertex-color bake pipeline). Verified directly against the real asset
    # before shipping it: THIS shell topology (`get_face_shell_topology`'s
    # own `shell_mask = valid_skin | hockey | ears | eye_sockets`, above)
    # includes `eye_sockets` (eyelid skin) but never `eyes`/`scleras`/
    # `irises`/`pupils` — measured 0 matching triangles for all three, so
    # that approach would have been a silent no-op on the exact mesh that
    # ships as baseline.glb (confirmed by `vertexCount` in baseline.json
    # matching `len(shell_vids)` exactly, i.e. this bake IS what patients
    # see, not a fallback path). There is no dedicated eyeball geometry in
    # this mesh at all — the visible eye region is ordinary skin-shell
    # triangles wrapped in real (ghosted, cross-view-blended) photo pixels.
    # Real fix below instead reuses `is_eye_zone` (this function's own
    # existing geometric eye-region box, already used a few lines down to
    # protect real eye pixels from the background-rejection filter) to force
    # single-view (no cross-blend) sampling specifically in that region —
    # targets the actual, real cause of the reported defect (this bake's own
    # multi-view ghosting, see the module-level "Direct Single-Layer
    # Projective Blending" step) using geometry proven to exist on this
    # mesh, instead of vertex groups that don't.

    # Real self-occlusion z-buffer — rejects a texel whose surface normal
    # faces the camera but is actually hidden behind another part of the
    # head (chin overhanging the neck, an ear fold, the far cheek in a
    # 45deg shot), something `facing_weight` alone (surface-normal
    # direction only) cannot tell apart from genuine visibility.
    #
    # Rendered by `gnm_pyrender_occlusion` — a real, perspective-correct
    # OpenGL-semantics rasterizer (pyrender, llvmpipe/EGL software backend
    # on this GPU-less host), replacing an earlier hand-rolled barycentric
    # z-buffer that was unit-test-free and, once actually measured, showed
    # both a false-occlusion bug (see that module's own docstring) and a
    # self-z-fighting stripe artifact from non-perspective-correct depth
    # interpolation. See gnm_pyrender_occlusion.py's own docstring for the
    # full measured history of both problems and
    # scratchpad/spike_pyrender_occlusion_unit.py for the ground-truth
    # (known wall + known occluder) test this rasterizer was validated
    # against before ever being run on real patient data.
    #
    # Built from the SHELL's own (shell_pos, triangles) — NOT the full
    # 17,821-vertex mesh — for the same reason: GNM's full template also
    # carries interior-mouth/eye geometry (`teeth`, `upper_teeth_and_gums`,
    # `lower_teeth_and_gums`, `tongue`, `gums`, `mouth_sock`, `eyes`,
    # `eye_interiors`) a real photo never shows but which would still
    # compete in a full-mesh z-buffer and falsely occlude the real lip
    # surface. A shell-vs-shell test is also the geometrically correct
    # scope: the only occlusion this bake needs to resolve is one EXTERIOR
    # part of the shell blocking another, never an interior cavity
    # structure the shell mesh doesn't even include.
    views_by_slot = {v.get("slot", f"view_{i}"): v for i, v in enumerate(views)}
    depth_buffers = pyrender_occlusion.render_shell_depth_buffers(shell_pos, triangles, views_by_slot)

    uv_px = np.column_stack([
        uvs[:, 0] * (tex_size - 1),
        (1.0 - uvs[:, 1]) * (tex_size - 1),
    ]).astype(np.float64)

    texel_pos = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
    texel_normal = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
    texel_trust = np.zeros((tex_size, tex_size), dtype=np.float64)
    texel_valid = np.zeros((tex_size, tex_size), dtype=bool)

    # 2026-08-27 -- UV collision tie-breaker (radius z-buffer). Root cause
    # (confirmed via 4 eliminated hypotheses + a direct texel_pos false-color
    # dump, this session): get_face_shell_topology's UV unwrap is only 2D
    # (theta = atan2(dx,dz) around the SAME vertical axis used below, plus
    # raw Y height) -- in the submental/chin-to-neck fold and the hairline
    # dome, the surface is not a single-valued graph over (theta,Y): two
    # anatomically different points (e.g. front-of-chin vs underside-of-chin)
    # can land on the same texel. Without a tie-breaker, whichever triangle
    # happens to rasterize LAST (array order, not geometry) wins, producing
    # the "vằn ngựa vằn" (zebra) striping. Fix: keep the texel whose
    # interpolated position has the LARGER radius from the SAME (cx=0.0,
    # cz=0.01) axis get_face_shell_topology's own theta unwrap uses (must
    # match that axis exactly, or this tie-break would disagree with the UV
    # layout it's resolving collisions for) -- the outer/exterior surface
    # (larger radius) wins over the folded-under interior surface, same
    # intent as a depth z-buffer but along the unwrap's own radial axis
    # rather than camera Z. Scoped ONLY to texels with a real collision (a
    # single-writer texel is never affected, since the comparison is a no-op
    # there) -- if this radius heuristic is ever wrong for some other
    # region's own geometry, it can only misassign an ALREADY-colliding
    # texel, never regress a texel that was unambiguous before.
    UV_AXIS_CX, UV_AXIS_CZ = 0.0, 0.01
    radius_buffer = np.full((tex_size, tex_size), -1.0, dtype=np.float64)

    # Standard continuous surface rasterization (no separate concentric eye tiers)
    for t_idx in range(len(triangles)):
        i0, i1, i2 = int(triangles[t_idx, 0]), int(triangles[t_idx, 1]), int(triangles[t_idx, 2])
        p0, p1, p2 = uv_px[i0], uv_px[i1], uv_px[i2]
        x_min = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        x_max = min(tex_size - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        y_min = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        y_max = min(tex_size - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
        if x_max < x_min or y_max < y_min:
            continue
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(denom) < 1e-9:
            continue

        xs = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
        ys = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((p1[1] - p2[1]) * (gx - p2[0]) + (p2[0] - p1[0]) * (gy - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (gx - p2[0]) + (p0[0] - p2[0]) * (gy - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue

        pos = w0[..., None] * shell_pos[i0] + w1[..., None] * shell_pos[i1] + w2[..., None] * shell_pos[i2]
        nrm = w0[..., None] * shell_normals[i0] + w1[..., None] * shell_normals[i1] + w2[..., None] * shell_normals[i2]
        nrm_len = np.linalg.norm(nrm, axis=-1, keepdims=True)
        nrm = nrm / np.clip(nrm_len, 1e-8, None)
        trust = w0 * shell_trust[i0] + w1 * shell_trust[i1] + w2 * shell_trust[i2]

        yy, xx = np.where(inside)
        gy_idx = y_min + yy
        gx_idx = x_min + xx

        pos_in = pos[yy, xx]
        current_radius = np.sqrt((pos_in[:, 0] - UV_AXIS_CX) ** 2 + (pos_in[:, 2] - UV_AXIS_CZ) ** 2)
        wins = current_radius > radius_buffer[gy_idx, gx_idx]
        if not wins.any():
            continue
        gy_idx, gx_idx = gy_idx[wins], gx_idx[wins]

        texel_pos[gy_idx, gx_idx] = pos_in[wins]
        texel_normal[gy_idx, gx_idx] = nrm[yy, xx][wins]
        texel_trust[gy_idx, gx_idx] = trust[yy, xx][wins]
        texel_valid[gy_idx, gx_idx] = True
        radius_buffer[gy_idx, gx_idx] = current_radius[wins]

    valid_lin_idx = np.where(texel_valid.reshape(-1))[0]
    pts_pos = texel_pos.reshape(-1, 3)[valid_lin_idx]
    pts_normal = texel_normal.reshape(-1, 3)[valid_lin_idx]
    n_pts = len(valid_lin_idx)
    n_views = len(views)

    frontal_idx = 0
    for k, v in enumerate(views):
        if v.get("slot") == "angle1":
            frontal_idx = k
            break

    pt_weights = np.zeros((n_views, n_pts), dtype=np.float32)
    pt_sampled_colors = np.zeros((n_views, n_pts, 3), dtype=np.float32)

    is_eye_zone = (np.abs(pts_pos[:, 0]) < 0.055) & (pts_pos[:, 1] > 0.270) & (pts_pos[:, 1] < 0.340)
    is_nostril_zone = (np.abs(pts_pos[:, 0]) < 0.028) & (pts_pos[:, 1] >= 0.220) & (pts_pos[:, 1] <= 0.275) & (pts_pos[:, 2] > 0.100)
    is_under_nose_tip = (np.abs(pts_pos[:, 0]) < 0.020) & (pts_pos[:, 1] >= 0.230) & (pts_pos[:, 1] <= 0.275) & (pts_pos[:, 2] > 0.100)
    is_ear_zone = (np.abs(pts_pos[:, 0]) > 0.050) & (pts_pos[:, 1] > 0.18) & (pts_pos[:, 1] < 0.32)
    # 2026-09-14 -- new "below" (chin-underside) checkpoint support: this
    # view's camera looks up from under the chin, so its own real photo
    # pixels are only geometrically valid for the underside/nostril patch
    # it actually saw -- everywhere else on the mesh, its pixels would be an
    # extreme, unusable grazing-angle sample. Reuses is_nostril_zone/
    # is_under_nose_tip above (already-defined zones, previously unused by
    # any weight branch) plus this new chin-underside box.
    is_chin_underside_zone = (np.abs(pts_pos[:, 0]) < 0.060) & (pts_pos[:, 1] >= 0.140) & (pts_pos[:, 1] <= 0.215)

    # Sample baseline warm skin tone directly from patient's detected face landmarks
    front_view = views[frontal_idx]
    front_img = front_view["image"]
    h_f, w_f = front_img.shape[:2]
    lms_f = front_view.get("landmarks_98")
    if lms_f is not None and len(lms_f) >= 60:
        lms_arr = np.asarray(lms_f, dtype=np.float64)
        nx, ny = int(np.clip(lms_arr[54, 0], 5, w_f - 6)), int(np.clip(lms_arr[54, 1], 5, h_f - 6))
        base_skin_bgr = front_img[ny - 4:ny + 5, nx - 4:nx + 5].mean(axis=(0, 1)).astype(np.float32)
    else:
        philtrum_pos = np.array([[0.0, 0.205, 0.095]])
        Xc_p = (front_view["R"] @ philtrum_pos.T).T + front_view["t"][None, :]
        K_f = front_view["camera_matrix"]
        px_p = int(np.clip(K_f[0, 0] * Xc_p[0, 0] / np.clip(Xc_p[0, 2], 1e-6, None) + K_f[0, 2], 0, w_f - 1))
        py_p = int(np.clip(K_f[1, 1] * Xc_p[0, 1] / np.clip(Xc_p[0, 2], 1e-6, None) + K_f[1, 2], 0, h_f - 1))
        base_skin_bgr = front_img[max(0, py_p - 10):py_p + 10, max(0, px_p - 10):px_p + 10].mean(axis=(0, 1)).astype(np.float32)


    for k, v in enumerate(views):
        R, t, K = v["R"], v["t"], v["camera_matrix"]
        img = v["image"]
        ih, iw = img.shape[:2]
        slot = v.get("slot", f"view_{k}")
        person_mask = v.get("person_mask")

        cam_pos = -R.T @ t
        ray = cam_pos[None, :] - pts_pos
        ray_dist = np.linalg.norm(ray, axis=1, keepdims=True)
        ray_dir = ray / np.clip(ray_dist, 1e-6, None)
        cos_angle = np.sum(pts_normal * ray_dir, axis=1)

        Xc = (R @ pts_pos.T).T + t[None, :]
        z = Xc[:, 2]
        fx, fy = K[0, 0], K[1, 1]
        cx_c, cy_c = K[0, 2], K[1, 2]
        px = fx * Xc[:, 0] / np.clip(z, 1e-6, None) + cx_c
        py = fy * Xc[:, 1] / np.clip(z, 1e-6, None) + cy_c

        is_frontal = (k == frontal_idx)
        in_bounds = (z > 0.05) & (px >= 2) & (px < iw - 3) & (py >= 2) & (py < ih - 3)

        view_yaw = abs(float(v.get("yaw", 0.0)))
        if is_frontal:
            # Frontal photo covers the entire facial shell with full natural fidelity
            min_cos = 0.01
            facing_weight = np.clip(cos_angle, 0.01, 1.0) ** 1.2
            valid = in_bounds & (cos_angle > min_cos)
        elif slot == "below":
            # Chin-underside / basal nostrils photo
            min_cos = 0.01
            facing_weight = np.clip(cos_angle, 0.01, 1.0) ** 1.2
            valid = in_bounds & (cos_angle > min_cos) & (is_chin_underside_zone | is_nostril_zone | is_under_nose_tip)
        else:
            # Oblique and profile views
            min_cos = 0.02
            facing_weight = np.clip(cos_angle, 0.02, 1.0) ** 1.2
            valid = in_bounds & (cos_angle > min_cos)

        # Depth-buffer occlusion check: only needed for angled views (frontal view sees entire anterior face)
        if slot in depth_buffers and not is_frontal:
            is_visible = pyrender_occlusion.visible(pts_pos, slot, depth_buffers, epsilon=0.012)
            valid &= is_visible

        # Smooth feathering from person mask edge (never a hard binary rectangular cut)
        mask_weight = np.ones(n_pts, dtype=np.float32)
        if person_mask is not None:
            mask_u8 = person_mask.astype(np.uint8) * 255
            dist_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
            py_int = np.clip(py.astype(int), 0, ih - 1)
            px_int = np.clip(px.astype(int), 0, iw - 1)
            dists = dist_map[py_int, px_int]
            mask_weight = np.clip(dists / 16.0, 0.0, 1.0)
            valid &= (dists > 1.0)

        img_f = img.astype(np.float32)
        pxc = np.clip(px, 0, iw - 1)
        pyc = np.clip(py, 0, ih - 1)
        x0 = np.floor(pxc).astype(int)
        x1 = np.clip(x0 + 1, 0, iw - 1)
        y0 = np.floor(pyc).astype(int)
        y1 = np.clip(y0 + 1, 0, ih - 1)
        wx = (pxc - x0)[:, None]
        wy = (pyc - y0)[:, None]
        c00 = img_f[y0, x0]
        c01 = img_f[y0, x1]
        c10 = img_f[y1, x0]
        c11 = img_f[y1, x1]
        sampled = (1.0 - wy) * ((1.0 - wx) * c00 + wx * c01) + wy * ((1.0 - wx) * c10 + wx * c11)

        # Background rejection: only reject near-pure white background pixels
        samp_u8 = np.clip(sampled, 0, 255).astype(np.uint8)
        lab_samp = cv2.cvtColor(samp_u8[None, :, :], cv2.COLOR_BGR2LAB)[0]
        is_samp_wall = (lab_samp[:, 0] > 252)
        valid &= (~is_samp_wall)

        pt_weights[k] = facing_weight * mask_weight * valid.astype(np.float32)
        pt_sampled_colors[k] = sampled


    # Exposure Gain Harmonization: match each view's overall brightness to the frontal photo on overlap
    for k in range(n_views):
        if k == frontal_idx:
            continue
        overlap = (pt_weights[frontal_idx] > 0.15) & (pt_weights[k] > 0.15)
        if overlap.sum() >= 50:
            f_mean = np.mean(pt_sampled_colors[frontal_idx][overlap], axis=0)
            v_mean = np.mean(pt_sampled_colors[k][overlap], axis=0)
            gain = np.clip((f_mean + 1.0) / (v_mean + 1.0), 0.75, 1.30)
            pt_sampled_colors[k] = np.clip(pt_sampled_colors[k] * gain[None, :], 0, 255)

    # Gaze & Eye Sharpness Guard: eyes smoothly prioritize frontal photo without sharp rectangular boundaries
    d_left = np.sqrt((pts_pos[:, 0] - 0.0308) ** 2 + (pts_pos[:, 1] - 0.3031) ** 2)
    d_right = np.sqrt((pts_pos[:, 0] - (-0.0309)) ** 2 + (pts_pos[:, 1] - 0.3031) ** 2)
    d_eye = np.minimum(d_left, d_right)
    eye_factor = _smoothstep(0.026, 0.016, d_eye)
    for k in range(n_views):
        if k != frontal_idx:
            pt_weights[k] *= (1.0 - eye_factor)

    # Natural cosine-facing weighting across all cameras with frontal anchor
    pt_weights[frontal_idx] *= 1.5

    # Normalize weights with smooth exponent for seamless transition
    total_w = np.sum(pt_weights, axis=0, keepdims=True)
    has_photo = total_w[0] > 1e-5

    # Direct clean photographic blending across visible views:
    norm_w = pt_weights / np.clip(total_w, 1e-6, None)
    blended_pts_color = np.sum(norm_w[:, :, None] * pt_sampled_colors, axis=0)

    # Render into 2K texture canvas
    tex_canvas = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    valid_mask = np.zeros((tex_size, tex_size), dtype=np.uint8)

    flat_tex = tex_canvas.reshape(-1, 3)
    flat_val = valid_mask.reshape(-1)

    valid_idx = np.where(has_photo)[0]
    flat_tex[valid_lin_idx[valid_idx]] = np.clip(blended_pts_color[valid_idx], 0, 255).astype(np.uint8)
    flat_val[valid_lin_idx[valid_idx]] = 255

    tex_canvas = flat_tex.reshape(tex_size, tex_size, 3)
    valid_mask = flat_val.reshape(tex_size, tex_size)

    # Seamless Natural Skin Boundary Diffusion (strictly isolated to small seam gaps, preventing hair bleed):
    shell_mask_u8 = texel_valid.astype(np.uint8) * 255
    unobserved = ((valid_mask == 0) & (shell_mask_u8 > 0)).astype(np.uint8) * 255

    if unobserved.any():
        # Telea inpainting with tight 3px radius to seal mesh seams without propagating dark hairline
        inpainted_base = cv2.inpaint(tex_canvas, unobserved, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        unfilled = (unobserved > 0) & ((inpainted_base.sum(axis=-1) == 0) | (valid_mask == 0))
        if unfilled.any():
            valid_texels = tex_canvas[valid_mask > 0]
            med_skin = np.median(valid_texels, axis=0) if len(valid_texels) > 0 else np.clip(base_skin_bgr, 0, 255)
            inpainted_base[unfilled] = med_skin.astype(np.uint8)
        final_texture = inpainted_base.copy()
        final_texture[valid_mask > 0] = tex_canvas[valid_mask > 0]
    else:
        final_texture = tex_canvas.copy()

    # Mild edge dilation for UV boundary seam elimination
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(shell_mask_u8, kernel, iterations=4)
    edge_padding = ((dilated_mask > 0) & (shell_mask_u8 == 0)).astype(np.uint8) * 255
    if edge_padding.any():
        final_texture = cv2.inpaint(final_texture, edge_padding, inpaintRadius=8, flags=cv2.INPAINT_TELEA)

    ok, png_bytes = cv2.imencode(".png", final_texture, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    print(f"Face Shell unified HD texture ready ({len(png_bytes)/1024/1024:.2f} MB, seamless photorealistic).", flush=True)

    coverage_stats = {
        "shellTexelCount": int(texel_valid.sum()),
        "coveredTexelCount": int(valid_mask.sum() / 255),
        "viewsUsed": [views[k].get("slot", f"view_{k}") for k in range(n_views)],
        "method": "Direct Single-Layer Projective Blending + Multi-Scale Skin Diffusion",
    }
    debug_maps = {"stats": coverage_stats}
    return final_texture, png_bytes.tobytes(), debug_maps


if __name__ == "__main__":
    import argparse
    import os
    from detect_pose import detect_pose
    from gnm_vertex_trust import compute_vertex_trust, blend_positions_by_trust

    parser.add_argument("--patient", type=str, required=True, help="Patient ID")
    parser.add_argument("--photos", type=str, default="angle1.png,angle2.png,angle3.png,angle4.png", help="Comma-separated photo list")
    parser.add_argument("--single-source", action="store_true", default=True, help="Use single-source anchor for central face")
    parser.add_argument("--multiband", action="store_true", default=True, help="Use Laplacian multi-band blending")
    parser.add_argument("--no-blur", action="store_true", default=True, help="Do not blur eyes")
    parser.add_argument("--rebuild-geo", action="store_true", default=False, help="Rebuild geometry")
    parser.add_argument("--force", action="store_true", default=False, help="Force overwrite")
    parser.add_argument("--yes", action="store_true", default=False, help="Auto accept")
    args, _ = parser.parse_known_args()

    repo_root = Path(__file__).parent.parent
    photos_dir = repo_root / ".data" / "patients" / args.patient / "photos"
    reconstruction_dir = repo_root / "public" / "models" / "patients" / args.patient / "reconstruction"

    photo_names = [p.strip() for p in args.photos.split(",")]
    images = {}
    for p in photo_names:
        slot = p.replace(".png", "").replace(".jpg", "")
        img_path = photos_dir / p
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                images[slot] = img
                print(f"Loaded {slot}: {img_path} ({img.shape[1]}x{img.shape[0]})")

    if not images:
        raise RuntimeError(f"No photos found for patient {args.patient} in {photos_dir}")

    # Full reconstruction pipeline
    view_inputs, view_slots, view_images, view_landmarks = [], [], [], []
    for slot, image_bgr in images.items():
        is_profile = slot == "angle3"
        h, w = image_bgr.shape[:2]
        res = detect_pose(image_bgr, is_profile_view=is_profile)
        if not res["has_face"]:
            print(f"Warning: {slot} no face detected")
            continue
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        view_inputs.append(ViewInput(res["landmarks_98"], K, is_profile, (h, w)))
        view_slots.append(slot)
        view_images.append(image_bgr)
        view_landmarks.append(res["landmarks_98"])

    fitted_positions, per_view_pose, _ = fit_multiview(view_inputs)
    fitted_positions, _ = enrich_with_dense_nose(fitted_positions, view_inputs, dict(zip(view_slots, view_images)))
    fitted_positions, _ = apply_width_correction(fitted_positions, view_inputs, view_slots)

    trust = compute_vertex_trust()
    blended_positions = blend_positions_by_trust(fitted_positions, get_template_positions(), trust)
    triangles_full = get_triangles()
    normals = compute_vertex_normals(blended_positions, triangles_full)

    view_landmarks_by_slot = dict(zip(view_slots, view_landmarks))
    bake_views = []
    for slot, image_bgr, pose in zip(view_slots, view_images, per_view_pose):
        if pose is None:
            continue
        R, t, _ = pose
        h, w = image_bgr.shape[:2]
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        p_mask = compute_person_silhouette_mask(view_landmarks_by_slot[slot], (h, w), slot=slot)
        bake_views.append({"image": image_bgr, "R": R, "t": t, "camera_matrix": K, "person_mask": p_mask, "slot": slot})

    print(f"\n--- Re-baking face_HD.png for patient {args.patient} with {len(bake_views)} views ---")
    final_tex, png_bytes, debug_maps = bake_unified_face_texture(blended_positions, normals, bake_views, tex_size=2048)

    data_recon_dir = repo_root / ".data" / "patients" / args.patient / "reconstruction"
    os.makedirs(reconstruction_dir, exist_ok=True)
    os.makedirs(data_recon_dir, exist_ok=True)

    shell_vids, triangles, uvs, _ = get_face_shell_topology()
    import time
    shell_ver = str(int(time.time() * 1000))
    topology_dict = {
        "shellVertexIndices": shell_vids.tolist(),
        "triangles": triangles.tolist(),
        "uvs": uvs.tolist(),
    }

    # Task 2 (2026-08-27 data-flow audit) -- desync guard, real bug proven on
    # patient 257d9bfe: this CLI intentionally runs its OWN independent
    # fit_multiview() (see the "nose x10 -> x30" experiment comments in
    # gnm_correspondence.py, made the same day this CLI was used to test
    # them) -- it must keep doing that, this tool's whole purpose is testing
    # identity-fit changes in isolation, so reusing an old positions.f32
    # would break that. But main.py's /reconstruct endpoint ALSO writes
    # positions.f32/colors.f32/atlas_contract.json into this exact same
    # target_dir from ONE shared fit (verified: /reconstruct calls
    # fit_multiview() exactly once, main.py:258) -- if this CLI's fresh fit
    # differs even slightly from whatever fit colors.f32/atlas were last
    # baked from, positions.f32 (about to be overwritten below) and
    # colors.f32/atlas (untouched by this CLI) silently stop matching, with
    # nothing recording that fact. Currently harmless (Canvas3D.tsx:1959
    # always prefers the Face Shell mesh this CLI's own output feeds, never
    # the vertex-color/atlas mesh, while a Face Shell is present) but a real
    # latent bug for the fallback path if Face Shell ever fails to decode.
    # Fix (minimal, isolated to this standalone CLI -- main.py's own
    # /reconstruct is untouched, already correct per Task 1): only when the
    # position data this run is about to write actually differs from
    # whatever is already on disk, delete the now-provably-stale
    # colors.f32/atlas_contract.json/atlas/ in THIS SAME target_dir instead
    # of leaving them silently mismatched. This never fabricates new
    # colors/atlas data (no ICM/atlas logic duplicated here) -- it only
    # removes a false claim of consistency. The next real page load's
    # `readPersistedReconstruction` (src/lib/gnm/reconstruction-service.ts)
    # will then fail its colors.f32 read and fall through to a real,
    # fully-in-sync /reconstruct call, exactly the self-healing path that
    # already exists for "no reconstruction yet".
    new_positions_f32 = blended_positions.astype(np.float32)
    for target_dir in (reconstruction_dir, data_recon_dir):
        old_positions_path = target_dir / "positions.f32"
        colors_path = target_dir / "colors.f32"
        if old_positions_path.exists() and colors_path.exists():
            old_positions = np.fromfile(str(old_positions_path), dtype="<f4")
            positions_changed = (
                old_positions.shape != new_positions_f32.reshape(-1).shape
                or not np.array_equal(old_positions, new_positions_f32.reshape(-1))
            )
            if positions_changed:
                stale_files = [colors_path, target_dir / "atlas_contract.json"]
                removed = []
                for stale in stale_files:
                    if stale.exists():
                        stale.unlink()
                        removed.append(stale.name)
                atlas_dir = target_dir / "atlas"
                if atlas_dir.is_dir():
                    import shutil
                    shutil.rmtree(atlas_dir)
                    removed.append("atlas/")
                if removed:
                    print(f"Task 2 desync guard: positions.f32 about to change in {target_dir} -- removed now-stale {removed} (Pipeline A output computed from the OLD fit); next real /reconstruct will regenerate them in sync.", flush=True)

    for target_dir in (reconstruction_dir, data_recon_dir):
        with open(target_dir / "face_HD.png", "wb") as f:
            f.write(png_bytes)
        # B4 audit (2026-08-27) -- additive diagnostic maps, never read back by
        # any renderer/loader; see debug_maps' own construction in
        # bake_unified_face_texture for what each one encodes.
        with open(target_dir / "coverage.png", "wb") as f:
            f.write(debug_maps["coverage_png"])
        with open(target_dir / "source_view.png", "wb") as f:
            f.write(debug_maps["source_view_png"])
        with open(target_dir / "confidence.png", "wb") as f:
            f.write(debug_maps["confidence_png"])
        with open(target_dir / "coverage_stats.json", "w") as f:
            json.dump(debug_maps["stats"], f, indent=2)
        blended_positions.astype(np.float32).tofile(str(target_dir / "positions.f32"))
        normals.astype(np.float32).tofile(str(target_dir / "normals.f32"))
        uvs.astype(np.float32).tofile(str(target_dir / "uvs.f32"))
        triangles.astype(np.int32).tofile(str(target_dir / "indices.f32"))
        with open(target_dir / "shell_topology.json", "w") as f:
            json.dump(topology_dict, f)

        meta_path = target_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as mf:
                    meta_data = json.load(mf)
                meta_data["faceShell"] = {
                    "method": "GNM Face Shell v1 (Z-buffer Occlusion Tested)",
                    "shellVertexCount": int(len(shell_vids)),
                    "shellTriangleCount": int(len(triangles)),
                    "textureUrl": f"/models/patients/{args.patient}/reconstruction/face_HD.png?v={shell_ver}",
                    "shellTopologyUrl": f"/models/patients/{args.patient}/reconstruction/shell_topology.json?v={shell_ver}",
                    "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with open(meta_path, "w") as mf:
                    json.dump(meta_data, mf, indent=2)
            except Exception as e:
                print(f"Warning: could not update metadata.json: {e}")

    print(f"Successfully written face_HD.png ({len(png_bytes)/1024:.1f} KB), topology & geometry to {reconstruction_dir} and {data_recon_dir}")
