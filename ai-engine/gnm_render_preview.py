"""
Software (pure numpy, no GPU/OpenGL) Gouraud-shaded, z-buffered preview
renderer for a GNM Head v3 mesh. Extracted from the now-removed
`gnm_v2_experimental.py` (Phase 5 cleanup — that module's own V1-vs-V2
landmark-source comparison was confirmed dead/unreachable from any real UI
flow, but this specific function is a genuinely reusable diagnostic utility,
now depended on by ai-engine/regression/'s own capture scripts for visual
before/after regression checks) — same barycentric-fill algorithm as
gnm_texture_bake.py's own `_rasterize_min_depth`, extended to interpolate
real baked vertex color per pixel. Downscaled (`PREVIEW_MAX_DIM`) to keep
runtime reasonable for a pure-Python/numpy triangle loop over 35,324
triangles — a diagnostic/regression tool, not a production render path
(the real product renderer is Three.js, client-side).
"""
import numpy as np

import gnm_texture_bake as gtb

PREVIEW_MAX_DIM = 420


def rasterize_preview(positions, triangles, vertex_colors, R, t, camera_matrix, image_shape):
    ih_full, iw_full = image_shape
    scale = PREVIEW_MAX_DIM / max(ih_full, iw_full)
    iw, ih = max(int(round(iw_full * scale)), 1), max(int(round(ih_full * scale)), 1)
    cm = camera_matrix.copy()
    cm[0, 0] *= scale
    cm[1, 1] *= scale
    cm[0, 2] *= scale
    cm[1, 2] *= scale

    u, v, z = gtb.project(positions, R, t, cm)
    depth_buffer = np.full((ih, iw), np.inf)
    color_buffer = np.zeros((ih, iw, 3), dtype=np.float64)

    tri_u, tri_v, tri_z = u[triangles], v[triangles], z[triangles]
    tri_color = vertex_colors[triangles]
    valid_tri = (tri_z > 0).all(axis=1)

    for idx in np.nonzero(valid_tri)[0]:
        pu, pv, pz = tri_u[idx], tri_v[idx], tri_z[idx]
        pc = tri_color[idx]
        x_min, x_max = max(int(np.floor(pu.min())), 0), min(int(np.ceil(pu.max())), iw - 1)
        y_min, y_max = max(int(np.floor(pv.min())), 0), min(int(np.ceil(pv.max())), ih - 1)
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
        region_d = depth_buffer[y_min:y_max + 1, x_min:x_max + 1]
        update = inside & (interp_z < region_d)
        if not update.any():
            continue
        region_d[update] = interp_z[update]
        interp_c = w0[..., None] * pc[0] + w1[..., None] * pc[1] + w2[..., None] * pc[2]
        region_c = color_buffer[y_min:y_max + 1, x_min:x_max + 1]
        region_c[update] = interp_c[update]

    return np.clip(color_buffer, 0, 255).astype(np.uint8)
