"""CPU z-buffer, reprojection overlays and actual textured GLB renders."""
from pathlib import Path
import cv2
import numpy as np
import trimesh


def project(vertices, pose, K):
    xyz = vertices @ pose[:3, :3].T + pose[:3, 3]
    pixels = xyz @ K.T
    return pixels[:, :2] / np.maximum(pixels[:, 2:], 1e-8), xyz[:, 2]


def rasterize(vertices, faces, pose, K, shape, uv=None, texture=None):
    h, w = shape
    xy, z = project(vertices, pose, K)
    depth = np.full((h, w), np.inf, np.float32)
    rgb = np.full((h, w, 3), 230, np.uint8)
    for ids in faces:
        if np.min(z[ids]) <= .02:
            continue
        pts = xy[ids]
        lo = np.maximum(np.floor(pts.min(axis=0)).astype(int), [0, 0])
        hi = np.minimum(np.ceil(pts.max(axis=0)).astype(int), [w-1, h-1])
        if np.any(hi < lo):
            continue
        a, b, c = pts
        den = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if abs(den) < 1e-8:
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0]+1)+.5, np.arange(lo[1], hi[1]+1)+.5)
        b1 = ((xx-a[0])*(c[1]-a[1]) - (yy-a[1])*(c[0]-a[0])) / den
        b2 = ((b[0]-a[0])*(yy-a[1]) - (b[1]-a[1])*(xx-a[0])) / den
        bary = np.stack([1-b1-b2, b1, b2], axis=-1)
        reciprocal = bary / z[ids]
        zz = 1 / np.maximum(reciprocal.sum(axis=-1), 1e-8)
        patch = depth[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        take = (bary.min(axis=-1) >= -1e-5) & (zz < patch)
        patch[take] = zz[take]
        if uv is not None and texture is not None:
            texuv = (reciprocal @ uv[ids]) * zz[..., None]
            th, tw = texture.shape[:2]
            tx = np.clip(np.rint(texuv[..., 0]*(tw-1)).astype(int), 0, tw-1)
            ty = np.clip(np.rint((1-texuv[..., 1])*(th-1)).astype(int), 0, th-1)
            rgb[lo[1]:hi[1]+1, lo[0]:hi[0]+1][take] = texture[ty[take], tx[take]]
    return depth, rgb


def write_diagnostics(output, frames, vertices, faces, glb_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    glb = trimesh.load(glb_path, force='mesh', process=False)
    texture = np.asarray(glb.visual.material.baseColorTexture.convert('RGB'))[:, :, ::-1]
    for frame in frames:
        img = frame['image_bgr']
        scale = min(1., 640 / img.shape[0])
        shape = (round(img.shape[0]*scale), round(img.shape[1]*scale))
        K = frame['intrinsics'].copy()
        K[:2] *= scale
        overlay = cv2.resize(img, (shape[1], shape[0]))
        for verts, color in ((frame['arface_vertices'], (0, 255, 0)), (vertices, (0, 80, 255))):
            xy, z = project(verts, frame['camera_pose'], K)
            for pt in xy[(z > .02) & np.isfinite(xy).all(axis=1)][::4]:
                if 0 <= pt[0] < shape[1] and 0 <= pt[1] < shape[0]:
                    cv2.circle(overlay, tuple(np.rint(pt).astype(int)), 1, color, -1)
        _, rendered = rasterize(glb.vertices, glb.faces, frame['camera_pose'], K, shape,
                                glb.visual.uv, texture)
        cv2.imwrite(str(output / (frame['stem'] + '_reprojection.jpg')), overlay)
        cv2.imwrite(str(output / (frame['stem'] + '_render.jpg')), rendered)
