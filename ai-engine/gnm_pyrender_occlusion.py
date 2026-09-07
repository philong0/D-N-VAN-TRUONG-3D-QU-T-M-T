"""
Real, GPU-grade (here: llvmpipe software, no GPU on this host — see the
compatibility spike this module's tests were validated against) self-
occlusion z-buffer for the Face Shell HD texture bake, replacing the
hand-rolled barycentric rasterizer `bake_unified_face_texture` used to
build its own visibility test with.

Why this module exists (measured, not assumed):
  1. `bake_unified_face_texture` originally had NO occlusion test at all —
     only a surface-normal "facing" weight, which cannot tell "facing the
     camera" apart from "facing the camera but hidden behind the chin/ear/
     far cheek". This produced the misplaced chin/neck texture patch and
     the doubled profile edge (see the 2026-08-26 diagnostic session).
  2. A first attempt reused this project's existing D8 z-buffer
     (`gnm_texture_bake.py`/`gnm_texture_atlas.py`, hand-rolled barycentric
     rasterizer) against the FULL 17,821-vertex mesh. Measured result: lip
     visibility from angle1 collapsed to ~0% — GNM's full template also
     carries interior-mouth/eye geometry (teeth, tongue, gums, eyeballs) a
     real photo never shows, which still projected nearer than the real lip
     surface at the lips' own screen pixels in that hand-rolled buffer.
  3. Scoping the SAME hand-rolled rasterizer to shell-only geometry fixed
     (1) (lip visibility 87.6%-97.6%) but left a visible self-z-fighting
     stripe pattern on the mouth/chin — the barycentric interpolation used
     there interpolates camera-space Z directly in screen space, which is
     NOT perspective-correct (a real rasterizer interpolates 1/z), and its
     own resolution/epsilon needed hand-tuning with no principled way to
     verify "correct enough".
  4. This module replaces that hand-rolled buffer with pyrender (a real,
     perspective-correct OpenGL-semantics rasterizer, here running on
     Mesa's llvmpipe/kms_swrast software backend since this host has no
     GPU — verified working via `PYOPENGL_PLATFORM=egl`, `osmesa` platform
     was tried first and found to hard-fail on this PyOpenGL/Mesa
     combination: `OSMesaCreateContextAttribs` import error), rendered at
     each view's own NATIVE photo resolution (not a downscaled cap), and
     scoped to shell-only geometry (fix from (3), kept). Validated against
     hand-computed ground truth (a flat wall + a known occluder — see
     scratchpad/spike_pyrender_occlusion_unit.py) before ever being run
     against real patient data.
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh

# Perpendicular-distance tolerance (meters) for "this point IS the surface
# at its own pixel, not something floating slightly in front/behind it from
# rasterization/interpolation noise". Same order of magnitude as this
# project's own D8 OCCLUSION_EPSILON (2.5mm) but not the same constant —
# this is a real, perspective-correct GPU-semantics buffer at native photo
# resolution, a materially different precision regime, re-measured on its
# own merits rather than inherited from the old hand-rolled buffer's tuning.
OCCLUSION_EPSILON_M = 0.003


def cv_camera_pose_to_gl(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """OpenCV-style extrinsics (world->camera, X right / Y down / Z forward
    — the convention every camera_matrix/R/t triple in this codebase's
    detect_pose.py/gnm_identity_fit.py already uses) to a pyrender camera
    pose (camera->world, OpenGL convention: X right / Y up / Z backward).
    Verified empirically against known wall/occluder placements — see this
    module's own docstring point 4 and scratchpad/spike_pyrender_occlusion_unit.py."""
    world_from_cam_cv = np.eye(4)
    world_from_cam_cv[:3, :3] = R.T
    world_from_cam_cv[:3, 3] = -R.T @ t
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    return world_from_cam_cv @ cv_to_gl


def render_shell_depth_buffers(shell_pos: np.ndarray, triangles: np.ndarray, views_by_slot: dict) -> dict:
    """Renders the shell mesh's own real depth buffer once per view, at that
    view's own native photo resolution. `views_by_slot`: {slot: {"image",
    "R", "t", "camera_matrix"}}. Returns {slot: {"depth": (h,w) float32,
    camera-space Z in meters, 0.0 = nothing rendered at that pixel}}.

    One mesh/scene is built per call (not cached across views) — at ~0.6s
    per view for this project's real photo resolutions (measured), this is
    a small fraction of the several-minute full bake and not worth the
    complexity of a persistent render context for now."""
    shell_trimesh = trimesh.Trimesh(vertices=shell_pos, faces=triangles, process=False)
    shell_mesh = pyrender.Mesh.from_trimesh(shell_trimesh, smooth=False)

    depth_buffers = {}
    for slot, v in views_by_slot.items():
        h, w = v["image"].shape[:2]
        R, t, K = v["R"], v["t"], v["camera_matrix"]

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
        scene.add(shell_mesh)
        cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=5.0)
        pose = cv_camera_pose_to_gl(R, t)
        scene.add(cam, pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=1.0), pose=pose)

        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        # NOTE: RenderFlags.DEPTH_ONLY returns an all-zero buffer on this
        # pyrender/EGL(kms_swrast) backend — a reproducible bug in this
        # specific backend combination, verified by brute-forcing camera
        # rotation signs against known wall placements (both this module's
        # own math AND the DEPTH_ONLY flag were tested independently before
        # concluding it was the flag, not the camera math). Rendering
        # color+depth and discarding color is the confirmed-working path.
        _color, depth = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        renderer.delete()

        depth_buffers[slot] = {"depth": depth, "R": R, "t": t, "camera_matrix": K, "w": w, "h": h}

    return depth_buffers


def visible(pts_pos: np.ndarray, slot: str, depth_buffers: dict, epsilon: float = OCCLUSION_EPSILON_M) -> np.ndarray:
    """True where `pts_pos[i]` is the (approximately) nearest shell surface
    at its own projected pixel in `depth_buffers[slot]` — i.e. not hidden
    behind another part of the shell (chin over the neck, an ear fold, the
    far cheek). False for out-of-frame/behind-camera points too."""
    entry = depth_buffers[slot]
    R, t, K, w, h, depth = entry["R"], entry["t"], entry["camera_matrix"], entry["w"], entry["h"], entry["depth"]

    Xc = (R @ pts_pos.T).T + t[None, :]
    z = Xc[:, 2]
    u = K[0, 0] * Xc[:, 0] / np.clip(z, 1e-9, None) + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / np.clip(z, 1e-9, None) + K[1, 2]

    in_bounds = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uu = np.clip(u.astype(int), 0, w - 1)
    vv = np.clip(v.astype(int), 0, h - 1)
    nearest = depth[vv, uu]

    return in_bounds & (nearest > 0) & (z <= nearest + epsilon)
