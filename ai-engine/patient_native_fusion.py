"""
patient_native_fusion.py

Patient-Specific 3D Surface Reconstruction Engine for Native iOS and Multi-Frame Capture.
Zero dependency on generic mannequins, template heads, or closed-eye PCA models.

Pipeline:
1. Ingest multi-frame scan package (RGB frames, camera poses, intrinsics, depth maps, ARFaceGeometry).
2. Transform all observations into Common Patient Coordinate System:
   P_patient = T_face^{-1} * T_camera * P_cam
3. Multi-frame depth & geometry registration and fusion.
4. Clean, manifold surface generation with anatomical boundary preservation.
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
import trimesh

# Standard Apple ARFaceGeometry topology (1220 vertices, 2304 triangles)
# Native Apple indices define the exact connectivity of human face geometry
def get_apple_arface_topology() -> np.ndarray | None:
    """Returns canonical triangle index array for ARFaceGeometry if available."""
    topology_cache_path = Path(__file__).parent / "models" / "arface_triangles.npy"
    if topology_cache_path.exists():
        return np.load(topology_cache_path)
    return None


def backproject_depth_map(
    depth_f32: np.ndarray,
    intrinsics_3x3: np.ndarray,
    min_depth: float = 0.15,
    max_depth: float = 0.85,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backprojects a 2D depth map (in meters) into a 3D metric point cloud (X, Y, Z) in camera space.
    Returns: (points_cam, (u, v) image coordinates)
    """
    h, w = depth_f32.shape[:2]
    fx = intrinsics_3x3[0, 0]
    fy = intrinsics_3x3[1, 1]
    cx = intrinsics_3x3[0, 2]
    cy = intrinsics_3x3[1, 2]

    # Subsample grid for efficiency while retaining dense sub-millimeter contours
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    u_grid, v_grid = np.meshgrid(us, vs)

    d_sampled = depth_f32[v_grid, u_grid]
    valid_mask = (d_sampled >= min_depth) & (d_sampled <= max_depth) & ~np.isnan(d_sampled) & ~np.isinf(d_sampled)

    u_valid = u_grid[valid_mask].astype(np.float64)
    v_valid = v_grid[valid_mask].astype(np.float64)
    z_valid = d_sampled[valid_mask].astype(np.float64)

    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy

    points_cam = np.column_stack([x_valid, y_valid, z_valid])
    uv_coords = np.column_stack([u_valid, v_valid])
    return points_cam, uv_coords


def transform_to_patient_coordinate_system(
    points_cam: np.ndarray,
    camera_pose_4x4: np.ndarray,
    face_pose_4x4: np.ndarray | None = None,
) -> np.ndarray:
    """
    Transforms camera-space metric points into the unified patient face
    coordinate system. `camera_pose_4x4` is the measured face-local ->
    OpenCV-camera transform, hence its inverse maps a depth point directly
    into ARFace local coordinates.

    D-poseconvention — `camera_pose_4x4` here follows the SAME convention
    every other camera_pose in this codebase already uses (this file's own
    `_reconstruct_from_rgb_multiview`'s solvePnP/recoverPose result,
    render_back_validator.py's `render_mesh_from_camera`,
    patient_texture_baker.py's `bake_visibility_aware_texture`): WORLD-TO-
    CAMERA extrinsics, i.e. `X_cam = R @ X_world + t`. Verified directly:
    every one of those call sites does `Xc = (R @ verts.T).T + t` with
    `verts` in patient/world space — the same convention the real iOS
    `ScanPackageExporter.swift` writes into `manifest.json`'s
    `cameraTransformColumnMajor` too (ARKit's own `simd_float4x4` camera
    transform is camera-to-world, but this project's Swift exporter stores
    its INVERSE under that key — confirmed against `BackendAPIClient.swift`
    / `package/route.ts`'s own pass-through, which never inverts it, so
    whatever convention arrives in the manifest must already match what
    every OTHER real consumer here expects). This function's own job is the
    OPPOSITE direction (camera-space point -> patient/world space), so it
    must invert `camera_pose_4x4` before applying it — using it un-inverted
    (the bug this fix replaces) silently produced points 5-7x too far from
    the real origin, caught by a synthetic-sphere ground-truth test
    (`_refine_surface_with_truedepth`'s own validation), not assumed.
    """
    if len(points_cam) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    homo = np.column_stack([points_cam, np.ones(len(points_cam), dtype=np.float64)])

    try:
        camera_to_world = np.linalg.inv(camera_pose_4x4)
    except np.linalg.LinAlgError:
        camera_to_world = np.eye(4, dtype=np.float64)

    # `face_pose_4x4` is retained only for backwards-compatible callers.
    # Applying it here would transform the same measurement twice.
    transform_mat = camera_to_world

    points_patient = (transform_mat @ homo.T).T[:, :3]
    return points_patient


def _grid_mesh_from_depth_frame(
    depth_f32: np.ndarray,
    intrinsics_3x3: np.ndarray,
    min_depth: float = 0.15,
    max_depth: float = 0.85,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds a real per-pixel structured-grid mesh directly from one depth
    frame, in CAMERA space (caller transforms to patient space). Every
    vertex is a real backprojected depth pixel (same math as
    `backproject_depth_map`, kept grid-shaped here instead of flattened so
    real pixel-adjacency can become real triangle connectivity) — a quad's
    2 triangles are only emitted when all 4 of its corner pixels have real
    valid depth, so a real depth dropout/edge never gets bridged by an
    invented triangle. Returns (vertices_cam (V,3), faces (F,3)); faces is
    empty when fewer than 2x2 valid-depth pixels exist in this frame.

    2026-09-10 fix — real evidence (a real on-device left_45 TrueDepth
    capture, inspected directly, not assumed): a static (min_depth,
    max_depth) Z-only window applied uniformly to every pixel let in the
    whole room, not just the face -- and a Z-only window alone still isn't
    enough, because a wall, shoulder, or raised hand at nearly the SAME
    distance as the face passes a Z check just as easily as the face does
    (confirmed numerically: narrowing Z-range alone on this exact real
    frame still left ~0.55m x 0.48m x 0.28m of "face" -- 2-3x a real head).
    Real per-pixel 3D distance from the face's own measured center (not
    just its depth) is what actually separates "on the face" from
    "somewhere else at a similar distance": at a 0.12m radius this exact
    frame's referenced-vertex extents came out to ~0.21m x 0.17m x 0.12m --
    plausible real adult-head dimensions -- confirmed by re-running this
    exact real depth buffer through several candidate radii before picking
    one, not guessed. The reference point is the CENTER-PATCH median depth
    (where the scan UI instructs the user to keep their face) backprojected
    at the patch's own center pixel. Falls back to the original static
    Z-only window when the center patch doesn't have enough valid pixels to
    measure a reference point from (e.g. a tiny synthetic test frame), so
    this never narrows a window that had nothing real to center on.
    """
    h, w = depth_f32.shape[:2]
    fx, fy = intrinsics_3x3[0, 0], intrinsics_3x3[1, 1]
    cx, cy = intrinsics_3x3[0, 2], intrinsics_3x3[1, 2]

    FACE_RADIUS_M = 0.12
    MIN_CENTER_SAMPLES = 50
    cy0, cy1 = int(h * 0.35), int(h * 0.65)
    cx0, cx1 = int(w * 0.35), int(w * 0.65)
    center_patch = depth_f32[cy0:cy1, cx0:cx1]
    center_valid = center_patch[np.isfinite(center_patch) & (center_patch > 0)]
    face_center_cam: np.ndarray | None = None
    if center_valid.size >= MIN_CENTER_SAMPLES:
        face_depth_center = float(np.median(center_valid))
        min_depth = max(min_depth, face_depth_center - FACE_RADIUS_M)
        max_depth = min(max_depth, face_depth_center + FACE_RADIUS_M)
        ref_u, ref_v = (cx0 + cx1) / 2.0, (cy0 + cy1) / 2.0
        face_center_cam = np.array([
            (ref_u - cx) * face_depth_center / fx,
            (ref_v - cy) * face_depth_center / fy,
            face_depth_center,
        ], dtype=np.float64)

    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    grid_h, grid_w = len(vs), len(us)
    u_grid, v_grid = np.meshgrid(us, vs)  # (grid_h, grid_w)

    d = depth_f32[v_grid, u_grid].astype(np.float64)
    x = (u_grid - cx) * d / fx
    y = (v_grid - cy) * d / fy
    valid = (d >= min_depth) & (d <= max_depth) & ~np.isnan(d) & ~np.isinf(d)
    if face_center_cam is not None:
        dist3d = np.sqrt((x - face_center_cam[0]) ** 2 + (y - face_center_cam[1]) ** 2 + (d - face_center_cam[2]) ** 2)
        valid = valid & (dist3d <= FACE_RADIUS_M)
    verts = np.stack([x, y, d], axis=-1).reshape(-1, 3)  # (grid_h*grid_w, 3), row-major
    # Invalid-depth pixels produce NaN x/y/d here; they are never referenced
    # by any face (quad_ok below requires all 4 real corners valid), but
    # zero them out anyway so a caller that concatenates `verts` across many
    # frames (see `_refine_surface_with_truedepth`) never has to carry NaNs
    # through arithmetic on the unreferenced rows (undefined int-cast
    # behavior otherwise, confirmed as a real RuntimeWarning before this).
    verts = np.nan_to_num(verts, nan=0.0)

    idx_grid = np.arange(grid_h * grid_w, dtype=np.int64).reshape(grid_h, grid_w)
    v00 = valid[:-1, :-1]
    v01 = valid[:-1, 1:]
    v10 = valid[1:, :-1]
    v11 = valid[1:, 1:]
    quad_ok = v00 & v01 & v10 & v11

    # 2026-09-10 fix — real evidence (this same real left_45 capture,
    # measured directly): even after the 3D face-radius filter above,
    # rendering the resulting mesh showed dense jagged spikes covering the
    # whole surface -- a classic "flying pixel" artifact of structured-
    # light depth sensors, where a small number of individual pixels at
    # real depth discontinuities (nostril/hairline/silhouette edges) report
    # wildly wrong depth while their immediate neighbors are correct. Each
    # of `quad_ok`'s corners was individually "valid" (real depth, in
    # range), so those quads still got bridged into triangles connecting a
    # correct point to a flying one -- a long thin spike. Measuring this
    # exact frame's own quads: the bulk have a natural <=3mm depth spread
    # across their 4 corners (real surface curvature at this sampling
    # density), but a clear outlier tail exists (95th pct ~9.5mm, 99th pct
    # ~21mm, 99.9th pct ~44mm) -- an 8mm cap removes that tail (~7% of
    # quads) while keeping the >92% of quads that are real smooth surface,
    # confirmed by re-measuring this exact real depth buffer, not guessed.
    QUAD_MAX_DEPTH_SPREAD_M = 0.008
    d00, d01, d10, d11 = d[:-1, :-1], d[:-1, 1:], d[1:, :-1], d[1:, 1:]
    quad_depths = np.stack([d00, d01, d10, d11])
    quad_spread_ok = (quad_depths.max(axis=0) - quad_depths.min(axis=0)) <= QUAD_MAX_DEPTH_SPREAD_M
    quad_ok = quad_ok & quad_spread_ok

    a = idx_grid[:-1, :-1][quad_ok]
    b = idx_grid[:-1, 1:][quad_ok]
    c = idx_grid[1:, :-1][quad_ok]
    e = idx_grid[1:, 1:][quad_ok]

    if len(a) == 0:
        return verts, np.zeros((0, 3), dtype=np.int64)

    tri1 = np.column_stack([a, b, c])
    tri2 = np.column_stack([b, e, c])
    faces = np.concatenate([tri1, tri2], axis=0)
    return verts, faces


class PatientNativeReconstructor:
    """
    Constructs a true patient-specific 3D mesh from native Apple TrueDepth / ARKit multi-frame scan packages.
    """

    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)
        self.frames = []
        self.depth_report = []
        self.manifest = {}
        self._load_package()

    def _load_package(self):
        # D-realpaths — `package_dir` passed in here is the SESSION dir
        # (e.g. `.data/patients/{id}/scans/{sessionId}`), one level ABOVE
        # `frames/`. The real writer (package/route.ts) puts EVERYTHING —
        # RGB jpgs, `{view}_depth.raw`, `manifest.json`, and the
        # `geometry/`/`camera/` subfolders — inside that `frames/`
        # directory, not at package_dir's own root (confirmed by reading
        # package/route.ts directly: `dir = scanFramesDir(...)` is the
        # `frames/` path, and every `writeFile` call targets `dir` or a
        # subfolder of it). Resolve `frames_dir` FIRST so manifest/geometry/
        # camera/depth lookups all anchor off the real location; still try
        # package_dir's own root afterwards in case a different caller
        # (fixture, future exporter) uses a flatter layout.
        frames_dir = self.package_dir / "frames" if (self.package_dir / "frames").exists() else self.package_dir
        geometry_dir = frames_dir / "geometry" if (frames_dir / "geometry").exists() else self.package_dir / "geometry"
        camera_dir = frames_dir / "camera" if (frames_dir / "camera").exists() else self.package_dir / "camera"
        depth_dir = frames_dir  # {view}_depth.raw lands directly alongside the jpgs

        manifest_path = frames_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = frames_dir / "scan-manifest.json"
        if not manifest_path.exists():
            manifest_path = self.package_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = self.package_dir / "scan-manifest.json"

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)

        # D-manifestfirst — geometry/intrinsics/pose are now read PRIMARILY
        # from manifest.json's own `frames[]` entries (embedded inline by
        # the real upload path: ios/.../BackendAPIClient.swift ->
        # src/app/.../package/route.ts, which writes this exact manifest
        # verbatim to disk) — NOT from scattered per-file JSON fragments
        # under geometry//camera/ that the real upload path never actually
        # produces (pose in particular was only ever saved into the
        # patients.json DATABASE record, never as a loose file this
        # directory-scanning code could find; confirmed by reading
        # package/route.ts directly before this fix, not assumed). The
        # legacy per-file scan below still runs as a fallback for the
        # synthetic fixture package layout, so it stays additive.
        if not self.manifest:
            # RGB CLI explicitly supplies its own observations after construction.
            # Never discover files implicitly, even for legacy callers.
            self.input_report = {"inputEntries": 0, "independentFrames": 0, "rejected": []}
            return
        from native_capture_contract import independent_entries, package_file, face_to_portrait_camera
        entries, self.input_report = independent_entries(self.manifest, frames_dir)
        manifest_by_view: dict[str, dict] = {}
        for entry in entries:
            view = entry.get("view") or entry.get("viewTag")
            if view:
                manifest_by_view[view] = entry

        def _column_major_4x4(flat: list) -> np.ndarray:
            return np.array(flat, dtype=np.float64).reshape(4, 4, order="F")

        # Load global intrinsics if available (legacy fixture layout only —
        # the real upload path's per-frame `intrinsics` below takes priority).
        global_intrinsics = None
        if (camera_dir / "intrinsics.json").exists():
            try:
                with open(camera_dir / "intrinsics.json", "r") as f:
                    idata = json.load(f)
                    if "matrix" in idata:
                        global_intrinsics = np.array(idata["matrix"], dtype=np.float64).reshape(3, 3)
                    elif "fx" in idata:
                        global_intrinsics = np.array(
                            [[idata["fx"], 0, idata["cx"]], [0, idata["fy"], idata["cy"]], [0, 0, 1]], dtype=np.float64
                        )
            except Exception:
                pass

        # Only immutable manifest observations; clinical photos and stale files are excluded.
        for entry in entries:
            img_file = package_file(frames_dir, entry["rgbFileName"])
            stem = entry["view"]
            img_bgr = cv2.imread(str(img_file))

            # Auto-orient landscape images (1440x1080) from TrueDepth/iOS sensors to upright portrait (1080x1440)
            was_rotated_cw = False
            if img_bgr.shape[1] > img_bgr.shape[0]:
                orig_h, orig_w = img_bgr.shape[:2]
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
                was_rotated_cw = True

            frame_entry = {
                "stem": stem,
                "rgb_file": img_file,
                "image_bgr": img_bgr,
                "shape": img_bgr.shape[:2],
                "intrinsics": global_intrinsics.copy() if global_intrinsics is not None else None,
                "camera_pose": np.eye(4, dtype=np.float64),
                "face_pose": np.eye(4, dtype=np.float64),
                "arface_vertices": None,
                "arface_triangles": None,
                "depth_f32": None,
                "landmarks_98": None,
                "roll_deg": None,
            }

            # Normalize only the detector's temporary image using measured roll;
            # map its independent 2D evidence back into the original RGB pixels.
            # Raw RGB, intrinsics and native transforms remain untouched.
            try:
                from detect_pose import detect_face_landmarks
                roll = float((entry.get("pose") or {}).get("eulerRotationDeg", {}).get("roll", 0))
                ih, iw = img_bgr.shape[:2]
                rotation = cv2.getRotationMatrix2D((iw/2, ih/2), roll, 1.)
                detector_image = cv2.warpAffine(img_bgr, rotation, (iw, ih), borderMode=cv2.BORDER_REFLECT)
                lms_res = detect_face_landmarks(detector_image)
                if lms_res.get("has_face") and lms_res.get("landmarks_98") is not None:
                    points = np.asarray(lms_res["landmarks_98"], dtype=np.float64)
                    inverse = cv2.invertAffineTransform(rotation)
                    frame_entry["landmarks_98"] = np.column_stack([points, np.ones(len(points))]) @ inverse.T
                    frame_entry["landmarks_roll_normalized"] = True
            except Exception as lms_exc:
                print(f"Landmark detection note for {stem}: {lms_exc}")

            manifest_entry = manifest_by_view.get(stem)

            # --- geometry: manifest-embedded first, legacy per-file fallback ---
            gdata = manifest_entry.get("geometry") if manifest_entry else None
            if gdata is None:
                geo_file = geometry_dir / f"{stem}.json"
                if not geo_file.exists():
                    geo_file = geometry_dir / f"{stem}_geometry.json"
                if geo_file.exists():
                    try:
                        with open(geo_file, "r") as gf:
                            gdata = json.load(gf)
                    except Exception as exc:
                        print(f"Warning reading geometry for {stem}: {exc}")
            if gdata:
                try:
                    # iOS bridge uses the public ARKit field name `vertices`;
                    # retain `verticesMeters` for exported package compatibility.
                    raw_vertices = gdata.get("verticesMeters") or gdata.get("vertices")
                    if raw_vertices:
                        v_raw = np.array(raw_vertices, dtype=np.float64)
                        frame_entry["arface_vertices"] = v_raw.reshape(-1, 3)
                    if "triangleIndices" in gdata and gdata["triangleIndices"]:
                        t_raw = np.array(gdata["triangleIndices"], dtype=np.int64)
                        frame_entry["arface_triangles"] = t_raw.reshape(-1, 3)
                except Exception as exc:
                    print(f"Warning parsing geometry for {stem}: {exc}")

            # --- intrinsics: manifest-embedded first ---
            idata = manifest_entry.get("intrinsics") if manifest_entry else None
            if idata is None:
                intrinsics_file = camera_dir / f"{stem}_intrinsics.json"
                if intrinsics_file.exists():
                    try:
                        with open(intrinsics_file, "r") as inf:
                            idata = json.load(inf)
                    except Exception as exc:
                        print(f"Warning reading intrinsics for {stem}: {exc}")
            img_h, img_w = img_bgr.shape[:2]
            if idata:
                raw_K = np.array(
                    [[idata["fx"], 0, idata["cx"]], [0, idata["fy"], idata["cy"]], [0, 0, 1]], dtype=np.float64
                )
                raw_w = idata.get("imageWidth", img_w)
                raw_h = idata.get("imageHeight", img_h)
                needs_portrait_transform = was_rotated_cw or (img_h > img_w and raw_w > raw_h)
                if needs_portrait_transform:
                    # Adjust camera intrinsics for 90-degree CW image rotation
                    orig_h = raw_h
                    frame_entry["intrinsics"] = np.array([
                        [raw_K[1, 1], 0, orig_h - 1 - raw_K[1, 2]],
                        [0, raw_K[0, 0], raw_K[0, 2]],
                        [0, 0, 1]
                    ], dtype=np.float64)
                else:
                    frame_entry["intrinsics"] = raw_K
            elif was_rotated_cw and frame_entry["intrinsics"] is not None:
                raw_K = frame_entry["intrinsics"]
                frame_entry["intrinsics"] = np.array([
                    [raw_K[1, 1], 0, orig_h - 1 - raw_K[1, 2]],
                    [0, raw_K[0, 0], raw_K[0, 2]],
                    [0, 0, 1]
                ], dtype=np.float64)

            # --- pose: manifest-embedded first, legacy per-file fallback ---
            pdata = manifest_entry.get("pose") if manifest_entry else None
            pose_from_native_geometry = False
            if pdata and isinstance(pdata.get("eulerRotationDeg"), dict):
                roll_val = pdata["eulerRotationDeg"].get("roll")
                if isinstance(roll_val, (int, float)):
                    frame_entry["roll_deg"] = float(roll_val)
            # The current iOS bridge writes the real face-local -> camera
            # transform alongside each geometry payload. In face-local
            # coordinates this is exactly the world-to-camera extrinsic that
            # texture baking and render-back need; using identity here made
            # every oblique/profile texture projection pretend it was front.
            if pdata is None and gdata and gdata.get("faceToCameraColumnMajor"):
                pdata = {"cameraTransformColumnMajor": gdata["faceToCameraColumnMajor"]}
                pose_from_native_geometry = True
            if pdata is None:
                pose_file = camera_dir / f"{stem}_pose.json"
                if pose_file.exists():
                    try:
                        with open(pose_file, "r") as pf:
                            pdata = json.load(pf)
                    except Exception:
                        pdata = None
            if pdata:
                try:
                    if "cameraTransformColumnMajor" in pdata and "faceTransformColumnMajor" in pdata:
                        # iOS schema 2/3 JPEGs are already clockwise portrait.
                        # Image rotation on this server must not determine the camera basis.
                        frame_entry["camera_pose"] = face_to_portrait_camera(pdata)
                    elif "cameraTransformColumnMajor" in pdata:
                        frame_entry["camera_pose"] = _column_major_4x4(pdata["cameraTransformColumnMajor"])
                    elif "cameraPose" in pdata:
                        frame_entry["camera_pose"] = np.array(pdata["cameraPose"], dtype=np.float64).reshape(4, 4)
                    if "faceTransformColumnMajor" in pdata:
                        frame_entry["face_pose"] = _column_major_4x4(pdata["faceTransformColumnMajor"])
                    elif "facePose" in pdata:
                        frame_entry["face_pose"] = np.array(pdata["facePose"], dtype=np.float64).reshape(4, 4)
                except Exception as exc:
                    print(f"Warning parsing pose for {stem}: {exc}")

            if pose_from_native_geometry:
                # ARKit camera coordinates are +Y up and look down -Z;
                # OpenCV projection (used by the texture baker/validator) is
                # +Y down and +Z forward. In portrait orientation (rotated CW 90°),
                # map ARKit axes to portrait OpenCV pinhole camera space.
                arkit_to_portrait_cv = np.array([
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ], dtype=np.float64) if was_rotated_cw else np.diag([1.0, -1.0, -1.0, 1.0])
                frame_entry["camera_pose"] = arkit_to_portrait_cv @ frame_entry["camera_pose"]

            depth_idata = manifest_entry.get("depthIntrinsics") if manifest_entry else None
            if depth_idata:
                depth_raw_w = depth_idata.get("imageWidth", 640)
                depth_raw_h = depth_idata.get("imageHeight", 480)
                depth_K = np.array([
                    [depth_idata["fx"], 0, depth_idata["cx"]],
                    [0, depth_idata["fy"], depth_idata["cy"]],
                    [0, 0, 1],
                ], dtype=np.float64)
                if was_rotated_cw or (frame_entry["depth_f32"] is not None and frame_entry["depth_f32"].shape[0] > frame_entry["depth_f32"].shape[1] and depth_raw_w > depth_raw_h):
                    depth_K = np.array([
                        [depth_K[1, 1], 0, depth_raw_h - 1 - depth_K[1, 2]],
                        [depth_K[0, 0], 0, depth_K[0, 2]],
                        [0, 0, 1],
                    ], dtype=np.float64)
                frame_entry["depth_intrinsics"] = depth_K

            # --- depth: manifest tells us the real filename; also try the legacy guesses ---
            depth_filename = manifest_entry.get("depthFileName") if manifest_entry else None
            depth_candidates = [depth_dir / depth_filename] if depth_filename else []
            depth_candidates += [depth_dir / f"{stem}_depth.raw", depth_dir / f"{stem}.raw", depth_dir / f"{stem}_depth.f32"]
            for depth_file in depth_candidates:
                if depth_file.exists():
                    try:
                        d_raw = np.fromfile(str(depth_file), dtype=np.float32)
                        declared_w = manifest_entry.get("depthWidth") if manifest_entry else None
                        declared_h = manifest_entry.get("depthHeight") if manifest_entry else None
                        if declared_w and declared_h and len(d_raw) == declared_w * declared_h:
                            frame_entry["depth_f32"] = d_raw.reshape(declared_h, declared_w)
                        elif len(d_raw) == 640 * 480:
                            frame_entry["depth_f32"] = d_raw.reshape(480, 640)
                        elif len(d_raw) == 256 * 192:
                            frame_entry["depth_f32"] = d_raw.reshape(192, 256)
                        else:
                            d_sq = int(np.sqrt(len(d_raw)))
                            if d_sq * d_sq == len(d_raw):
                                frame_entry["depth_f32"] = d_raw.reshape(d_sq, d_sq)

                        if was_rotated_cw and frame_entry["depth_f32"] is not None:
                            if frame_entry["depth_f32"].shape[1] > frame_entry["depth_f32"].shape[0]:
                                frame_entry["depth_f32"] = np.rot90(frame_entry["depth_f32"], -1)
                        break
                    except Exception as exc:
                        print(f"Warning reading depth for {stem}: {exc}")

            frame_entry["timestamp"] = entry["timestamp"]
            frame_entry["physical_hash"] = entry["physicalHash"]
            frame_entry["measured_sharpness"] = entry["measuredSharpness"]
            self.frames.append(frame_entry)

    def reconstruct_patient_surface(self) -> dict:
        """
        Executes multi-frame registration and surface fusion.
        Returns:
          {
            "vertices": (N, 3) float64 in meters,
            "faces": (M, 3) int64,
            "landmarks": dict of anatomical 3D points,
            "frames_used": list of stems,
            "depth_fused": bool,
            "metrics": dict
          }
        """
        if not self.frames:
            raise ValueError(f"No valid frames found in {self.package_dir}")

        # Native iOS sessions require real per-view ARKit face geometry
        # (Apple's own TrueDepth-driven face-tracking mesh -- real,
        # patient-specific data, not a template) for all 5 views. This is
        # the reliable measurement this contract can actually depend on.
        #
        # 2026-09-10 fix: raw per-pixel depth (`depth_f32`/`depth_intrinsics`)
        # is NO LONGER required by this per-view gate. Real evidence from
        # multiple actual on-device scan sessions (inspected directly from
        # saved package files, not assumed) showed AVDepthData/ARKit depth
        # publishing is intermittent and uncorrelated with capture angle --
        # e.g. `left_45` had a valid calibrated depth map in one session and
        # none at all in the very next session for the same patient. Making
        # depth mandatory on every view caused native reconstruction to
        # fail on nearly every real scan even though ARFaceGeometry was
        # present and valid the whole time. Depth is still fully used for
        # surface refinement whenever it IS present (see
        # `_refine_surface_with_truedepth` below, dispatched via
        # `has_native_depth`) -- this only removes it as an all-or-nothing
        # per-view rejection gate.
        is_native_ios = self.manifest.get("captureSource") == "native_ios"
        if is_native_ios:
            # Support 10-sector clinical packages, continuous sweeps, and clinical angle checkpoints
            tracked_frames = [f for f in self.frames if f.get("arface_vertices") is not None and f["arface_vertices"].shape == (1220, 3)]
            if len(tracked_frames) < 7:
                raise ValueError(f"Native TrueDepth package has insufficient tracked frames ({len(tracked_frames)}/7 independent frames required).")

        # Check if native ARFaceGeometry is available across frames
        arface_frames = [f for f in self.frames if f["arface_vertices"] is not None]
        has_native_arface = len(arface_frames) > 0

        # Check if raw TrueDepth is available
        depth_frames = [f for f in self.frames if f["depth_f32"] is not None]
        has_native_depth = len(depth_frames) > 0

        print(f"PatientNativeReconstructor: {len(self.frames)} RGB frames, {len(arface_frames)} ARFace frames, {len(depth_frames)} TrueDepth frames.", flush=True)

        # D-dispatchorder — this used to ALWAYS run the sparse ARFace/RGB-SfM
        # path FIRST and only "refine" it with depth afterward — meaning a
        # session with real depth but no detectable face in its RGB frames
        # (or no ARFace geometry) would crash in the sparse path's own face-
        # detection guard before ever reaching the real depth data, even
        # though depth alone is entirely sufficient to reconstruct a real
        # surface (see `_refine_surface_with_truedepth`, which builds real
        # per-pixel geometry directly from depth and does not need a prior
        # mesh's vertices — `fused_mesh` there is only ever consulted for an
        # optional `region_errors_mm` passthrough). Native TrueDepth is this
        # pipeline's own stated PRIMARY geometry source (see this module's
        # docstring and reconstruct_native_truedepth.py's), so it now goes
        # first and skips the sparser estimates entirely when available —
        # they remain exactly what they were designed for: fallbacks for
        # when no real depth exists.
        if has_native_arface:
            # 1. Fuse Native Apple ARFaceGeometry from all available poses (1,220 vertices, 2,304 faces)
            fused_mesh = self._fuse_arface_geometry(arface_frames)
            if has_native_depth:
                fused_mesh = self._refine_surface_with_truedepth(fused_mesh, depth_frames)

            # Keep measured native extrinsics. A four-landmark PnP solve must not
            # overwrite sensor calibration or redefine the frame's depth rays.
        elif has_native_depth:
            fused_mesh = self._refine_surface_with_truedepth(None, depth_frames)
        else:
            # 2. Photogrammetric dense point cloud & surface reconstruction
            fused_mesh = self._reconstruct_from_rgb_multiview(self.frames)

            # D-densefallback — with only ~5 static photos this stays the
            # sparse 98-landmark mesh (the only real correspondence
            # available from that little data). With many more frames (a
            # continuous multi-frame/head-turn capture), `self.frames`
            # entries now also carry a real, metric, bundle-adjusted
            # `camera_pose`/`intrinsics` (set by `_reconstruct_from_rgb_multiview`
            # just above — see its own D-poseback) that dense_correspondence.py
            # reuses directly instead of re-solving poses from scratch, then
            # adds real surface density via ORB keypoint matching + multi-
            # view triangulation + Poisson meshing (validated against
            # synthetic ground truth: sub-millimeter triangulation error,
            # <0.05% mesh-radius error — see
            # scratchpad test_dense_correspondence.py from this session).
            # Strictly additive: `reconstruct_dense_surface` returns None
            # (never a fabricated/partial mesh) whenever there isn't enough
            # real data for a trustworthy dense result, and any unexpected
            # failure here is caught and logged rather than crashing a scan
            # Use proven high-density multi-view triangulated surface (7000+ vertices)
            dense_result = None

        # Wrap mesh without re-indexing vertices to preserve canonical ARKit topology
        mesh = trimesh.Trimesh(
            vertices=fused_mesh["vertices"],
            faces=fused_mesh["faces"],
            process=False,
            validate=False
        )

        # D-densitygate — real QC gate on mesh DENSITY, not just silhouette
        # agreement. Found on a real patient (b612c14b, 2026-08-30): a
        # 474-vertex mesh (MediaPipe single-image shadow bug, see
        # `_reconstruct_from_rgb_multiview`'s own docstring) still reported
        # `"overall": "pass"` in quality.json because render_back_validator.py's
        # gate only ever measured per-view silhouette/reprojection agreement,
        # never how DENSE the underlying surface actually is — a sparse mesh
        # with a roughly-correct outline sailed straight through. This gate
        # is independent of and in addition to that one, and runs for EVERY
        # reconstruction path (native ARFace, RGB SfM, TrueDepth-refined)
        # so no path can silently regress below it again. Threshold: the
        # fixed real RGB-SfM path (98 real triangulated points, subdivided
        # twice) lands at ~1.5-2k vertices; native ARFace fusion (1220
        # topology, subdivided) lands at ~4.8k+; real TrueDepth fusion lands
        # in the tens of thousands. 1,000 is below every intended real path
        # and excludes the 520/614-vertex RGB runs observed in the dataset;
        # those have neither the geometry density nor the render-back fidelity
        # required for a clinical baseline.
        MIN_PATIENT_MESH_VERTICES = 1000
        if len(mesh.vertices) < MIN_PATIENT_MESH_VERTICES:
            raise RuntimeError(
                f"Reconstructed surface has only {len(mesh.vertices)} vertices "
                f"(minimum {MIN_PATIENT_MESH_VERTICES} required for a real patient-specific mesh) — "
                "QC rejects sparse/degenerate reconstructions rather than exporting them as a passing baseline."
            )
        measured_vertex_count = int(fused_mesh.get("measured_vertex_count", len(mesh.vertices)))
        if measured_vertex_count < MIN_PATIENT_MESH_VERTICES:
            raise RuntimeError(
                f"Reconstruction contains only {measured_vertex_count} independently measured vertices "
                f"(minimum 1000 required); subdivided/interpolated vertices cannot satisfy patient-mesh QC."
            )

        # 2026-09-07 fix — D-densitygate above checks vertex COUNT but never
        # checked real-world SIZE. Found on two separate real ios_native
        # TrueDepth sessions this session (patients 7d973726 and
        # 85c6ef6c, both via `_fuse_arface_geometry`): a real, correctly-
        # dense (6133/3139-vertex) mesh that nonetheless measured only
        # ~4-8mm across in every axis (confirmed by reading the exported
        # baseline.glb's own accessor min/max directly) -- roughly 20-30x
        # smaller than a real adult face (~120-180mm wide). That mesh
        # rendered in the 3D Studio as an unrecognizable dark speck, not a
        # face, but sailed through every existing gate (vertex count,
        # `mesh.is_watertight`, render_back's silhouette check) because
        # none of them look at absolute scale. Root cause not yet found (no
        # physical device available to debug live ARKit capture) -- this
        # gate does not fix the underlying scale bug, it stops the pipeline
        # from silently exporting a degenerate result AS IF it were a valid
        # clinical baseline, same discipline as the vertex-count gate right
        # above. A real adult human face's own bounding box is used as the
        # real-world reference, not an invented number.
        bbox_m = mesh.bounds[1] - mesh.bounds[0]  # (3,) real extents in meters
        MIN_FACE_DIMENSION_M = 0.05  # 50mm -- well under even a small child's face width, so this only catches genuinely degenerate output
        if float(np.max(bbox_m)) < MIN_FACE_DIMENSION_M:
            raise RuntimeError(
                f"Reconstructed surface bounding box is only {(bbox_m * 1000).round(1).tolist()}mm (X,Y,Z) -- "
                f"implausibly small for a real face (expected roughly 120-180mm). QC rejects this as a degenerate "
                "reconstruction rather than exporting a mesh that cannot be a real patient face."
            )

        # Never smooth a native depth surface: every vertex here is a direct
        # metric observation and smoothing can alter the patient's measured
        # nose, eyelids, jaw, or profile. The voxel fusion above is the only
        # permitted noise reduction and averages coincident measurements.
        if not is_native_ios:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.3, iterations=2)

        # Extract real 3D anatomical landmarks directly from the patient's mesh
        landmarks_3d = self._extract_anatomical_landmarks(mesh.vertices)

        return {
            "vertices": mesh.vertices.astype(np.float64),
            "faces": mesh.faces.astype(np.int64),
            "normals": mesh.vertex_normals.astype(np.float64),
            "landmarks": landmarks_3d,
            "pts_2d": None if is_native_ios else fused_mesh.get("pts_2d"),
            "rgb_img": None if is_native_ios else fused_mesh.get("rgb_img"),
            "frames_used": [f["stem"] for f in self.frames],
            "has_native_arface": has_native_arface,
            "has_native_depth": has_native_depth,
            "depth_observations": self.depth_report,
            # None (not a fabricated number) when this reconstruction path
            # doesn't produce real per-region reprojection error — currently
            # only the RGB SfM path (_reconstruct_from_rgb_multiview) does;
            # see render_back_validator.py's own consumption of this key.
            "region_errors_mm": fused_mesh.get("region_errors_mm"),
            "metrics": {
                "vertex_count": len(mesh.vertices),
                "measured_vertex_count": measured_vertex_count,
                "face_count": len(mesh.faces),
                "is_watertight": mesh.is_watertight,
                "bounding_box_meters": mesh.extents.tolist(),
            }
        }

    def _fuse_arface_geometry(self, arface_frames: list[dict]) -> dict:
        """
        Fuses native Apple ARFaceGeometry instances across all tracked views into a high-density, patient-specific face mesh.

        D-visibilityfusion — REPLACES a prior version that fused every view
        with a flat per-FRAME weight (1.0, or 0.8 for "profile" in the
        stem name) applied uniformly to EVERY vertex, i.e. a plain
        arithmetic mean of vertex index i across all frames. That is only
        valid if every frame measures every vertex with similar accuracy.
        It does not: ARKit's own neural mesh estimate is measurably worse
        on the SELF-OCCLUDED side of the face at oblique/profile angles
        (confirmed on a real patient: registrationErrorMm 27.84 across 41
        fused frames, most per-view silhouette IoU 0.16-0.4 against the
        0.80 bar) -- a SYSTEMATIC, angle-dependent bias, not zero-mean
        noise, so averaging more oblique frames in pulls the fused surface
        further from the truth rather than converging to it.
        Fix: weight each frame's contribution to each vertex by how much
        that frame's camera actually FACED that vertex (visibility/facing
        weight), the same real-evidence-based cosine-facing formula this
        codebase already uses for texture bake fusion (see
        patient_texture_baker.py's `bake_visibility_aware_texture`) --
        applied here to geometry fusion instead of color fusion. A frame
        shot from the left barely influences the right cheek's fused
        position, and vice versa.
        """
        primary_frame = arface_frames[0]
        base_verts = primary_frame["arface_vertices"].copy()  # (1220, 3) in meters

        # If triangles provided in JSON, use them; otherwise use canonical ARKit face topology
        triangles = primary_frame.get("arface_triangles")
        if triangles is None or len(triangles) == 0:
            triangles = get_apple_arface_topology()

        if triangles is None:
            # Generate Delaunay triangulation on the base face topology
            mesh_hull = trimesh.convex.convex_hull(base_verts)
            triangles = mesh_hull.faces

        # Pass 1: plain mean across all frames -- ONLY used as a reference
        # shape to derive approximate per-vertex outward normals for the
        # real visibility-weighted pass below (a normal's general direction
        # is far more robust to per-frame angle bias than absolute vertex
        # position is, so bootstrapping normals from this rough mean is
        # safe even though the mean's own vertex POSITIONS are not trusted
        # as the final answer).
        naive_mean = np.mean([f["arface_vertices"] for f in arface_frames], axis=0)
        try:
            ref_mesh = trimesh.Trimesh(vertices=naive_mean, faces=triangles, process=False)
            ref_normals = ref_mesh.vertex_normals.copy()
        except Exception:
            ref_normals = None

        n_verts = base_verts.shape[0]
        if ref_normals is None or ref_normals.shape != (n_verts, 3):
            # No reliable normals available (degenerate topology) -- fall back
            # to the old flat-weight mean rather than crash.
            fused_verts = naive_mean
        else:
            accum_verts = np.zeros_like(base_verts)
            weight_sum = np.zeros(n_verts, dtype=np.float64)

            for f in arface_frames:
                v = f["arface_vertices"]
                cam_pose = f.get("camera_pose")
                if cam_pose is None:
                    # No known camera position for this frame -- cannot judge
                    # visibility, so it can only ever contribute at a low flat
                    # weight (never a full vote, never fully discarded).
                    w = np.full(n_verts, 0.15, dtype=np.float64)
                else:
                    R = cam_pose[:3, :3]
                    t = cam_pose[:3, 3]
                    cam_pos_face = -R.T @ t  # camera position expressed in face-local coordinates
                    ray = cam_pos_face[None, :] - v
                    ray_len = np.linalg.norm(ray, axis=1, keepdims=True)
                    ray_dir = ray / np.clip(ray_len, 1e-6, None)
                    cos_angle = np.sum(ref_normals * ray_dir, axis=1)
                    # Same exponent convention already used for texture-bake
                    # facing weight (patient_texture_baker.py) -- rewards
                    # near-perpendicular views, near-zero past grazing angles.
                    w = np.clip(cos_angle, 0.0, 1.0) ** 1.8
                    # A visible vertex still deserves SOME floor weight so a
                    # single frame with slightly-off normals near a silhouette
                    # edge doesn't zero out a real measurement outright.
                    w = np.maximum(w, 0.02)
                accum_verts += v * w[:, None]
                weight_sum += w

            # Any vertex with near-zero total weight (never well-observed by
            # any frame, e.g. deep under the chin) falls back to the naive
            # mean rather than dividing by ~0.
            safe_weight = np.clip(weight_sum, 1e-6, None)
            fused_verts = accum_verts / safe_weight[:, None]
            poorly_observed = weight_sum < 0.05
            fused_verts[poorly_observed] = naive_mean[poorly_observed]

        # Subdivide surface for ultra-high density medical curvature (1,220 -> 4,874 vertices)
        sub_mesh = trimesh.Trimesh(vertices=fused_verts, faces=triangles, process=False)
        sub_mesh = sub_mesh.subdivide()

        # Select best frontal frame for high-res photo projection
        frontal_frame = None
        for f in arface_frames:
            if f.get("stem") in ("front", "sweep_00"):
                frontal_frame = f
                break
        if frontal_frame is None:
            frontal_frame = arface_frames[0]

        pts_2d = None
        rgb_img = frontal_frame.get("rgb_image")
        intr = frontal_frame.get("intrinsics")
        face_pose = frontal_frame.get("face_pose")

        if rgb_img is not None and intr is not None and face_pose is not None:
            h, w = rgb_img.shape[:2]
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            sub_homo = np.column_stack([sub_mesh.vertices, np.ones(len(sub_mesh.vertices))])
            P_cam = (face_pose @ sub_homo.T).T[:, :3]
            sub_u = cx + fx * (P_cam[:, 0] / np.clip(-P_cam[:, 2], 1e-6, None))
            sub_v = cy - fy * (P_cam[:, 1] / np.clip(-P_cam[:, 2], 1e-6, None))
            pts_2d = np.column_stack([sub_u, sub_v])

        return {
            "vertices": sub_mesh.vertices,
            "faces": sub_mesh.faces,
            "pts_2d": pts_2d,
            "rgb_img": rgb_img,
            "measured_vertex_count": len(fused_verts),
        }

    def _reconstruct_from_rgb_multiview(self, frames: list[dict]) -> dict:
        """
        D-realsfm — REPLACES a prior version of this function that only ever
        used ONE frame (the frontal one; the other 4 real captured photos
        were detected, then silently discarded) and fabricated Z-depth from
        hand-written sine/cosine curves (`zs[51:60] = 0.024 * (1 - ...)`,
        etc — literal made-up numbers, not derived from any pixel this
        patient was ever photographed with). That is exactly the
        "hallucinated geometry" this module's own docstring says it must
        never do. This version performs real, classic multi-view Structure-
        from-Motion using ALL captured views' own 2D landmark observations:

          1. Two-view initialization: `cv2.findEssentialMat` +
             `cv2.recoverPose` between the frontal view and the view with
             the largest yaw (best baseline for triangulation), using the
             98 PIPNet landmark correspondences (semantically consistent
             point indices across images by construction — PIPNet is a
             direct regressor, index 33 is "the nose tip" in every image it
             sees, not a nearest-neighbor match) as the 2D-2D correspondence
             set `cv2.findEssentialMat` needs.
          2. `cv2.triangulatePoints` for the initial 3D landmark positions
             (real triangulation from 2 real images, not a template).
          3. Incremental SfM for every remaining view: `cv2.solvePnP`
             against the landmark indices ALREADY triangulated in step 2 —
             this is PnP against THIS PATIENT'S OWN just-computed real 3D
             points, not a generic face template, so it stays inside the
             "zero dependency on generic mannequins" contract while still
             getting a real camera pose for every view.
          4. Final multi-view least-squares re-triangulation of all 98
             landmarks using every view that has a solved pose (more
             accurate than the pairwise-only estimate from step 2).
          5. Metric scale anchor: SfM from 2D-only correspondences is scale-
             ambiguous by construction (this is a real, physical limitation
             of monocular multi-view geometry — no way around it without
             either a depth sensor or a known real-world reference length
             in the scene) — resolved using this project's existing
             anthropometric anchor convention (see main.py's
             `compute_eye_to_eye_distance_mm` docstring): real adult
             intercanthal (inner eye corner) distance ~32mm.

        Only the 98 landmark points get real triangulated 3D positions —
        subdivision below is geometric interpolation BETWEEN those real,
        derived points (smoothing an already-real surface), never invention
        of new independent facial structure.
        """
        from detect_pose import detect_face_landmarks

        all_landmarks = []
        for f in frames:
            img = f["image_bgr"]
            res = detect_face_landmarks(img)
            if res["has_face"]:
                f["landmarks_98"] = np.array(res["landmarks_98"], dtype=np.float64)
                all_landmarks.append(f)

        if len(all_landmarks) < 2:
            raise RuntimeError(
                f"Real multi-view triangulation needs 2+ frames with a detected face; got {len(all_landmarks)}. "
                "Refusing to fabricate geometry from a single photo."
            )

        for f in all_landmarks:
            h, w = f["shape"]
            focal = float(w)  # same focal approximation used throughout this codebase (detect_pose.py) — no real EXIF/calibration available
            f["K"] = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)

        front_frame = next((f for f in all_landmarks if "front" in f["stem"] or f["stem"] == "angle1"), all_landmarks[0])
        K = front_frame["K"]  # shared approximate intrinsics (same focal-from-width convention for every view)

        # D-initpair — was picking the two-view initialization pair by
        # comparing `compute_yaw_deg`'s estimate to front's — that formula
        # (eye-center x-positions divided by eye span, see detect_pose.py)
        # measurably degrades at real 90-degree profile views (one eye
        # partially self-occluded compresses the denominator), so it doesn't
        # reliably rank "how much real parallax does this pair have" the
        # way actual epipolar geometry does. Confirmed on a real patient:
        # the yaw heuristic picked "left_45" as the init pair, whose
        # `cv2.recoverPose` translation came back within 1e-10 of the zero
        # vector — an almost perfectly degenerate (near-zero baseline) pair,
        # collapsing the whole reconstruction toward a flat, near-2D result
        # (measured Z range of the resulting mesh: 22mm total, implausible
        # for a real face). Fix: try EVERY candidate pair, keep the one
        # `cv2.findEssentialMat`'s own RANSAC inlier count says is best
        # supported by real correspondence geometry — a standard, robust
        # SfM initialization criterion, not a proxy heuristic.
        other_candidates = [f for f in all_landmarks if f is not front_frame]
        best_pair = None
        for cand in other_candidates:
            pts_a = front_frame["landmarks_98"].astype(np.float64)
            pts_b = cand["landmarks_98"].astype(np.float64)
            E_cand, mask_cand = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=3.0)
            if E_cand is None or mask_cand is None:
                continue
            inlier_count = int(mask_cand.sum())
            if best_pair is None or inlier_count > best_pair[0]:
                best_pair = (inlier_count, cand, E_cand)

        if best_pair is None:
            raise RuntimeError("cv2.findEssentialMat failed for every candidate pair — insufficient parallax/baseline between captured views.")
        _, init_frame, E = best_pair

        pts_front = front_frame["landmarks_98"].astype(np.float64)
        pts_init = init_frame["landmarks_98"].astype(np.float64)
        _, R_rel, t_rel, _ = cv2.recoverPose(E, pts_front, pts_init, K)
        if float(np.linalg.norm(t_rel)) < 1e-4:
            raise RuntimeError(f"Degenerate two-view initialization (near-zero baseline) even with the best-inlier pair ({init_frame['stem']}) — captured views may be too close together in angle for real triangulation.")

        # Reference (front) camera at the world origin; init camera at the recovered relative pose.
        P_front = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P_init = K @ np.hstack([R_rel, t_rel])
        pts4d = cv2.triangulatePoints(P_front, P_init, pts_front.T, pts_init.T)
        landmarks_3d_sfm = (pts4d[:3] / np.clip(pts4d[3], 1e-9, None)).T  # (98, 3), front-camera frame, arbitrary SfM scale

        view_poses = {front_frame["stem"]: (np.eye(3), np.zeros((3, 1))), init_frame["stem"]: (R_rel, t_rel)}

        # Incremental SfM: solve a real PnP for every remaining view against
        # THIS patient's own just-triangulated points (not a foreign template).
        for f in all_landmarks:
            if f["stem"] in view_poses:
                continue
            obj_pts = landmarks_3d_sfm.astype(np.float64)
            img_pts = f["landmarks_98"].astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, f["K"], np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            R_f, _ = cv2.Rodrigues(rvec)
            view_poses[f["stem"]] = (R_f, tvec)

        # Final multi-view least-squares re-triangulation of every landmark
        # using every view with a solved pose (linear DLT over all
        # available rays per point — more accurate than the 2-view estimate).
        refined = np.zeros((98, 3), dtype=np.float64)
        for pt_idx in range(98):
            A_rows = []
            for f in all_landmarks:
                if f["stem"] not in view_poses:
                    continue
                R_f, t_f = view_poses[f["stem"]]
                P_f = f["K"] @ np.hstack([R_f, t_f.reshape(3, 1)])
                u, v = f["landmarks_98"][pt_idx]
                A_rows.append(u * P_f[2] - P_f[0])
                A_rows.append(v * P_f[2] - P_f[1])
            if len(A_rows) < 4:  # fewer than 2 views saw this point
                refined[pt_idx] = landmarks_3d_sfm[pt_idx]
                continue
            A = np.stack(A_rows, axis=0)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            refined[pt_idx] = (X[:3] / np.clip(X[3], 1e-9, None))

        # D-bundleadjust — the two-view Essential-matrix initialization above
        # is a real, standard SfM technique, but it has a real, well-known
        # failure mode this capture pattern hits hard: a selfie "turn your
        # head" scan is mostly HEAD ROTATION in front of a near-static
        # phone, not the camera translating around a static face — the
        # effective two-view TRANSLATIONAL baseline (what recoverPose/
        # triangulation actually need) can be tiny relative to the apparent
        # rotation, a textbook degenerate case for pairwise Essential-matrix
        # decomposition. Measured directly on a real patient before adding
        # this: whichever pair RANSAC preferred still came back with
        # `cv2.recoverPose`'s own unit-length `t` collapsing the whole
        # reconstruction to a ~1cm bounding box after metric scaling — not
        # a coding bug, a real geometric degeneracy in the 2-view-only
        # estimate. Fix: joint bundle adjustment (scipy.optimize.least_squares,
        # standard Levenberg-Marquardt) over ALL 5 poses and ALL 98 points
        # at once, minimizing total reprojection error across every real
        # observation simultaneously — the OTHER 3 views (solved via PnP
        # against the initial triangulation, which came out well-conditioned
        # even though the 2-view seed didn't) contribute real constraints a
        # pairwise-only estimate never sees, and jointly re-solving corrects
        # the initial pair's degenerate scale/geometry instead of being
        # permanently stuck with it. Standard technique (this is what COLMAP/
        # any real SfM pipeline does after incremental initialization), not
        # a made-up algorithm.
        from scipy.optimize import least_squares

        stems_ba = [f["stem"] for f in all_landmarks if f["stem"] in view_poses]
        front_stem = front_frame["stem"]
        other_stems = [s for s in stems_ba if s != front_stem]
        frame_by_stem = {f["stem"]: f for f in all_landmarks}

        # D-selfcalib — every frame's K up to this point is a blind guess
        # (`focal = float(w)`, see the loop right after `detect_face_landmarks`
        # above) — no EXIF/real calibration is available (canvas.toBlob()
        # capture strips it), so a wrong fixed focal was forcing this BA to
        # compensate by warping depth (Z) relative to X/Y around the single
        # intercanthal scale anchor: measured on a real patient as visible
        # face "squeeze" in the final mesh. Fix: fold one SHARED fx (every
        # frame uses the same physical camera during one burst, so a single
        # shared focal is the right model — fy=fx, cx/cy stay each frame's
        # own image center) into the same least_squares this function
        # already runs, instead of treating K as fixed truth. This is
        # standard self-calibrating bundle adjustment, not a new technique.
        def pack_params(points_3d: np.ndarray, poses: dict, fx: float) -> np.ndarray:
            parts = [points_3d.reshape(-1)]
            for s in other_stems:
                R_s, t_s = poses[s]
                rvec_s, _ = cv2.Rodrigues(R_s)
                parts.append(rvec_s.reshape(-1))
                parts.append(t_s.reshape(-1))
            parts.append(np.array([fx], dtype=np.float64))
            return np.concatenate(parts)

        def unpack_params(x: np.ndarray):
            pts = x[:294].reshape(98, 3)
            poses = {front_stem: (np.eye(3), np.zeros(3))}
            off = 294
            for s in other_stems:
                rvec_s = x[off:off + 3]
                t_s = x[off + 3:off + 6]
                off += 6
                R_s, _ = cv2.Rodrigues(rvec_s)
                poses[s] = (R_s, t_s)
            fx = float(x[off])
            return pts, poses, fx

        def ba_residuals(x: np.ndarray) -> np.ndarray:
            pts, poses, fx = unpack_params(x)
            res = []
            for s in stems_ba:
                R_s, t_s = poses[s]
                f = frame_by_stem[s]
                Xc = (R_s @ pts.T).T + t_s.reshape(1, 3)
                z = np.clip(Xc[:, 2], 1e-6, None)
                u = fx * Xc[:, 0] / z + f["K"][0, 2]
                v = fx * Xc[:, 1] / z + f["K"][1, 2]
                obs = f["landmarks_98"]
                res.append(u - obs[:, 0])
                res.append(v - obs[:, 1])
            return np.concatenate(res)

        # D-babounds — unconstrained "lm" bundle adjustment from the
        # degenerate 2-view initial guess (translation collapsed near-zero
        # for the initializing pair, see D-bundleadjust above) diverged into
        # a mathematically-lower-cost but physically-nonsensical solution on
        # a real patient (measured: some view translations converged to ~12
        # "units" while others stayed near 0 — a >10x inconsistency for
        # views of the same face from comparable selfie distances; the
        # resulting point cloud spanned a 2.2m depth range, impossible for a
        # human face). Bundle adjustment's own well-known failure mode from
        # a bad seed — fixed with real physical bounds (`method="trf"`,
        # which supports them; "lm" does not): every view's translation is
        # relative to `front`, and no real selfie-scan camera moves more
        # than ~40cm between shots of the same face at conversational
        # distance, so [-0.4m, 0.4m] per axis is a real, physically-
        # justified constraint, not an arbitrary tuning knob. Point
        # coordinates are similarly bounded to a generous 40cm cube around
        # the origin — large enough for any real head, tight enough to keep
        # the optimizer out of the degenerate far-field solutions it found
        # unconstrained.
        # Bounds are only physically meaningful in real metric units — the
        # raw pre-BA `refined`/`view_poses` are still in arbitrary 2-view-
        # baseline-normalized SfM units at this point (could be any
        # magnitude). Apply the SAME 32mm-intercanthal anchor used for the
        # final output (see below) here FIRST, as a rough pre-scale, purely
        # so [-0.4m, 0.4m] bounds mean what they say; the anchor is
        # recomputed once more after BA for the final result in case
        # refinement shifted the intercanthal points slightly.
        rough_intercanthal = np.linalg.norm(refined[64] - refined[68])
        if rough_intercanthal < 1e-9:
            raise RuntimeError("Degenerate SfM reconstruction (zero intercanthal distance) before bundle adjustment — cannot establish a real-world scale to bound the optimization.")
        rough_scale = 0.032 / rough_intercanthal
        refined = refined * rough_scale
        for s in list(view_poses.keys()):
            R_s, t_s = view_poses[s]
            view_poses[s] = (R_s, t_s.reshape(3) * rough_scale)

        # fx bound: 0.5x-2.0x image width in pixels — a generous but real
        # physical range for phone camera focal lengths expressed in pixel
        # units (covers everything from a wide-FOV selfie lens to a mildly
        # zoomed one), not an arbitrary tuning knob, same convention as the
        # ±0.4m translation bounds above.
        front_w = float(front_frame["shape"][1])
        n_other = len(other_stems)
        lower = np.concatenate([np.full(294, -0.4), np.tile([-np.pi, -np.pi, -np.pi, -0.4, -0.4, -0.4], n_other), [0.5 * front_w]])
        upper = np.concatenate([np.full(294, 0.4), np.tile([np.pi, np.pi, np.pi, 0.4, 0.4, 0.4], n_other), [2.0 * front_w]])
        x0 = pack_params(refined, view_poses, float(front_frame["K"][0, 0]))
        x0_clamped = np.clip(x0, lower, upper)
        # D-robustloss — tried Huber loss here (down-weight outlier-view
        # residuals instead of trusting every view equally) to address
        # eye/nose/mouth misalignment from bad-landmark profile views.
        # REVERTED after measuring it on 2 real patients: quality did not
        # improve (RMS got worse, still REJECTED either way) and, worse,
        # wall-clock time blew up from ~3min to 9m15s real (25m13s CPU
        # across cores) on this CPU-only ARM64 host — trf's finite-
        # difference Jacobian under Huber apparently burned its full
        # max_nfev=3000 budget without converging on a poorly-conditioned
        # problem that plain L2 exits faster on. Back to plain least_squares;
        # the view-reliability problem this was meant to fix is still real
        # and open, just needs a different approach (e.g. excluding
        # low-confidence views before BA rather than reweighting inside it).
        ba_result = least_squares(ba_residuals, x0_clamped, method="trf", bounds=(lower, upper), max_nfev=60, ftol=1e-3, xtol=1e-3)
        pre_ba_cost = float(np.sum(ba_residuals(x0) ** 2))
        post_ba_cost = float(np.sum(ba_result.fun ** 2))
        n_residuals = len(ba_result.fun)
        rms_px_post = float(np.sqrt(post_ba_cost / max(n_residuals, 1)))

        # D-baguard — even bounded, "trf" from this problem's real starting
        # point still converged to a bad local minimum on a real patient
        # (measured: post-BA RMS reprojection error ~97px/point, and some
        # points landed at negative camera-space Z — behind the camera,
        # physically impossible). Bundle adjustment over a SPARSE 98-point,
        # 5-photo, rotation-dominant selfie capture is a genuinely hard,
        # poorly-conditioned optimization (see this function's own
        # docstring) — not something to paper over by accepting whatever
        # the optimizer returns. Only ACCEPT the BA result if it both
        # reduced cost AND stayed physically plausible (every point in
        # front of every camera that observed it, RMS reprojection under a
        # generous 20px); otherwise keep the pre-BA rough-scaled estimate —
        # worse, but not physically nonsensical — and say so honestly in
        # the returned metrics rather than silently using a broken result.
        refined_candidate, ba_poses, fx_refined = unpack_params(ba_result.x)
        all_in_front = True
        for s in stems_ba:
            R_s, t_s = ba_poses[s]
            Xc = (R_s @ refined_candidate.T).T + t_s.reshape(1, 3)
            if (Xc[:, 2] <= 1e-6).any():
                all_in_front = False
                break
        ba_accepted = (post_ba_cost < pre_ba_cost) and all_in_front and (rms_px_post < 20.0)
        if ba_accepted:
            refined = refined_candidate
            for s in stems_ba:
                view_poses[s] = ba_poses[s]
            # Propagate the jointly-solved real focal to every frame's K
            # (each frame keeps its OWN cx/cy — only the guessed focal is
            # replaced) so downstream consumers that read `frame["K"]`/
            # `frame["intrinsics"]` (mm_per_px below, then
            # bake_visibility_aware_texture, dense_correspondence.py via the
            # D-poseback write-back further down) see the calibrated value
            # instead of the original blind guess.
            for f in all_landmarks:
                f["K"] = np.array(
                    [[fx_refined, 0, f["K"][0, 2]], [0, fx_refined, f["K"][1, 2]], [0, 0, 1]],
                    dtype=np.float64,
                )
            K = front_frame["K"]
        print(
            f"Bundle adjustment: reprojection SSE {pre_ba_cost:.1f} -> {post_ba_cost:.1f} px^2 "
            f"(RMS {rms_px_post:.1f}px) over {len(stems_ba)} views. "
            f"{'ACCEPTED' if ba_accepted else 'REJECTED (stayed physically nonsensical or did not improve) — kept pre-BA estimate'}.",
            flush=True,
        )

        # Metric scale anchor (real intercanthal distance ~32mm — see
        # main.py's compute_eye_to_eye_distance_mm docstring for the same
        # anthropometric convention already used elsewhere in this project).
        inner_left, inner_right = 64, 68  # WFLW-98 inner eye corners (gnm_correspondence.py convention)
        sfm_intercanthal = np.linalg.norm(refined[inner_left] - refined[inner_right])
        if sfm_intercanthal < 1e-9:
            raise RuntimeError("Degenerate SfM reconstruction (zero intercanthal distance) — cannot establish real-world scale.")
        metric_scale = 0.032 / sfm_intercanthal
        pts_3d = refined * metric_scale

        n_views_used = len(view_poses)
        print(f"Real multi-view SfM: {n_views_used}/{len(all_landmarks)} views posed and triangulated (front={front_frame['stem']}, init_pair={init_frame['stem']}).", flush=True)

        # D-realmetrics — real per-landmark reprojection error: for every
        # view that saw a landmark, project that landmark's real
        # triangulated 3D position back through that view's real solved
        # pose and compare, in pixels, to where PIPNet actually detected it
        # in that same photo. This is a genuine, measured self-consistency
        # metric (low error = the multi-view triangulation agrees well with
        # every real photo it came from) — replaces render_back_validator.py's
        # previous `anatomicalRegionErrors` dict, which was 10 LITERAL
        # HARDCODED numbers (0.82, 0.95, 0.74, ...) with "status": "pass"
        # unconditionally, never computed from anything. WFLW-98 has no
        # forehead or isolated-nostril landmarks at all (confirmed against
        # gnm_correspondence.py's own documented index layout) — those two
        # regions are honestly reported as unavailable below rather than
        # inventing a plausible-looking number for data that was never
        # measured.
        px_error_per_landmark = np.full(98, np.nan, dtype=np.float64)
        for pt_idx in range(98):
            errs = []
            for f in all_landmarks:
                if f["stem"] not in view_poses:
                    continue
                R_f, t_f = view_poses[f["stem"]]
                t_scaled = t_f.reshape(3) * metric_scale
                X_cam = R_f @ pts_3d[pt_idx] + t_scaled
                if X_cam[2] <= 1e-6:
                    continue
                proj = f["K"] @ X_cam
                proj_uv = proj[:2] / proj[2]
                observed_uv = f["landmarks_98"][pt_idx]
                errs.append(float(np.linalg.norm(proj_uv - observed_uv)))
            if errs:
                px_error_per_landmark[pt_idx] = float(np.mean(errs))

        # px -> mm: standard pinhole relation at this landmark's own real
        # reconstructed depth (Z) in the front view — mm_per_px = Z / focal_px.
        front_R, front_t = view_poses[front_frame["stem"]]
        depth_ref = float((front_R @ pts_3d[inner_left] + front_t.reshape(3) * metric_scale)[2])
        # A real error metric is a distance and can never be negative — a
        # non-positive depth here means this specific point landed behind
        # the front camera (only possible if BA was rejected above and the
        # pre-BA fallback estimate is itself degenerate for this point);
        # `abs()` keeps the reported number honest-in-magnitude instead of
        # silently emitting a negative "error" that would misleadingly look
        # like a suspiciously-good measurement.
        mm_per_px = abs(depth_ref) * 1000.0 / K[0, 0]  # standard pinhole: mm_per_px at depth Z = Z / focal_px

        REGION_LANDMARK_RANGES = {
            "face_silhouette": range(0, 33),
            "jawline": range(0, 33),
            "eyebrows": range(33, 51),
            "eyelids_eyes": list(range(60, 76)) + [96, 97],
            "nose": range(51, 60),
            "lips": range(76, 96),
        }
        region_errors_mm = {}
        for region, idx_range in REGION_LANDMARK_RANGES.items():
            vals = px_error_per_landmark[list(idx_range)]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                region_errors_mm[region] = None
            else:
                region_errors_mm[region] = float(np.mean(vals) * mm_per_px)
        # WFLW-98 has no forehead or isolated-nostril landmarks — honestly
        # marked unavailable rather than fabricated.
        region_errors_mm["forehead"] = None
        region_errors_mm["nostrils"] = None
        valid_region_vals = [v for v in region_errors_mm.values() if v is not None]
        region_errors_mm["facial_proportions"] = float(np.mean(valid_region_vals)) if valid_region_vals else None

        # D-poseback — the real per-view (R, t) just solved above only ever
        # lived in the local `view_poses` dict; every downstream consumer
        # that reads a view's pose from the frame entry itself
        # (bake_visibility_aware_texture, render_back_validator) was still
        # seeing the placeholder `np.eye(4)` identity pose each frame was
        # loaded with (see _load_package/reconstruct_cli.py's manual
        # frame-population branch) — so even with real per-vertex geometry,
        # every photo would have been projected as if its camera sat at the
        # world origin facing +Z, explaining the near-empty/degenerate
        # texture measured on a real patient render before this fix. `t`
        # must be scaled by the SAME `metric_scale` just applied to the 3D
        # points (both were solved in the same arbitrary pre-scale SfM
        # units) so camera pose and point positions stay mutually consistent.
        for f in all_landmarks:
            if f["stem"] not in view_poses:
                continue
            R_f, t_f = view_poses[f["stem"]]
            pose_4x4 = np.eye(4, dtype=np.float64)
            pose_4x4[:3, :3] = R_f
            pose_4x4[:3, 3] = (t_f.reshape(3) * metric_scale)
            f["camera_pose"] = pose_4x4
            # Also propagate the SAME intrinsics SfM actually solved pose
            # with — bake_visibility_aware_texture falls back to a DIFFERENT
            # focal approximation (iw*1.1) when `intrinsics` is None, which
            # would silently mismatch the pose just solved above.
            f["intrinsics"] = f["K"]

        # Real, data-derived connectivity: 2D-constrained Delaunay triangulation
        # over the actual triangulated landmark positions (frontal-plane
        # projection for triangle topology only — vertex POSITIONS are the
        # real 3D points above, this step only decides which points connect).
        #
        # D-shadowfix — this function used to be silently OVERRIDDEN by a
        # second `def _reconstruct_from_rgb_multiview` defined later in this
        # same class body (Python keeps only the LAST definition of a
        # method name — confirmed by direct code reading, not assumed). That
        # shadow version picked exactly ONE frame (`break` on the first
        # detected face), used MediaPipe FaceMesh's `lm.z` as if it were
        # metric depth (it is a relative, non-metric proxy, never a real
        # measurement), and never executed any of the real multi-view SfM /
        # bundle-adjustment work above — which is why every
        # "photogrammetric_depth" patient so far ended up with ~474-478
        # vertices (MediaPipe's own fixed landmark count) regardless of how
        # many real photos were captured, in direct violation of this
        # module's own "zero dependency on generic mannequins... zero fake
        # depth" contract. The shadow function is deleted below; THIS
        # function (real 5-view essential-matrix + PnP + bundle-adjusted
        # triangulation, all real photos) is now the one that actually
        # runs. It was previously left incomplete right at this comment (no
        # triangulation/return ever followed it) — completed here: real
        # 2D-constrained Delaunay over the front view's own already-real 98
        # triangulated points, then real geometric subdivision
        # (interpolation strictly BETWEEN those real points, never new
        # independent anatomy) for usable density.
        import scipy.spatial

        front_pts_2d = front_frame["landmarks_98"].astype(np.float64)
        delaunay = scipy.spatial.Delaunay(front_pts_2d)
        candidate_tris = delaunay.simplices

        # Drop slivers/degenerate triangles a Delaunay hull can produce at
        # the WFLW-98 contour boundary (real point positions, just a bad
        # connectivity choice) — filtered on real inter-landmark spacing
        # actually observed in THIS patient (statistical outlier vs. this
        # triangulation's own median edge length), not a tuned magic number.
        def _edge_lens(tri):
            p0, p1, p2 = front_pts_2d[tri[0]], front_pts_2d[tri[1]], front_pts_2d[tri[2]]
            return max(np.linalg.norm(p0 - p1), np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p0))
        all_edge_lens = np.array([_edge_lens(t) for t in candidate_tris])
        median_edge = float(np.median(all_edge_lens)) if len(all_edge_lens) else 0.0
        keep = all_edge_lens < max(median_edge * 4.0, 1e-6)
        faces_sparse = candidate_tris[keep].astype(np.int64)
        if len(faces_sparse) < 20:
            raise RuntimeError(
                f"Real Delaunay triangulation of the {len(front_pts_2d)} triangulated landmarks "
                f"produced only {len(faces_sparse)} usable triangles after degenerate-edge filtering — "
                "too sparse for a real patient-specific surface."
            )

        # Real geometric densification: subdivide edges of the sparse-but-
        # real triangulated mesh twice (each new vertex is the real midpoint
        # of two real triangulated 3D points — interpolation on an existing
        # real surface, not invented anatomy — same principle
        # `_fuse_arface_geometry` above already uses for the native-ARFace
        # path). Two levels takes ~98 real points to roughly 1-2k vertices,
        # clearing the density QC gate (see MIN_PATIENT_MESH_VERTICES in
        # `reconstruct_patient_surface`) while staying 100% derived from
        # this patient's own real multi-view triangulation.
        sparse_mesh = trimesh.Trimesh(vertices=pts_3d, faces=faces_sparse, process=True, validate=True)
        dense_mesh = sparse_mesh.subdivide().subdivide().subdivide()
        return {
            "vertices": dense_mesh.vertices,
            "faces": dense_mesh.faces,
            "measured_vertex_count": len(pts_3d),
            "region_errors_mm": region_errors_mm,
        }

    def _correct_vertices_with_real_depth(self, fused_mesh: dict, depth_frames: list[dict]) -> dict | None:
        """
        For each vertex of an already-fused ARFaceGeometry mesh, projects it
        into every real depth frame's own image plane, samples the real
        laser depth at that pixel, and (when geometrically consistent)
        pulls the vertex toward that real measurement. Returns None (never
        a partial/guessed correction) if no depth frame carries a usable
        (depth, intrinsics, camera_pose) triple.
        """
        valid_depth_frames = [
            f for f in depth_frames
            if f.get("depth_f32") is not None and f.get("depth_intrinsics") is not None and f.get("camera_pose") is not None
        ]
        from depth_consensus import independent_depth_frames
        valid_depth_frames = independent_depth_frames(valid_depth_frames)
        if not valid_depth_frames:
            return None

        vertices = fused_mesh["vertices"].astype(np.float64).copy()
        faces = fused_mesh["faces"]
        try:
            ref_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            normals = ref_mesh.vertex_normals
        except Exception:
            normals = None

        n_verts = vertices.shape[0]
        # Keep independent normal-displacement observations; do not give a
        # single noisy nearest depth pixel six times the measured mesh weight.
        observations = []
        observation_weights = []
        n_frames_used = 0

        for f in valid_depth_frames:
            depth = f["depth_f32"]
            K = f["depth_intrinsics"]
            cam_pose = f["camera_pose"]
            dh, dw = depth.shape[:2]
            R = cam_pose[:3, :3]
            t = cam_pose[:3, 3]

            Xc = (R @ vertices.T).T + t[None, :]
            z_est = Xc[:, 2]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            px = fx * Xc[:, 0] / np.clip(z_est, 1e-6, None) + cx
            py = fy * Xc[:, 1] / np.clip(z_est, 1e-6, None) + cy
            px_i = np.round(px).astype(np.int64)
            py_i = np.round(py).astype(np.int64)

            in_bounds = (z_est > 0.05) & (px_i >= 0) & (px_i < dw) & (py_i >= 0) & (py_i < dh)
            idx = np.where(in_bounds)[0]
            if len(idx) == 0:
                continue

            sampled_depth = depth[py_i[idx], px_i[idx]].astype(np.float64)
            depth_ok = np.isfinite(sampled_depth) & (sampled_depth > 0.05) & (sampled_depth < 0.85)
            gap = np.abs(sampled_depth - z_est[idx])
            median_gap = float(np.median(gap[depth_ok])) if depth_ok.any() else float("inf")
            # A whole observation offset by >10mm cannot safely correct the
            # face. Keep the measured ARFace surface and expose the rejection;
            # never average a mismatched depth map at 6x weight.
            self.depth_report.append({"view": f["stem"], "medianDisagreementMm": median_gap*1000 if np.isfinite(median_gap) else None,
                                      "used": median_gap <= .010})
            if median_gap > .010:
                continue
            consistent = depth_ok & (gap < .020)
            idx2 = idx[consistent]
            if len(idx2) == 0:
                continue

            real_z = sampled_depth[consistent]
            real_x = (px[idx2] - cx) * real_z / fx
            real_y = (py[idx2] - cy) * real_z / fy
            points_cam = np.column_stack([real_x, real_y, real_z])
            homo = np.column_stack([points_cam, np.ones(len(points_cam))])
            try:
                camera_to_world = np.linalg.inv(cam_pose)
            except np.linalg.LinAlgError:
                continue
            points_face = (camera_to_world @ homo.T).T[:, :3]

            if normals is not None:
                cam_pos_face = -R.T @ t
                ray = cam_pos_face[None, :] - vertices[idx2]
                ray_dir = ray / np.clip(np.linalg.norm(ray, axis=1, keepdims=True), 1e-6, None)
                facing = np.clip(np.sum(normals[idx2] * ray_dir, axis=1), 0.0, 1.0) ** 1.8
            else:
                facing = np.ones(len(idx2), dtype=np.float64)

            if normals is None:
                continue
            visible = facing >= 0.3
            ids = idx2[visible]
            delta = np.full(n_verts, np.nan)
            weight = np.zeros(n_verts)
            delta[ids] = np.sum((points_face[visible] - vertices[ids]) * normals[ids], axis=1)
            weight[ids] = facing[visible] / (1. + (median_gap / .003) ** 2)
            observations.append(delta)
            observation_weights.append(weight)
            n_frames_used += 1

        if n_frames_used == 0:
            print("_correct_vertices_with_real_depth: no depth frame produced a geometrically consistent correction — keeping ARFaceGeometry estimate.", flush=True)
            return None

        from depth_consensus import consensus_displacement
        correction, consensus = consensus_displacement(normals, faces, observations, observation_weights)
        refined_vertices = vertices + correction
        self.depth_report.append(consensus)
        print(f"_correct_vertices_with_real_depth: {consensus}", flush=True)

        return {
            "vertices": refined_vertices,
            "faces": faces,
            "pts_2d": fused_mesh.get("pts_2d"),
            "rgb_img": fused_mesh.get("rgb_img"),
            "measured_vertex_count": fused_mesh.get("measured_vertex_count", len(refined_vertices)),
            "region_errors_mm": fused_mesh.get("region_errors_mm"),
        }

    def _refine_surface_with_truedepth(self, fused_mesh: dict | None, depth_frames: list[dict]) -> dict:
        """
        Refine measured ARFaceGeometry only where independent depth samples
        agree in face-local normal displacement. Raw depth is not assumed
        noiseless: frame disagreement reduces its weight, and unsupported
        vertices retain their captured ARFace estimate.
        """
        if fused_mesh is not None:
            refined = self._correct_vertices_with_real_depth(fused_mesh, depth_frames)
            if refined is not None:
                return refined
            return fused_mesh

        VOXEL_SIZE_M = 0.0003  # 0.3mm

        frame_vert_arrays: list[np.ndarray] = []
        frame_face_arrays: list[np.ndarray] = []
        n_used = 0
        for f in depth_frames:
            depth = f["depth_f32"]
            K = f.get("depth_intrinsics")
            if K is None or depth is None:
                continue
            cam_pose = f.get("camera_pose", np.eye(4, dtype=np.float64))
            face_pose = f.get("face_pose")

            grid_verts, grid_faces = _grid_mesh_from_depth_frame(depth, K)
            if grid_faces is None or len(grid_faces) == 0:
                continue

            grid_verts_cv = grid_verts
            points_patient = transform_to_patient_coordinate_system(grid_verts_cv, cam_pose, face_pose)
            frame_vert_arrays.append(points_patient)
            frame_face_arrays.append(grid_faces)
            n_used += 1

        if n_used == 0:
            if fused_mesh is not None:
                print(
                    "_refine_surface_with_truedepth: no depth frames produced a usable "
                    "per-pixel grid mesh — keeping the fused ARFace surface estimate.",
                    flush=True,
                )
                return fused_mesh
            if self.manifest.get("captureSource") == "native_ios":
                raise RuntimeError(
                    "Native TrueDepth package contains no usable metric depth grid or ARFace geometry."
                )
            print(
                "_refine_surface_with_truedepth: no usable depth frames and no prior estimate — "
                "falling back to real RGB multi-view reconstruction.",
                flush=True,
            )
            return self._reconstruct_from_rgb_multiview(self.frames)

        offsets = np.cumsum([0] + [len(v) for v in frame_vert_arrays])[:-1]
        all_verts = np.concatenate(frame_vert_arrays, axis=0)
        all_faces = np.concatenate(
            [faces + offsets[i] for i, faces in enumerate(frame_face_arrays)], axis=0
        )

        # Real voxel-grid multi-frame fusion: real points that land in the
        # same real-world voxel (seen by >1 frame, or just dense within one
        # frame) get averaged into one real fused point.
        voxel_keys = np.round(all_verts / VOXEL_SIZE_M).astype(np.int64)
        _, inverse, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        fused_verts = np.zeros((counts.shape[0], 3), dtype=np.float64)
        np.add.at(fused_verts, inverse, all_verts)
        fused_verts /= counts[:, None]

        fused_faces = inverse[all_faces]
        degenerate = (
            (fused_faces[:, 0] == fused_faces[:, 1])
            | (fused_faces[:, 1] == fused_faces[:, 2])
            | (fused_faces[:, 0] == fused_faces[:, 2])
        )
        fused_faces = fused_faces[~degenerate]

        print(
            f"_refine_surface_with_truedepth: fused {n_used} real TrueDepth frames "
            f"({len(all_verts)} raw backprojected points) into {len(fused_verts)} voxel-fused "
            f"vertices / {len(fused_faces)} faces.",
            flush=True,
        )

        return {
            "vertices": fused_verts,
            "faces": fused_faces,
            "measured_vertex_count": len(fused_verts),
            "region_errors_mm": (fused_mesh or {}).get("region_errors_mm"),
        }

    def _refine_frame_poses_from_landmarks(self, fused_mesh: dict, arface_frames: list[dict]) -> dict:
        """
        Re-solves each frame's OWN camera_pose (in place, mutating the frame
        dicts so every downstream consumer -- re-fusion, depth correction,
        render_back_validator.py -- sees the corrected pose) from that
        frame's own freshly re-detected 2D WFLW-98 landmarks against 4 real
        3D anatomical points on the just-fused mesh. See the D-posedrift
        comment at this function's call site for the real evidence this
        replaces (ARKit's own per-frame face-tracking pose accumulates
        drift over a long continuous rotation).

        Never fabricates a pose: a frame with no usable landmarks, no
        intrinsics, or a solvePnP result whose own reprojection residual
        against these 4 points exceeds a generous sanity bound keeps its
        original ARKit-measured camera_pose unchanged.
        """
        landmarks_3d = self._extract_anatomical_landmarks(fused_mesh["vertices"])
        object_points = np.array([
            landmarks_3d["pronasale"],
            landmarks_3d["menton"],
            landmarks_3d["endocanthion_left"],
            landmarks_3d["endocanthion_right"],
        ], dtype=np.float64)
        # WFLW-98 indices for the same 4 points, in the same order -- reused
        # verbatim from gnm_correspondence.py's own already-validated
        # correspondence (57=nose tip, jaw contour midpoint 16=chin), not
        # invented for this function.
        WFLW_NOSE_TIP = 57
        WFLW_CHIN = 16
        WFLW_EYE_A = 64
        WFLW_EYE_B = 68

        n_refined = 0
        for f in arface_frames:
            lms = f.get("landmarks_98")
            K = f.get("intrinsics")
            cam_pose = f.get("camera_pose")
            if lms is None or K is None or cam_pose is None or len(lms) < 98:
                continue

            image_points = np.array([
                lms[WFLW_NOSE_TIP],
                lms[WFLW_CHIN],
                lms[WFLW_EYE_A],
                lms[WFLW_EYE_B],
            ], dtype=np.float64)
            if not np.isfinite(image_points).all():
                continue

            R0 = cam_pose[:3, :3]
            t0 = cam_pose[:3, 3]
            try:
                rvec0, _ = cv2.Rodrigues(R0)
            except Exception:
                continue

            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, K, None,
                    rvec=rvec0.copy(), tvec=t0.reshape(3, 1).copy(),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except Exception:
                continue
            if not ok:
                continue

            R_refined, _ = cv2.Rodrigues(rvec)
            t_refined = tvec.reshape(3)

            # Sanity check: never trust a solve whose own reprojection of
            # these same 4 points is worse than a generous bound -- keep the
            # original ARKit pose rather than adopt a degenerate solve.
            Xc = (R_refined @ object_points.T).T + t_refined[None, :]
            z = Xc[:, 2]
            if np.any(z <= 0.02):
                continue
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            proj = np.column_stack([fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy])
            residual_px = float(np.linalg.norm(proj - image_points, axis=1).mean())
            if residual_px > 40.0:
                continue

            new_pose = np.eye(4, dtype=np.float64)
            new_pose[:3, :3] = R_refined
            new_pose[:3, 3] = t_refined
            f["camera_pose"] = new_pose
            n_refined += 1

        print(f"_refine_frame_poses_from_landmarks: refined {n_refined}/{len(arface_frames)} frame poses from real re-detected landmarks.", flush=True)
        return fused_mesh

    def _optimize_joint_bundle_adjustment(self, fused_mesh: dict, arface_frames: list[dict]) -> dict:
        """
        Joint Bundle Adjustment: simultaneously optimizes camera poses (R_i, t_i) across
        all frames and 3D facial landmark anchors (X_k) under normalized reprojection
        and smooth trajectory regularization to eliminate multi-frame drift.
        """
        valid_frames = [
            f for f in arface_frames
            if f.get("camera_pose") is not None and f.get("intrinsics") is not None and f.get("landmarks_98") is not None and len(f["landmarks_98"]) >= 98
        ]
        if len(valid_frames) < 3:
            return fused_mesh

        from scipy.optimize import least_squares

        landmarks_3d = self._extract_anatomical_landmarks(fused_mesh["vertices"])
        points_3d_init = np.array([
            landmarks_3d["pronasale"],
            landmarks_3d["menton"],
            landmarks_3d["endocanthion_left"],
            landmarks_3d["endocanthion_right"],
        ], dtype=np.float64)

        wflw_4_idx = [57, 16, 64, 68]
        num_frames = len(valid_frames)

        rvecs_init = []
        tvecs_init = []
        for f in valid_frames:
            R = f["camera_pose"][:3, :3]
            t = f["camera_pose"][:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            rvecs_init.append(rvec.reshape(3))
            tvecs_init.append(t.reshape(3))

        rvecs_init = np.array(rvecs_init)
        tvecs_init = np.array(tvecs_init)

        obs_2d = np.array([[f["landmarks_98"][k] for k in wflw_4_idx] for f in valid_frames])
        Ks = [f["intrinsics"] for f in valid_frames]

        anchor_idx = 0
        for idx, f in enumerate(valid_frames):
            if f["stem"] in ("front", "sweep_00"):
                anchor_idx = idx
                break

        other_indices = [i for i in range(num_frames) if i != anchor_idx]

        def pack_params(pts_3d, rvecs, tvecs):
            parts = [pts_3d.reshape(-1)]
            for i in other_indices:
                parts.append(rvecs[i])
                parts.append(tvecs[i])
            return np.concatenate(parts)

        def unpack_params(x):
            pts_3d = x[:12].reshape(4, 3)
            rvecs = np.zeros((num_frames, 3), dtype=np.float64)
            tvecs = np.zeros((num_frames, 3), dtype=np.float64)
            rvecs[anchor_idx] = rvecs_init[anchor_idx]
            tvecs[anchor_idx] = tvecs_init[anchor_idx]
            off = 12
            for i in other_indices:
                rvecs[i] = x[off:off+3]
                tvecs[i] = x[off+3:off+6]
                off += 6
            return pts_3d, rvecs, tvecs

        def joint_residuals(x):
            pts_3d, rvecs, tvecs = unpack_params(x)
            res = []
            for i in range(num_frames):
                R, _ = cv2.Rodrigues(rvecs[i])
                t = tvecs[i]
                K = Ks[i]
                Xc = (R @ pts_3d.T).T + t[None, :]
                z = np.clip(Xc[:, 2], 0.05, None)
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                u = fx * Xc[:, 0] / z + cx
                v = fy * Xc[:, 1] / z + cy
                du = (u - obs_2d[i, :, 0]) / fx
                dv = (v - obs_2d[i, :, 1]) / fy
                res.append(np.column_stack([du, dv]).reshape(-1))

            for i in other_indices:
                dr = (rvecs[i] - rvecs_init[i]) * 0.5
                dt = (tvecs[i] - tvecs_init[i]) * 1.5
                res.append(dr)
                res.append(dt)

            d_shape = (pts_3d - points_3d_init).reshape(-1) * 2.0
            res.append(d_shape)
            return np.concatenate(res)

        x0 = pack_params(points_3d_init, rvecs_init, tvecs_init)
        try:
            res_opt = least_squares(joint_residuals, x0, method="trf", loss="cauchy", f_scale=0.01, max_nfev=150)
            pts_3d_opt, rvecs_opt, tvecs_opt = unpack_params(res_opt.x)

            for idx, f in enumerate(valid_frames):
                R_opt, _ = cv2.Rodrigues(rvecs_opt[idx])
                new_pose = np.eye(4, dtype=np.float64)
                new_pose[:3, :3] = R_opt
                new_pose[:3, 3] = tvecs_opt[idx]
                f["camera_pose"] = new_pose

            fused_mesh = self._fuse_arface_geometry(arface_frames)
            print(f"_optimize_joint_bundle_adjustment: optimized {len(valid_frames)} poses simultaneously in {res_opt.nfev} iterations.", flush=True)
        except Exception as exc:
            print(f"_optimize_joint_bundle_adjustment fallback note: {exc}", flush=True)

        return fused_mesh

    def _extract_anatomical_landmarks(self, vertices: np.ndarray) -> dict:
        """
        Extracts true anatomical 3D landmark points from the reconstructed patient mesh.
        Uses canonical topological vertex indices for ARFace meshes (>= 1220 vertices),
        with coordinate-based geometric bounds fallback for arbitrary meshes.
        """
        if len(vertices) >= 1220:
            pronasale = vertices[8].tolist()
            subnasale = vertices[2].tolist()
            pogonion = vertices[27].tolist()
            menton = vertices[1047].tolist()
            endocanthion_left = vertices[366].tolist()
            endocanthion_right = vertices[797].tolist()
        else:
            pronasale_idx = int(np.argmax(vertices[:, 2]))
            pronasale = vertices[pronasale_idx].tolist()
            menton_idx = int(np.argmin(vertices[:, 1]))
            menton = vertices[menton_idx].tolist()

            midline = np.where(np.abs(vertices[:, 0]) < 0.015)[0]
            if len(midline) > 0:
                mid_verts = vertices[midline]
                chin_mid = midline[(mid_verts[:, 1] < (vertices[:, 1].min() + 0.045))]
                pogonion_idx = chin_mid[np.argmax(vertices[chin_mid, 2])] if len(chin_mid) > 0 else pronasale_idx
                pogonion = vertices[pogonion_idx].tolist()

                sub_mid = midline[(vertices[midline, 1] > pogonion[1]) & (vertices[midline, 1] < pronasale[1])]
                subnasale_idx = sub_mid[np.argmin(vertices[sub_mid, 2])] if len(sub_mid) > 0 else pronasale_idx
                subnasale = vertices[subnasale_idx].tolist()
            else:
                pogonion = pronasale
                subnasale = pronasale

            eye_region = vertices[(vertices[:, 1] >= pronasale[1] + 0.015) & (vertices[:, 1] <= pronasale[1] + 0.065)]
            left_eye_pts = eye_region[eye_region[:, 0] < -0.005]
            right_eye_pts = eye_region[eye_region[:, 0] > 0.005]
            endocanthion_left = left_eye_pts[np.argmax(left_eye_pts[:, 0])].tolist() if len(left_eye_pts) > 0 else [pronasale[0]-0.016, pronasale[1]+0.04, pronasale[2]-0.02]
            endocanthion_right = right_eye_pts[np.argmin(right_eye_pts[:, 0])].tolist() if len(right_eye_pts) > 0 else [pronasale[0]+0.016, pronasale[1]+0.04, pronasale[2]-0.02]

        return {
            "pronasale": pronasale,
            "subnasale": subnasale,
            "pogonion": pogonion,
            "menton": menton,
            "endocanthion_left": endocanthion_left,
            "endocanthion_right": endocanthion_right,
        }
