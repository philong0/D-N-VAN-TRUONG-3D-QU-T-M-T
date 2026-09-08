"""
reconstruct_cli.py

Patient-Specific 3D Face Reconstruction CLI.
Zero dependency on generic mannequins, template heads, or closed-eye PCA models.

Reads multi-angle patient frames, fuses authentic patient geometry,
unwraps UV via xatlas, bakes visibility-aware textures, and runs Render-Back Validation Gate.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from patient_native_fusion import PatientNativeReconstructor
from patient_texture_baker import unwrap_mesh_uv_xatlas, bake_visibility_aware_texture, export_patient_glb
from render_back_validator import validate_render_back_fidelity

RECONSTRUCTION_METHOD = "Patient-Specific Multi-Frame Reconstruction (Zero-Template)"


def reconstruct_patient_scan(
    image_paths: dict[str, str],
    patient_id: str,
    session_id: str,
    output_dir: str | Path,
    burst_dir: str | Path | None = None,
) -> dict:
    t0 = time.time()
    out_dir = Path(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # D-burstload — a burst session has many (sequenceIndex-ordered)
    # `burst_NNNN.<ext>` frames instead of the 5 fixed-name views; loaded
    # here directly (bypassing PatientNativeReconstructor's named-view
    # package discovery, which doesn't know about this naming) IN CAPTURE
    # ORDER — dense_correspondence.py's consecutive-frame ORB matching
    # depends on that real temporal order to know which frames actually
    # overlap.
    from detect_pose import detect_pose

    images = {}
    if burst_dir is not None:
        burst_path = Path(burst_dir)
        # 1. Check for named origin files first (highest quality):
        origin_map = [
            (["front_origin", "front", "angle1_origin", "angle1"], "angle1"),
            (["left45_origin", "left45", "left_45_origin", "left_45", "angle2_origin", "angle2"], "angle2"),
            (["left80_origin", "left90_origin", "left80", "left90", "left_profile", "angle3_origin", "angle3"], "angle3"),
            (["right45_origin", "right45", "right_45_origin", "right_45", "angle4_origin", "angle4"], "angle4"),
        ]
        for name_list, slot in origin_map:
            for name in name_list:
                matches = list(burst_path.glob(f"{name}.*"))
                if matches and slot not in images:
                    im = cv2.imread(str(matches[0]))
                    if im is not None:
                        images[slot] = im
                        print(f"[BurstLoad] Found origin photo for {slot}: {matches[0].name}", flush=True)
                        break

        # 2. If any slots missing, analyze all burst frames by real measured pose:
        all_burst_files = sorted(set(list(burst_path.glob("burst_*.*")) + list(burst_path.glob("*.jpg")) + list(burst_path.glob("*.png"))))
        if all_burst_files and len(images) < 4:
            analyzed = []
            for f in all_burst_files:
                im = cv2.imread(str(f))
                if im is None:
                    continue
                res = detect_pose(im, is_profile_view=False)
                if res.get("has_face"):
                    analyzed.append({"img": im, "yaw": float(res.get("yaw", 0.0)), "file": f.name})

            if analyzed:
                if "angle1" not in images:
                    front_candidates = [x for x in analyzed if abs(x["yaw"]) <= 3.0]
                    if front_candidates:
                        # Prefer earliest calm frontal frame with neutral gaze
                        best_f = front_candidates[0]
                    else:
                        best_f = min(analyzed, key=lambda x: abs(x["yaw"]))
                    images["angle1"] = best_f["img"]
                    print(f"[Selection] angle1 <- {best_f['file']} (yaw={best_f['yaw']:.1f}°)", flush=True)

                if "angle2" not in images:
                    left_pool = [x for x in analyzed if x["yaw"] < -12.0]
                    if left_pool:
                        best_l = min(left_pool, key=lambda x: abs(x["yaw"] - (-45.0)))
                        images["angle2"] = best_l["img"]
                        print(f"[Selection] angle2 <- {best_l['file']} (yaw={best_l['yaw']:.1f}°)", flush=True)

                if "angle3" not in images:
                    # ONLY populate angle3 if a genuine profile (yaw <= -50°) exists!
                    profile_pool = [x for x in analyzed if x["yaw"] <= -50.0]
                    if profile_pool:
                        best_p = min(profile_pool, key=lambda x: abs(x["yaw"] - (-80.0)))
                        images["angle3"] = best_p["img"]
                        print(f"[Selection] angle3 (profile) <- {best_p['file']} (yaw={best_p['yaw']:.1f}°)", flush=True)
                    else:
                        print(f"[Selection] angle3 omitted (no genuine profile <= -50° in burst sweep)", flush=True)

                if "angle4" not in images:
                    right_pool = [x for x in analyzed if x["yaw"] > 12.0]
                    if right_pool:
                        best_r = min(right_pool, key=lambda x: abs(x["yaw"] - 45.0))
                        images["angle4"] = best_r["img"]
                        print(f"[Selection] angle4 <- {best_r['file']} (yaw={best_r['yaw']:.1f}°)", flush=True)
    else:
        slot_map = {
            "front": "angle1",
            "angle1": "angle1",
            "left45": "angle2",
            "left_45": "angle2",
            "angle2": "angle2",
            "left90": "angle3",
            "left_profile": "angle3",
            "right90": "angle3",
            "right_profile": "angle3",
            "angle3": "angle3",
            "right45": "angle4",
            "right_45": "angle4",
            "angle4": "angle4",
        }
        for tag, fpath in image_paths.items():
            if os.path.exists(fpath):
                img = cv2.imread(str(fpath))
                slot = slot_map.get(tag, tag)
                if img is not None and slot in ("angle1", "angle2", "angle3", "angle4"):
                    images[slot] = img

    # Fallback to single/available image if not all slots present
    if "angle1" not in images and images:
        images["angle1"] = next(iter(images.values()))

    # D-notemplate — bridge the selected real per-angle frames (`images`,
    # angle1..4 -> real captured BGR image, chosen above from real measured
    # yaw/pitch, never a template) into a PatientNativeReconstructor so
    # `_run_reconstruction` below runs the actual zero-template pipeline
    # (multi-view SfM / ARFace / TrueDepth fusion). This runtime must NEVER
    # call `reconstruct_gnm_fullhead` — that generic-template pipeline was
    # wired in here twice already (both times reverted, see D-notemplate in
    # earlier backups and test_anti_template.py's stricter direct-import
    # check) and both times it silently replaced this patient's own
    # captured geometry with a template deformed toward their photos,
    # bypassing the render-back QC gate entirely.
    reconstructor = PatientNativeReconstructor(Path(out_dir))
    reconstructor.frames = []
    for slot, bgr in images.items():
        reconstructor.frames.append({
            "stem": slot,
            "rgb_file": Path(out_dir) / f"{slot}.jpg",
            "image_bgr": bgr,
            "shape": bgr.shape[:2],
            "intrinsics": None,
            "camera_pose": np.eye(4, dtype=np.float64),
            "face_pose": np.eye(4, dtype=np.float64),
            "arface_vertices": None,
            "arface_triangles": None,
            "depth_f32": None,
        })

    return _run_reconstruction(reconstructor, patient_id, session_id, out_dir, t0)


def _run_reconstruction(reconstructor: PatientNativeReconstructor, patient_id: str, session_id: str, out_dir: Path, t0: float) -> dict:
    if not reconstructor.frames:
        return {
            "ok": False,
            "error": "Không có ảnh hợp lệ nào được cung cấp.",
            "evaluatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # 1. Reconstruct true patient surface
    surface_result = reconstructor.reconstruct_patient_surface()
    vertices = surface_result["vertices"]
    faces = surface_result["faces"]
    landmarks = surface_result["landmarks"]
    frames = reconstructor.frames

    # 2. Authentic Texture Mapping
    # D-noneguard — `surface_result` always CARRIES these two keys (see
    # `reconstruct_patient_surface`'s own `fused_mesh.get("pts_2d")` /
    # `.get("rgb_img")`), so `"pts_2d" in surface_result` is True even when
    # the real reconstruction path (multi-view SfM, ARFace fusion, TrueDepth
    # fusion) legitimately has neither — none of those real, multi-frame
    # paths produce a single "the photo" to paste as UV; only a would-be
    # single-image shortcut ever would. Checking VALUE truthiness instead of
    # key presence is what actually distinguishes "no single-photo UV
    # shortcut available" from "it's available" — confirmed necessary by a
    # real crash (`rgb_img.shape` on `None`) hitting this exact branch
    # during E2E testing of the real 5-view SfM path.
    if surface_result.get("pts_2d") is not None and surface_result.get("rgb_img") is not None:
        pts_2d = surface_result["pts_2d"]
        rgb_img = surface_result["rgb_img"]
        h, w = rgb_img.shape[:2]
        uvs_2d = np.column_stack([
            pts_2d[:, 0] / float(w),
            1.0 - (pts_2d[:, 1] / float(h))
        ])
        unwrapped_verts = vertices
        unwrapped_faces = faces
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        _, png_buf = cv2.imencode(".png", bgr_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        texture_png_bytes = png_buf.tobytes()
    else:
        unwrapped_verts, unwrapped_faces, uvs_2d = unwrap_mesh_uv_xatlas(vertices, faces)
        _, texture_png_bytes = bake_visibility_aware_texture(
            unwrapped_verts, unwrapped_faces, uvs_2d, frames, tex_size=1024
        )

    # 4. Save raw arrays & assets
    unwrapped_verts.astype("<f4").tofile(out_dir / "positions.f32")
    with open(out_dir / "face_HD.png", "wb") as f:
        f.write(texture_png_bytes)

    with open(out_dir / "landmarks.json", "w") as f:
        json.dump(landmarks, f, indent=2)

    # 5. Export binary glTF (baseline.glb)
    glb_path = out_dir / "baseline.glb"
    export_patient_glb(unwrapped_verts, unwrapped_faces, uvs_2d, texture_png_bytes, str(glb_path))

    # 6. Execute Render-Back Verification Gate
    report_path = out_dir / "reconstruction_report.json"
    report = validate_render_back_fidelity(glb_path, frames, surface_result, report_path)

    elapsed_ms = int((time.time() - t0) * 1000)

    baseline_meta = {
        "patientId": patient_id,
        "sessionId": session_id,
        "modelVersion": RECONSTRUCTION_METHOD,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processingTimeMs": elapsed_ms,
        "vertexCount": len(unwrapped_verts),
        "triangleCount": len(unwrapped_faces),
        "glbFile": "baseline.glb",
        "textureFile": "face_HD.png",
        "viewsUsed": surface_result["frames_used"],
        "reconstructionStatus": report["reconstructionStatus"],
    }

    with open(out_dir / "baseline.json", "w") as f:
        json.dump(baseline_meta, f, indent=2)

    with open(out_dir / "quality.json", "w") as f:
        json.dump({
            "overall": "pass" if report["reconstructionStatus"] == "completed" else "fail",
            "registrationErrorMm": report["registrationErrorMm"],
            "depthCoverage": report["depthCoverage"],
            "textureCoverage": report["textureCoverage"],
            "anatomicalRegionErrors": report["anatomicalRegionErrors"],
        }, f, indent=2)

    return {
        "ok": True,
        "patientId": patient_id,
        "sessionId": session_id,
        "baselineGlbPath": str(glb_path),
        "baselineMeta": baseline_meta,
        "qualityMeta": report,
        "landmarks": landmarks,
    }


def main():
    parser = argparse.ArgumentParser(description="Patient-Specific Reconstruction CLI")
    parser.add_argument("--patient-id", type=str, required=True, help="Patient UUID")
    parser.add_argument("--session-id", type=str, required=True, help="Session UUID")
    parser.add_argument("--output-dir", type=str, required=True, help="Output destination folder")
    parser.add_argument("--front", type=str, help="Path to frontal RGB frame")
    parser.add_argument("--left45", type=str, help="Path to -45 deg RGB frame")
    parser.add_argument("--left90", type=str, help="Path to -90 deg profile RGB frame")
    parser.add_argument("--right45", type=str, help="Path to +45 deg RGB frame")
    parser.add_argument("--right90", type=str, help="Path to +90 deg profile RGB frame")
    parser.add_argument("--angle1", type=str, help="Path to angle 1 (frontal)")
    parser.add_argument("--angle2", type=str, help="Path to angle 2 (left 45)")
    parser.add_argument("--angle3", type=str, help="Path to angle 3 (left 90)")
    parser.add_argument("--angle4", type=str, help="Path to angle 4 (right 45)")
    parser.add_argument("--burst-dir", type=str, help="Directory of burst_NNNN.<ext> continuous-capture frames")
    args = parser.parse_args()

    image_paths = {}
    if args.front: image_paths["front"] = args.front
    if args.left45: image_paths["left_45"] = args.left45
    if args.left90: image_paths["left_profile"] = args.left90
    if args.right45: image_paths["right_45"] = args.right45
    if args.right90: image_paths["right_profile"] = args.right90
    if args.angle1: image_paths["angle1"] = args.angle1
    if args.angle2: image_paths["angle2"] = args.angle2
    if args.angle3: image_paths["angle3"] = args.angle3
    if args.angle4: image_paths["angle4"] = args.angle4

    result = reconstruct_patient_scan(
        image_paths=image_paths,
        patient_id=args.patient_id,
        session_id=args.session_id,
        output_dir=args.output_dir,
        burst_dir=args.burst_dir,
    )
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
