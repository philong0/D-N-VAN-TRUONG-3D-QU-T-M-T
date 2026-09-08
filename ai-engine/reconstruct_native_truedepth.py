"""
reconstruct_native_truedepth.py

Native iOS TrueDepth & ARKit Patient-Specific 3D Reconstruction Pipeline.
Zero dependency on generic mannequins, template heads, or closed-eye PCA models.

Pipeline:
1. Ingests Multi-Frame Scan Package (frames/, geometry/, depth/, camera/, manifest).
2. Fuses true patient geometry into the Common Patient Coordinate System.
3. Computes distortion-free UV unwrapping via xatlas.
4. Projects authentic multi-view photographic textures.
5. Runs Render-Back Quality Gate and exports baseline.glb + reports.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import trimesh

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from patient_native_fusion import PatientNativeReconstructor
from patient_texture_baker import unwrap_mesh_uv_xatlas, bake_visibility_aware_texture, export_patient_glb
from render_back_validator import validate_render_back_fidelity

RECONSTRUCTION_METHOD = "Patient-Specific Native Multi-Frame Fusion (Zero-Template)"

def reconstruct_native_package(
    package_dir: str | Path,
    patient_id: str,
    session_id: str,
    output_dir: str | Path,
) -> dict:
    t0 = time.time()
    pkg_dir = Path(package_dir)
    out_dir = Path(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting Patient-Specific Reconstruction for patient={patient_id}, session={session_id}...", flush=True)

    # 1. Initialize Native Reconstructor and fuse surface
    reconstructor = PatientNativeReconstructor(pkg_dir)
    surface_result = reconstruct_patient_surface_safe(reconstructor)
    if not surface_result["ok"]:
        return surface_result

    vertices = surface_result["vertices"]
    faces = surface_result["faces"]
    landmarks = surface_result["landmarks"]
    frames = reconstructor.frames

    print(f"Surface fused: {len(vertices)} vertices, {len(faces)} faces from {len(frames)} frames.", flush=True)

    # 2. Authentic Texture Mapping
    # See reconstruct_cli.py's D-noneguard comment — same bug, same fix:
    # `surface_result` always carries these two keys (possibly None), so
    # presence-check alone doesn't tell you whether a single-photo UV
    # shortcut is actually available.
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
            unwrapped_verts, unwrapped_faces, uvs_2d, frames, tex_size=2048
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
    # Keep a geometry-only OBJ beside the GLB for clinical systems that
    # consume OBJ.  It is exported from the exact same measured vertices and
    # triangles as the GLB, never regenerated from photos or a template.
    obj_path = out_dir / "baseline.obj"
    trimesh.Trimesh(vertices=unwrapped_verts, faces=unwrapped_faces, process=False).export(obj_path)

    # 6. Execute Render-Back Verification Gate
    report_path = out_dir / "reconstruction_report.json"
    report = validate_render_back_fidelity(glb_path, frames, surface_result, report_path)

    elapsed_ms = int((time.time() - t0) * 1000)

    # 7. Metadata output
    baseline_meta = {
        "patientId": patient_id,
        "sessionId": session_id,
        "modelVersion": RECONSTRUCTION_METHOD,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processingTimeMs": elapsed_ms,
        "vertexCount": len(unwrapped_verts),
        "triangleCount": len(unwrapped_faces),
        "glbFile": "baseline.glb",
        "objFile": "baseline.obj",
        "textureFile": "face_HD.png",
        "viewsUsed": surface_result["frames_used"],
        "hasNativeARFace": surface_result["has_native_arface"],
        "hasNativeDepth": surface_result["has_native_depth"],
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
        "baselineObjPath": str(obj_path),
        "baselineMeta": baseline_meta,
        "qualityMeta": report,
        "landmarks": landmarks,
    }


def reconstruct_patient_surface_safe(reconstructor: PatientNativeReconstructor) -> dict:
    try:
        res = reconstructor.reconstruct_patient_surface()
        res["ok"] = True
        return res
    except Exception as exc:
        print(f"Surface reconstruction error: {exc}", flush=True)
        return {
            "ok": False,
            "error": str(exc),
            "evaluatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def main():
    parser = argparse.ArgumentParser(description="Patient-Specific Native Reconstruction Pipeline")
    parser.add_argument("--package-dir", type=str, required=True, help="Path to unzipped scan package directory")
    parser.add_argument("--patient-id", type=str, required=True, help="Patient UUID")
    parser.add_argument("--session-id", type=str, required=True, help="Session UUID")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder for baseline.glb and artifacts")
    args = parser.parse_args()

    result = reconstruct_native_package(
        package_dir=args.package_dir,
        patient_id=args.patient_id,
        session_id=args.session_id,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
