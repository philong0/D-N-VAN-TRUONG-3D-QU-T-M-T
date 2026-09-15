import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from patient_native_fusion import PatientNativeReconstructor
from patient_texture_baker import unwrap_mesh_uv_xatlas, bake_visibility_aware_texture, export_patient_glb
from render_back_validator import validate_render_back_fidelity

package_dir = Path("/home/ubuntu/dr-vantruong-3d-studio/.data/patients/5006ee37-4f83-48d5-92ab-ef24a640b3e0/scans/ffcc02e3-6934-4cc8-a3c5-8d8460d6cf1a")
output_dir = Path("/tmp/test_recon_5006ee37")
os.makedirs(output_dir, exist_ok=True)

reconstructor = PatientNativeReconstructor(package_dir)
print(f"Loaded {len(reconstructor.frames)} frames.")

for f in reconstructor.frames:
    print(f"Frame {f['stem']}: shape={f['shape']}, has_arface={f['arface_vertices'] is not None}, K=\n{f['intrinsics']}\npose=\n{f['camera_pose']}")

surface_result = reconstructor.reconstruct_patient_surface()
print(f"Surface vertices: {len(surface_result['vertices'])}, faces: {len(surface_result['faces'])}")

vertices = surface_result["vertices"]
faces = surface_result["faces"]

unwrapped_verts, unwrapped_faces, uvs_2d = unwrap_mesh_uv_xatlas(vertices, faces)
final_texture, texture_png_bytes = bake_visibility_aware_texture(
    unwrapped_verts, unwrapped_faces, uvs_2d, reconstructor.frames, tex_size=2048
)

glb_path = output_dir / "baseline.glb"
export_patient_glb(unwrapped_verts, unwrapped_faces, uvs_2d, texture_png_bytes, str(glb_path))

report_path = output_dir / "report.json"
report = validate_render_back_fidelity(glb_path, reconstructor.frames, surface_result, report_path)
print("Validation Report:")
print(json.dumps(report, indent=2))

