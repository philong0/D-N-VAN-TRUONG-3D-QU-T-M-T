"""
Exports patient-specific fitted 3D face mesh and baked HD texture
as a standard, self-contained binary glTF (.glb) file.
Compatible with Three.js GLTFLoader, Blender, and web 3D viewers.
"""

import io
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial

def export_glb_from_fitted(
    fitted_positions: np.ndarray,
    shell_topology: dict,
    face_hd_png_bytes: bytes,
    output_glb_path: str | Path,
) -> str:
    """
    fitted_positions: (17821, 3) or (N, 3) float array in meters or mm.
    shell_topology: dict with {shellVertexIndices, triangles, uvs}
    face_hd_png_bytes: raw PNG bytes of the 2K baked face texture.
    output_glb_path: destination .glb file path.
    """
    output_glb_path = Path(output_glb_path)
    os.makedirs(output_glb_path.parent, exist_ok=True)

    shell_vids = np.array(shell_topology["shellVertexIndices"], dtype=np.int64)
    triangles = np.array(shell_topology["triangles"], dtype=np.int64)
    uvs = np.array(shell_topology["uvs"], dtype=np.float64)
    vertices = fitted_positions[shell_vids].copy()

    # Load texture image and ensure RGB order for glTF standard
    raw_img = cv2.imdecode(np.frombuffer(face_hd_png_bytes, np.uint8), cv2.IMREAD_COLOR)
    if raw_img is not None:
        rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_img)
    else:
        pil_image = Image.open(io.BytesIO(face_hd_png_bytes)).convert("RGB")

    # Invert UV V-coordinate if required for glTF standard (glTF expects origin at top-left)
    uvs_gltf = uvs.copy()
    uvs_gltf[:, 1] = 1.0 - uvs_gltf[:, 1]

    # Create trimesh texture material
    material = SimpleMaterial(
        image=pil_image,
        roughnessFactor=0.5,
        metallicFactor=0.05
    )

    visual = TextureVisuals(
        uv=uvs_gltf,
        material=material
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=triangles,
        visual=visual,
        process=False,
        validate=True
    )

    # Recompute vertex normals for smooth shading
    mesh.vertex_normals = trimesh.geometry.mean_vertex_normals(
        len(mesh.vertices),
        mesh.faces,
        mesh.face_normals
    )

    # Export to binary GLB
    glb_data = mesh.export(file_type="glb")
    with open(output_glb_path, "wb") as f:
        f.write(glb_data)

    return str(output_glb_path)

