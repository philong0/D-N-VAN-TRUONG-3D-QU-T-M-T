import json
import os
import shutil
import numpy as np
from pathlib import Path

FIXTURE_DIR = Path("fixtures/scans/sample_truedepth_package")
SRC_DIR = Path("fixtures/scans/sample_truedepth_scan")

os.makedirs(FIXTURE_DIR, exist_ok=True)

# Copy RGB frames
shutil.copy(SRC_DIR / "angle1.png", FIXTURE_DIR / "front_rgb.jpg")
shutil.copy(SRC_DIR / "angle2.png", FIXTURE_DIR / "left_45_rgb.jpg")
shutil.copy(SRC_DIR / "angle3.jpg", FIXTURE_DIR / "left_profile_rgb.jpg")
shutil.copy(SRC_DIR / "angle2.png", FIXTURE_DIR / "right_45_rgb.jpg")
shutil.copy(SRC_DIR / "angle4.png", FIXTURE_DIR / "right_profile_rgb.jpg")

# Generate synthetic 1220-vertex ARFaceGeometry vertices in meters (head scale ~0.2m)
vertices_1220 = np.random.uniform(-0.08, 0.08, size=(1220, 3)).astype(np.float32)
triangles_2304 = np.random.randint(0, 1220, size=(2304, 3)).astype(np.int32)
uvs_1220 = np.random.uniform(0.0, 1.0, size=(1220, 2)).astype(np.float32)

# Generate synthetic 16-bit millimeter depth buffer
depth_raw = (np.random.uniform(350, 450, size=(480, 640))).astype(np.uint16)
depth_raw.tofile(FIXTURE_DIR / "front_depth.raw")

views = ["front", "left_45", "left_profile", "right_45", "right_profile"]
yaws = [0.0, -45.0, -90.0, 45.0, 90.0]

frames = []
for view, yaw in zip(views, yaws):
    rad = np.radians(yaw)
    cos_y = float(np.cos(rad))
    sin_y = float(np.sin(rad))
    
    pose_matrix = [
        [cos_y, 0.0, sin_y, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sin_y, 0.0, cos_y, -0.45], # 45cm camera distance
        [0.0, 0.0, 0.0, 1.0]
    ]
    
    frame = {
        "view": view,
        "timestamp": 1725000000.0,
        "rgbFileName": f"{view}_rgb.jpg",
        "depthFileName": "front_depth.raw" if view == "front" else None,
        "depthWidth": 640 if view == "front" else None,
        "depthHeight": 480 if view == "front" else None,
        "intrinsics": {
            "fx": 1450.5,
            "fy": 1450.5,
            "cx": 720.0,
            "cy": 960.0,
            "imageWidth": 1440,
            "imageHeight": 1920
        },
        "pose": {
            "columns": pose_matrix,
            "translationMeters": {"x": 0.0, "y": 0.0, "z": -0.45},
            "eulerRotationDeg": {"pitch": 0.0, "yaw": yaw, "roll": 0.0}
        },
        "geometry": {
            "vertexCount": 1220,
            "triangleCount": 2304,
            "verticesMeters": vertices_1220.flatten().tolist(),
            "triangleIndices": triangles_2304.flatten().tolist(),
            "textureCoordinates": uvs_1220.flatten().tolist(),
            "blendShapes": {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0, "jawOpen": 0.0}
        }
    }
    frames.append(frame)

manifest = {
    "schemaVersion": "1.0.0",
    "deviceModel": "iPhone15,2 (iPhone 14 Pro)",
    "systemVersion": "iOS 17.5.1",
    "hasTrueDepth": True,
    "hasLiDAR": True,
    "patientId": "test-patient-truedepth",
    "sessionId": "test-session-truedepth-001",
    "capturedAt": "2026-08-30T05:36:00Z",
    "frames": frames
}

with open(FIXTURE_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("✓ Created Sample TrueDepth Package Fixture:", FIXTURE_DIR)
