import json
import os
import shutil
import numpy as np
from pathlib import Path

PKG_DIR = Path("fixtures/scans/sample_native_ios_package")
SRC_DIR = Path("fixtures/scans/sample_truedepth_scan")

os.makedirs(PKG_DIR / "frames", exist_ok=True)
os.makedirs(PKG_DIR / "geometry", exist_ok=True)
os.makedirs(PKG_DIR / "depth", exist_ok=True)
os.makedirs(PKG_DIR / "camera", exist_ok=True)
os.makedirs(PKG_DIR / "quality", exist_ok=True)

# Copy frames
shutil.copy(SRC_DIR / "angle1.png", PKG_DIR / "frames" / "front.jpg")
shutil.copy(SRC_DIR / "angle2.png", PKG_DIR / "frames" / "left_45.jpg")
shutil.copy(SRC_DIR / "angle3.jpg", PKG_DIR / "frames" / "left_profile.jpg")
shutil.copy(SRC_DIR / "angle2.png", PKG_DIR / "frames" / "right_45.jpg")
shutil.copy(SRC_DIR / "angle4.png", PKG_DIR / "frames" / "right_profile.jpg")

# Generate 1220-vertex ARFaceGeometry in meters
v_1220 = (np.random.uniform(-0.07, 0.07, size=(1220, 3))).astype(np.float32)
tris_2304 = np.random.randint(0, 1220, size=(2304, 3)).astype(np.int32)
uvs_1220 = np.random.uniform(0.0, 1.0, size=(1220, 2)).astype(np.float32)

views = [
    ("front", 0.0, 0.0, True),
    ("left_45", -35.0, 0.0, True),
    ("left_profile", -75.0, 0.0, False), # Tracking degrades at high profile
    ("right_45", 35.0, 0.0, True),
    ("right_profile", 75.0, 0.0, False),
]

manifest_frames = []

for view_tag, yaw, pitch, is_tracked in views:
    geom_data = {
        "viewTag": view_tag,
        "vertexCount": 1220,
        "triangleCount": 2304,
        "verticesMeters": v_1220.flatten().tolist(),
        "triangleIndices": tris_2304.flatten().tolist(),
        "textureCoordinates": uvs_1220.flatten().tolist(),
        "isTracked": is_tracked,
    }
    with open(PKG_DIR / "geometry" / f"{view_tag}_geometry.json", "w") as gf:
        json.dump(geom_data, gf, indent=2)
    
    if view_tag == "front":
        depth_raw = (np.random.uniform(360, 440, size=(480, 640))).astype(np.float32)
        depth_raw.tofile(PKG_DIR / "depth" / "front_depth.raw")
    
    manifest_frames.append({
        "viewTag": view_tag,
        "timestamp": 1725001000.0,
        "rgbRelativePath": f"frames/{view_tag}.jpg",
        "depthRelativePath": "depth/front_depth.raw" if view_tag == "front" else None,
        "geometryRelativePath": f"geometry/{view_tag}_geometry.json",
        "cameraRelativePath": "camera/intrinsics.json",
        "isTracked": is_tracked,
        "yawDeg": yaw,
        "pitchDeg": pitch,
    })

# Camera Intrinsics
intrinsics = {
    "fx": 1462.4,
    "fy": 1462.4,
    "cx": 720.0,
    "cy": 960.0,
    "imageWidth": 1440,
    "imageHeight": 1920,
    "lensDistortionCoefficients": [-0.045, 0.082, 0.0, 0.0]
}
with open(PKG_DIR / "camera" / "intrinsics.json", "w") as cf:
    json.dump(intrinsics, cf, indent=2)

# Quality Summary
quality_summary = {
    "totalViewsCaptured": 5,
    "trackedViewsCount": 3,
    "depthAvailable": True,
    "overallAssessment": "pass"
}
with open(PKG_DIR / "quality" / "tracking_quality.json", "w") as qf:
    json.dump(quality_summary, qf, indent=2)

manifest = {
    "packageVersion": "2.0.0",
    "device": {
        "model": "iPhone 14 Pro (iPhone15,2)",
        "systemName": "iOS",
        "systemVersion": "17.5.1",
        "hasTrueDepth": True,
        "arkitSupported": True,
    },
    "patientId": "test-patient-native-ios",
    "sessionId": "test-session-native-001",
    "captureTimestamp": "2026-08-30T05:38:00Z",
    "frames": manifest_frames,
    "qualitySummary": quality_summary
}

with open(PKG_DIR / "scan-manifest.json", "w", encoding="utf-8") as mf:
    json.dump(manifest, mf, indent=2)

print("✓ Created Versioned Native iOS Package Fixture at:", PKG_DIR)
