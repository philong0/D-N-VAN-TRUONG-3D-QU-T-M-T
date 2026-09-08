import json
import numpy as np
import trimesh

mesh_native = trimesh.load("fixtures/scans/sample_native_ios_package/output/baseline.glb", force="mesh", process=False)
mesh_2d = trimesh.load("fixtures/scans/sample_truedepth_scan/output/baseline.glb", force="mesh", process=False)

v_native = mesh_native.vertices
v_2d = mesh_2d.vertices

diff = np.linalg.norm(v_native - v_2d, axis=1) * 1000.0 # in millimeters

# Nose vertices in shell (around tip/bridge)
nose_vids = [12296, 12279, 10055, 12287] # indices
nose_diff = [diff[vid] for vid in nose_vids if vid < len(diff)]

chin_vids = [3710, 3707, 3708, 3717]
chin_diff = [diff[vid] for vid in chin_vids if vid < len(diff)]

res = {
    "comparison": "Native ARKit/TrueDepth Capture vs Multi-View 2D RGB Photographic Pipeline",
    "vertex_count": len(diff),
    "overall_diff_mm": {
        "mean_mm": float(np.mean(diff)),
        "median_mm": float(np.median(diff)),
        "rms_mm": float(np.sqrt(np.mean(diff**2))),
        "p95_mm": float(np.percentile(diff, 95)),
        "max_mm": float(np.max(diff)),
    },
    "regional_diff_mm": {
        "nose_mean_mm": float(np.mean(nose_diff)) if nose_diff else 0.0,
        "chin_mean_mm": float(np.mean(chin_diff)) if chin_diff else 0.0,
    }
}

print(json.dumps(res, indent=2))
