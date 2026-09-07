"""Evidence-only diagnostic for RGB dense reconstruction sequences.

It does not generate geometry or publish assets.  It reports the measured
inputs/outputs of the existing landmark pose stage and the ORB dense stage so
an insufficient sequence cannot be confused with a pipeline success.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from patient_native_fusion import PatientNativeReconstructor
from dense_correspondence import _detect_and_match_consecutive, _triangulate_and_filter_tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    frame_paths = sorted(Path(args.frames_dir).glob("burst_*.*"))
    reconstructor = PatientNativeReconstructor(Path(args.frames_dir))
    reconstructor.frames = []
    for path in frame_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        reconstructor.frames.append({
            "stem": path.stem, "rgb_file": path, "image_bgr": bgr,
            "shape": bgr.shape[:2], "intrinsics": None, "camera_pose": None,
            "face_pose": np.eye(4), "arface_vertices": None,
            "arface_triangles": None, "depth_f32": None,
        })

    report: dict[str, object] = {"inputFrameCount": len(reconstructor.frames)}
    try:
        sparse = reconstructor._reconstruct_from_rgb_multiview(reconstructor.frames)
        report["landmarkPoseStage"] = {
            "status": "completed", "vertices": len(sparse["vertices"]), "triangles": len(sparse["faces"]),
            "posedFrameCount": sum(f.get("camera_pose") is not None for f in reconstructor.frames),
        }
    except Exception as exc:
        report["landmarkPoseStage"] = {"status": "failed", "error": str(exc)}

    posed = [f for f in reconstructor.frames if f.get("camera_pose") is not None and f.get("intrinsics") is not None]
    report["denseInputPoseCount"] = len(posed)
    if len(posed) >= 2:
        tracks = _detect_and_match_consecutive(posed)
        report["orbTrackCount"] = len(tracks)
        report["orbTrackLengthHistogram"] = {
            str(length): sum(len(track) == length for track in tracks.values())
            for length in sorted({len(track) for track in tracks.values()})
        }
        points, _ = _triangulate_and_filter_tracks(tracks, {f["stem"]: f for f in posed})
        report["triangulatedDensePointCount"] = len(points)
        if len(points):
            report["pointCloudBoundsMeters"] = np.ptp(points, axis=0).tolist()
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
