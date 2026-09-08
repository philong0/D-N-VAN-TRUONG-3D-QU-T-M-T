# Reconstruction audit — 2026-09-01

## Data selected

- Patient: `634cf7bb-3394-4203-983d-229d36b024df`
- Session: `a1968424-6c79-4ec4-a500-42437d433b8f`
- Real source frames: `front.jpg`, `left_45.jpg`, `left_profile.jpg`,
  `right_45.jpg`, `right_profile.jpg` (five views: 0°, -45°, -90°, +45°, +90°).
- The repository contains no stored TrueDepth depth maps, ARFaceGeometry,
  camera intrinsics, or native scan manifests.

## Measured reconstruction result

The zero-template RGB multi-view worker was run against all five frames in a
staging directory. It produced a 520-vertex / 620-triangle mesh, which is too
sparse to publish. Render-back also failed: mean contour reprojection error
was 60.61 mm; per-view errors were 49.89, 74.57, 59.18, 58.86 and 60.57 mm;
silhouette IoU values were 0.3748, 0.4697, 0.4780, 0.3687 and 0.4441.

A second run against a separate real 28-frame burst session
(`b10e39a0-2d61-44a7-9c48-38966f1b0cec` /
`84e53ffe-c907-45cf-8296-4ea7d55ff060`) was also rejected: 614 vertices / 596
triangles, 161.04 mm mean contour reprojection error, and its bundle-adjustment
solution was physically implausible.

Neither staging GLB was published.

## Runtime safeguards added

- Studio now loads only the provenance-bound `baseline.glb`; it no longer
  generates a photo-derived mesh when that GLB is unavailable.
- The service no longer treats a pre-existing GLB as success when worker JSON
  cannot be parsed.
- Render-back scores are measured from the independently detected 2D landmark
  hull: silhouette IoU and symmetric contour distance, rather than a
  coverage-derived score.
- The density gate requires at least 1,000 vertices. The sparse known RGB
  session is covered by the anti-template test as an expected rejection.
- The former GNM full-head baseline for the selected patient was moved out of
  the public runtime path to `.backups/invalid-generic-baseline-634cf7bb-20260901`.

## Remaining data requirement

The current RGB-only captures do not support a clinically faithful dense 3D
face. A successful publish requires a TrueDepth/ARKit session containing real
depth, intrinsics, camera pose and/or ARFaceGeometry, or a capture sequence
with verified parallax sufficient for dense multi-view triangulation.

## Continuous RGB dense-stage audit

The 28-frame real burst sequence `b10e39a0-2d61-44a7-9c48-38966f1b0cec` /
`84e53ffe-c907-45cf-8296-4ea7d55ff060` was inspected with
`ai-engine/audit_dense_sequence.py`. It contains 9,159 chained ORB tracks,
but only 124 points survive multi-view triangulation. Their bounding extent is
0.615 × 0.668 × 0.556 m, which is not a plausible facial point cloud.

The measured cause is upstream pose degeneracy: the dense branch consumes the
landmark/PnP pose estimate, whose bundle adjustment was rejected as physically
implausible. The next reconstruction implementation step is therefore an
ORB-RANSAC pose graph and bundle adjustment feeding Open3D Poisson/MVS, not
relaxing point or mesh thresholds. `open3d` is installed; `pycolmap`/COLMAP is
not installed and package installation could not resolve from PyPI in this
environment.
