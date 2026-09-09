"""Regression tests for the native TrueDepth evidence contract.

These use deterministic synthetic measurements only to verify coordinate math
and rejection behaviour; they never claim a synthetic face is clinically valid.
"""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patient_native_fusion import PatientNativeReconstructor, _grid_mesh_from_depth_frame


class NativeMetricContractTests(unittest.TestCase):
    def test_depth_grid_backprojects_metric_portrait_pixels(self):
        depth = np.full((2, 2), 0.4, dtype=np.float32)
        k = np.array([[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]])
        vertices, faces = _grid_mesh_from_depth_frame(depth, k, stride=1)
        self.assertEqual(len(vertices), 4)
        self.assertEqual(len(faces), 2)
        np.testing.assert_allclose(vertices[0], [-0.002, -0.002, 0.4], atol=1e-8)

    def test_native_package_without_metric_depth_fails_loudly(self):
        reconstructor = PatientNativeReconstructor.__new__(PatientNativeReconstructor)
        reconstructor.package_dir = Path("/deterministic-empty-fixture")
        reconstructor.manifest = {"captureSource": "native_ios"}
        reconstructor.frames = []
        with self.assertRaisesRegex(ValueError, "No valid frames"):
            reconstructor.reconstruct_patient_surface()

    def test_native_package_with_missing_view_cannot_fallback(self):
        reconstructor = PatientNativeReconstructor.__new__(PatientNativeReconstructor)
        reconstructor.manifest = {"captureSource": "native_ios"}
        reconstructor.frames = [{"stem": "front"}]
        with self.assertRaisesRegex(ValueError, "Native TrueDepth package is incomplete"):
            reconstructor.reconstruct_patient_surface()


if __name__ == "__main__":
    unittest.main()
