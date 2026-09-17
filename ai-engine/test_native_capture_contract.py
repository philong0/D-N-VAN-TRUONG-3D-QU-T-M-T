import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np
from native_capture_contract import face_to_portrait_camera, independent_entries
from native_diagnostics import project, rasterize


class NativeContractTests(unittest.TestCase):
    def test_portrait_projection_matches_rotated_sensor_pixels(self):
        camera = np.eye(4)
        camera[:3, 3] = [.1, .2, .3]
        face = camera.copy()
        face[:3, 3] += [.02, -.03, -.5]
        pose = {"cameraTransformColumnMajor": camera.flatten(order='F'),
                "faceTransformColumnMajor": face.flatten(order='F')}
        vertices = np.array([[.01, .02, .03], [-.02, -.01, .01]])
        sensor = vertices + [.02, -.03, -.5]
        fx, fy, cx, cy, height = 900., 910., 720., 540., 1080
        u = fx*sensor[:, 0]/(-sensor[:, 2])+cx
        v = cy-fy*sensor[:, 1]/(-sensor[:, 2])
        expected = np.column_stack([height-1-v, u])
        K = np.array([[fy, 0, height-1-cy], [0, fx, cx], [0, 0, 1]])
        actual, depth = project(vertices, face_to_portrait_camera(pose), K)
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        # Depth backprojection round-trip recovers the same face-local points.
        rays = np.column_stack([actual, np.ones(len(actual))]) @ np.linalg.inv(K).T
        xyz = rays*depth[:, None]
        inv = np.linalg.inv(face_to_portrait_camera(pose))
        np.testing.assert_allclose(xyz@inv[:3, :3].T + inv[:3, 3], vertices, atol=1e-9)

    def test_clinical_and_unlisted_jpegs_are_never_observations(self):
        with tempfile.TemporaryDirectory() as root:
            image = np.random.default_rng(5).integers(50, 220, (64, 64, 3), dtype=np.uint8)
            for name in ('real.jpg', 'clinical.jpg', 'unlisted.jpg', 'clone.jpg'):
                cv2.imwrite(str(Path(root)/name), image)
            frame = {'view': 'sweep_00', 'rgbFileName': 'real.jpg', 'timestamp': 1., 'isTracked': True}
            m = {'schemaVersion': '3.0.0', 'imageOrientation': 'portrait_cw', 'reconstructionFrames': [frame],
                 'clinicalPhotos': [dict(frame, rgbFileName='clinical.jpg')]}
            entries, report = independent_entries(m, root)
            self.assertEqual(len(entries), 1)
            m['reconstructionFrames'].append(dict(frame, view='sweep_01', timestamp=2., rgbFileName='clone.jpg'))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                independent_entries(m, root)
            legacy = {'frames': m['reconstructionFrames']}
            entries, report = independent_entries(legacy, root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(len(report['rejected']), 1)

    def test_visibility_zbuffer_keeps_nearest_observation(self):
        verts = np.array([[-.1, -.1, 1], [.1, -.1, 1], [0, .1, 1],
                          [-.2, -.2, 2], [.2, -.2, 2], [0, .2, 2]])
        K = np.array([[100, 0, 30], [0, 100, 30], [0, 0, 1]])
        z, _ = rasterize(verts, np.array([[3, 4, 5], [0, 1, 2]]), np.eye(4), K, (60, 60))
        self.assertAlmostEqual(float(z[30, 30]), 1.)

    def test_unobserved_texture_is_not_black_or_counted_as_observed(self):
        from patient_texture_baker import bake_visibility_aware_texture
        vertices = np.array([[-.1, -.1, 1], [0, .1, 1], [.1, -.1, 1],
                             [1, -.1, 1], [1.1, .1, 1], [1.2, -.1, 1]])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        uv = np.array([[.05, .05], [.25, .45], [.45, .05],
                       [.55, .55], [.75, .95], [.95, .55]])
        frame = {'image_bgr': np.full((60, 60, 3), 120, np.uint8),
                 'intrinsics': np.array([[100, 0, 30], [0, 100, 30], [0, 0, 1]]),
                 'camera_pose': np.eye(4)}
        metrics = {}
        texture, _ = bake_visibility_aware_texture(vertices, faces, uv, [frame], 64, metrics)
        self.assertGreater(metrics['observedTexels'], 0)
        self.assertGreater(metrics['unobservedTexels'], 0)
        np.testing.assert_array_equal(metrics['observedMask'].astype(int)
                                      + metrics['unobservedMask'] + metrics['outsideMask'], 1)
        self.assertGreater(texture[metrics['unobservedMask']].min(), 0)
        self.assertLess(metrics['observedFraction'], .9)

if __name__ == '__main__':
    unittest.main()
