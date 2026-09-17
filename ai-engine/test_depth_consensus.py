import unittest
import numpy as np
from depth_consensus import consensus_displacement, independent_depth_frames

class DepthConsensusTests(unittest.TestCase):
    def setUp(self):
        self.normals = np.tile([0., 0., 1.], (3, 1))
        self.faces = np.array([[0, 1, 2]])

    def test_single_or_disagreeing_observation_cannot_deform_face(self):
        for samples in [np.full((1, 3), .005), np.array([[-.005]*3, [.005]*3])]:
            delta, report = consensus_displacement(self.normals, self.faces, samples, np.ones_like(samples))
            np.testing.assert_array_equal(delta, 0.)
            self.assertEqual(report['supportedVertices'], 0)

    def test_consensus_rejects_outlier_and_moves_only_along_normal(self):
        samples = np.array([[.004]*3, [.004]*3, [.08]*3])
        delta, report = consensus_displacement(self.normals, self.faces, samples, np.ones_like(samples))
        np.testing.assert_array_equal(delta[:, :2], 0.)
        np.testing.assert_allclose(delta[:, 2], .004*2/3)
        self.assertEqual(report['supportedVertices'], 3)

    def test_duplicate_depth_cannot_count_as_two_independent_votes(self):
        a = {'depth_f32': np.ones((3, 3), np.float32), 'stem': 'a'}
        b = {'depth_f32': a['depth_f32'].copy(), 'stem': 'renamed'}
        c = {'depth_f32': a['depth_f32'] + .001, 'stem': 'c'}
        result = independent_depth_frames([a, b, c])
        self.assertEqual([f['stem'] for f in result], ['a', 'c'])

    def test_no_observation_remains_exactly_unchanged(self):
        samples = np.array([[.004, np.nan, .004], [.004, np.nan, .004]])
        delta, report = consensus_displacement(self.normals, self.faces, samples, np.ones_like(samples))
        np.testing.assert_array_equal(delta[1], 0.)
        self.assertTrue(np.isfinite(delta).all())
        self.assertEqual(report['unchangedVertices'], 1)

if __name__ == '__main__':
    unittest.main()
