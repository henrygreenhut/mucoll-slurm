#!/usr/bin/env python3

import unittest

import numpy as np

from audit_gen_k1_phi import fourier_metrics, transform_phi


class PhiInterventionTests(unittest.TestCase):
    def setUp(self):
        phi = np.asarray([-2.1, -0.4, 0.7, 2.4], dtype=np.float32)
        self.names = ["logpt", "cosphi", "sinphi", "loge"]
        self.features = np.column_stack([
            np.arange(4), np.cos(phi), np.sin(phi), np.arange(4) + 10
        ]).astype(np.float32)

    def test_global_rotation_preserves_relative_angles(self):
        changed = transform_phi(
            self.features, self.names, "global_rotation",
            np.random.default_rng(4)
        )
        original = np.unwrap(np.arctan2(self.features[:, 2], self.features[:, 1]))
        rotated = np.unwrap(np.arctan2(changed[:, 2], changed[:, 1]))
        np.testing.assert_allclose(
            np.diff(rotated), np.diff(original), atol=2e-6
        )
        np.testing.assert_array_equal(changed[:, [0, 3]], self.features[:, [0, 3]])

    def test_phi_shuffle_preserves_angle_pairs(self):
        changed = transform_phi(
            self.features, self.names, "shuffle_phi",
            np.random.default_rng(8)
        )
        original = sorted(map(tuple, self.features[:, [1, 2]].tolist()))
        shuffled = sorted(map(tuple, changed[:, [1, 2]].tolist()))
        self.assertEqual(shuffled, original)
        np.testing.assert_array_equal(changed[:, [0, 3]], self.features[:, [0, 3]])

    def test_uniform_phi_stays_on_unit_circle(self):
        changed = transform_phi(
            self.features, self.names, "uniform_phi",
            np.random.default_rng(12)
        )
        np.testing.assert_allclose(
            changed[:, 1] ** 2 + changed[:, 2] ** 2, 1.0, atol=1e-6
        )

    def test_fourier_metrics(self):
        metrics = fourier_metrics(self.features, self.names, maximum=2)
        self.assertEqual(metrics["particles"], 4)
        self.assertAlmostEqual(
            metrics["r1"], np.hypot(metrics["cos1"], metrics["sin1"])
        )


if __name__ == "__main__":
    unittest.main()
