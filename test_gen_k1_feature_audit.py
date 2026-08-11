#!/usr/bin/env python3

import unittest

import numpy as np

from audit_gen_k1_features import transform


class FeatureAuditTests(unittest.TestCase):
    def setUp(self):
        phi = np.asarray([-2.1, -0.4, 0.7, 2.4], dtype=np.float32)
        self.names = [
            "logpt", "cosphi", "sinphi", "pdg_gamma", "pdg_mu"
        ]
        self.features = np.column_stack([
            np.arange(4),
            np.cos(phi),
            np.sin(phi),
            [1, 0, 0, 0],
            [0, 1, 1, 1],
        ]).astype(np.float32)
        self.events = [self.features[:2], self.features[2:]]

    def test_individual_shuffle_changes_only_requested_column(self):
        changed = transform(
            self.events,
            self.names,
            "shuffle:logpt",
            np.random.default_rng(8),
        )
        shuffled = np.concatenate(changed)
        self.assertCountEqual(shuffled[:, 0].tolist(), self.features[:, 0].tolist())
        np.testing.assert_array_equal(shuffled[:, 1:], self.features[:, 1:])

    def test_phi_pair_shuffle_preserves_valid_angles(self):
        changed = transform(
            self.events,
            self.names,
            "shuffle:phi_pair",
            np.random.default_rng(9),
        )
        original = sorted(map(tuple, self.features[:, 1:3].tolist()))
        shuffled = sorted(map(tuple, np.concatenate(changed)[:, 1:3].tolist()))
        self.assertEqual(shuffled, original)

    def test_pdg_group_shuffle_preserves_one_hot_rows(self):
        changed = transform(
            self.events,
            self.names,
            "shuffle:pdg_group",
            np.random.default_rng(10),
        )
        np.testing.assert_array_equal(
            np.concatenate(changed)[:, 3:].sum(axis=1), np.ones(4)
        )

    def test_pi_over_two_rotation(self):
        changed = transform(
            self.events,
            self.names,
            "phi_plus_pi_over_2",
            np.random.default_rng(11),
        )
        changed = np.concatenate(changed)
        np.testing.assert_allclose(changed[:, 1], -self.features[:, 2])
        np.testing.assert_allclose(changed[:, 2], self.features[:, 1])
        np.testing.assert_array_equal(changed[:, [0, 3, 4]],
                                      self.features[:, [0, 3, 4]])


if __name__ == "__main__":
    unittest.main()
