#!/usr/bin/env python3
import unittest

import numpy as np

from variable_reuse_common import MotherStore, cycle_split_mothers, sample_definition


class VariableReuseTests(unittest.TestCase):
    def make_store(self):
        store = MotherStore.__new__(MotherStore)
        store.path = "synthetic"
        store.raw = {
            "px": np.array([1.0, 0.0, 2.0], np.float32),
            "py": np.array([0.0, 1.0, 0.0], np.float32),
            "pz": np.array([3.0, 4.0, 5.0], np.float32),
            "E": np.array([4.0, 5.0, 6.0], np.float32),
            "t": np.array([7.0, 8.0, 9.0], np.float32),
            "vx": np.array([10.0, 0.0, 20.0], np.float32),
            "vy": np.array([0.0, 10.0, 0.0], np.float32),
            "vz": np.array([11.0, 12.0, 13.0], np.float32),
            "pdg": np.array([22, 11, 2112], np.int32),
        }
        store.mother_offsets = np.array([0, 2, 3], np.int64)
        store.mother_cycle_ids = np.array([100, 101], np.int64)
        store.mother_local_ids = np.array([0, 0], np.int32)
        store.cycle_ids = np.array([100, 101], np.int64)
        store.cycle_offsets = np.array([0, 1, 2], np.int64)
        store.n_mothers = 2
        store.n_cycles = 2
        return store

    def test_coherent_rotation_and_invariants(self):
        store = self.make_store()
        raw = store.rotated_mothers(
            np.array([0]), np.array([[0.0, np.pi / 2.0]]))
        np.testing.assert_allclose(raw["px"], [1.0, 0.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(raw["py"], [0.0, 1.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(raw["vx"], [10.0, 0.0, 0.0, -10.0], atol=1e-6)
        np.testing.assert_array_equal(raw["pz"], [3.0, 4.0, 3.0, 4.0])
        np.testing.assert_array_equal(raw["pdg"], [22, 11, 22, 11])

    def test_fixed_mother_equivalents_and_unique_sources(self):
        rng = np.random.default_rng(7)
        pool = np.arange(1000)
        for reuse_k in (1, 2, 3, 6, 10, 14, 21, 42):
            definition = sample_definition(rng, pool, reuse_k, 420)
            self.assertEqual(len(definition["mothers"]) * reuse_k, 420)
            self.assertEqual(len(np.unique(definition["mothers"])),
                             len(definition["mothers"]))
            self.assertEqual(definition["angles"].shape,
                             (420 // reuse_k, reuse_k))

    def test_baseline_policy_leaves_k1_unrotated(self):
        definition = sample_definition(
            np.random.default_rng(9), np.arange(20), 1, 10,
            "baseline-unrotated")
        np.testing.assert_array_equal(definition["angles"], 0.0)

    def test_cycle_splits_are_mother_disjoint(self):
        store = self.make_store()
        # Use more cycles so every split is nonempty.
        store.n_cycles = 8
        store.cycle_ids = np.arange(8)
        store.cycle_offsets = np.arange(9)
        splits = cycle_split_mothers(store, (0.5, 0.25, 0.25), seed=3)
        self.assertEqual(len(splits["train"]), 4)
        self.assertEqual(len(splits["val"]), 2)
        self.assertEqual(len(splits["test"]), 2)
        self.assertFalse(set(splits["train"]) & set(splits["val"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))

    def test_select_all_mothers_from_cycles(self):
        store = self.make_store()
        np.testing.assert_array_equal(store.mothers_for_cycles([100, 101]), [0, 1])
        with self.assertRaises(KeyError):
            store.mothers_for_cycles([99])


if __name__ == "__main__":
    unittest.main()
