#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import reco_libtest_prepare_pools as pools
import train_reco_libtest_pfn as trainer


class RecoSourceSplitTests(unittest.TestCase):
    def test_split_is_fixed_shuffled_and_source_disjoint(self):
        cycles = list(range(100))
        first = pools.split_cycles(cycles)
        second = pools.split_cycles(cycles)

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "val", "test"})
        self.assertEqual([len(first[name]) for name in ("train", "val", "test")],
                         [60, 15, 25])
        self.assertNotEqual(first["train"], cycles[:60])
        self.assertEqual(set().union(*map(set, first.values())), set(cycles))
        for left, right in (("train", "val"), ("train", "test"),
                            ("val", "test")):
            self.assertTrue(set(first[left]).isdisjoint(first[right]))


class RecoFeatureTests(unittest.TestCase):
    def test_pdg_and_charge_define_particle_categories(self):
        raw = np.zeros((1, 3, len(trainer.RAW_FEATURES)), dtype=np.float32)
        raw[0, 0, trainer.RAW["pt"]] = 1.0
        raw[0, 0, trainer.RAW["pdg"]] = 22
        raw[0, 1, trainer.RAW["pt"]] = 2.0
        raw[0, 1, trainer.RAW["pdg"]] = 211
        raw[0, 1, trainer.RAW["charge"]] = 1.0

        features = trainer.pfn_features(raw)
        photon = trainer.FEATURES.index("is_photon")
        charged = trainer.FEATURES.index("is_charged")
        neutral = trainer.FEATURES.index("is_neutral")
        np.testing.assert_array_equal(features[0, 0, [charged, photon, neutral]],
                                      [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(features[0, 1, [charged, photon, neutral]],
                                      [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(features[0, 2], 0.0)

    def test_store_feature_order_must_match_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset("particles", data=np.zeros((1, 1, 10)))
                h5.create_dataset("source_file", data=np.asarray([b"file.root"]))
                h5.create_dataset("source_event", data=np.asarray([0]))
                h5.attrs["features"] = "pt,eta,phi,energy,mass,charge,type,px,py,pz"
            with self.assertRaisesRegex(ValueError, "unexpected features"):
                trainer.load_store(path)


if __name__ == "__main__":
    unittest.main()
