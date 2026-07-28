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
    def test_particle_category_indicators_are_not_model_inputs(self):
        raw = np.zeros((1, 3, len(trainer.RAW_FEATURES)), dtype=np.float32)
        raw[0, 0, trainer.RAW["pt"]] = 1.0
        raw[0, 0, trainer.RAW["energy"]] = 1.0
        raw[0, 0, trainer.RAW["pdg"]] = 22
        raw[0, 1, trainer.RAW["pt"]] = 2.0
        raw[0, 1, trainer.RAW["energy"]] = 2.0
        raw[0, 1, trainer.RAW["pdg"]] = 211
        raw[0, 1, trainer.RAW["charge"]] = 1.0

        features = trainer.pfn_features(raw)
        self.assertEqual(
            trainer.FEATURES,
            ("log_pt", "eta", "sin_phi", "cos_phi", "log_energy", "charge"),
        )
        self.assertNotIn("is_charged", trainer.FEATURES)
        np.testing.assert_array_equal(features[0, 2], 0.0)
        self.assertNotIn("is_photon", trainer.FEATURES)
        self.assertNotIn("is_neutral", trainer.FEATURES)

    def test_continuous_features_are_direct_and_padding_stays_zero(self):
        raw = np.zeros((1, 2, len(trainer.RAW_FEATURES)), dtype=np.float32)
        raw[0, 0, trainer.RAW["pt"]] = np.exp(2.0)
        raw[0, 0, trainer.RAW["eta"]] = 7.0
        raw[0, 0, trainer.RAW["phi"]] = np.pi / 2
        raw[0, 0, trainer.RAW["energy"]] = np.exp(3.0)
        raw[0, 0, trainer.RAW["charge"]] = 5.0

        features = trainer.pfn_features(raw)
        expected = {
            "log_pt": 2.0,
            "eta": 7.0,
            "sin_phi": 1.0,
            "cos_phi": 0.0,
            "log_energy": 3.0,
            "charge": 5.0,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(
                float(features[0, 0, trainer.FEATURES.index(name)]),
                value,
                places=6,
            )
        np.testing.assert_array_equal(features[0, 1], 0.0)

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

    def test_store_provenance_fingerprints_data_and_track_links(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset(
                    "particles", data=np.zeros((2, 3, len(trainer.RAW_FEATURES)))
                )
                h5.create_dataset(
                    "source_file",
                    data=np.asarray([b"a.root", b"b.root"]),
                )
                h5.create_dataset("source_event", data=np.asarray([0, 0]))
                h5.create_dataset("n_particles", data=np.asarray([1, 2]))
                h5.create_dataset("n_tracks", data=np.asarray([1, 1]))
                h5.create_dataset("n_clusters", data=np.asarray([2, 3]))
                h5.create_dataset("pfo_track_links", data=np.asarray([1, 2]))
                h5.attrs["features"] = ",".join(trainer.RAW_FEATURES)
                h5.attrs["pfo_collection"] = "PandoraPFOs"

            first = trainer.store_provenance(path)
            second = trainer.store_provenance(path)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["particles_shape"], [2, 3, 10])
            self.assertEqual(first["source_root_files"], 2)
            self.assertEqual(
                first["collection_statistics"]["pfo_track_links"]["total"], 3
            )

    def test_trackfix_guard_rejects_unlinked_stores(self):
        dataset = {
            "stores": {
                "train": {
                    "U": {"collection_statistics": {
                        "pfo_track_links": {"total": 0}
                    }}
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "no PFO-track links"):
            trainer.require_pfo_track_links(dataset)


class RecoRecipeTests(unittest.TestCase):
    def test_stabilized_recipes_only_differ_by_dropout(self):
        plain = trainer.recipe_config("stabilized", steps_per_epoch=125)
        dropout = trainer.recipe_config(
            "stabilized_dropout", steps_per_epoch=125)

        self.assertEqual(plain["learning_rate"], 1e-4)
        self.assertEqual(plain["warmup_steps"], 125)
        self.assertEqual(plain["decay_steps"], 3750)
        self.assertEqual(plain["min_learning_rate"], 1e-6)
        self.assertEqual(plain["jit_compile"], False)
        self.assertEqual(plain["clipnorm"], 0.0)
        self.assertEqual(plain["f_dropout"], 0.0)
        self.assertEqual(dropout["f_dropout"], 0.1)
        self.assertEqual(
            {key: value for key, value in plain.items()
             if key != "f_dropout"},
            {key: value for key, value in dropout.items()
             if key != "f_dropout"},
        )

    def test_baseline_retains_energyflow_internal_compile(self):
        baseline = trainer.recipe_config("baseline", steps_per_epoch=125)
        self.assertFalse(baseline["explicit_compile"])
        self.assertIsNone(baseline["jit_compile"])
        self.assertEqual(baseline["learning_rate"], 1e-3)
        self.assertEqual(baseline["warmup_steps"], 0)
        self.assertEqual(baseline["decay_steps"], 0)
        self.assertEqual(baseline["f_dropout"], 0.0)


if __name__ == "__main__":
    unittest.main()
