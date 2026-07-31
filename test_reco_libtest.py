#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import reco_libtest_prepare_pools as pools
import submit_reco_libtest_packed as submitter
import train_reco_libtest_pfn as trainer


class RecoSourceSplitTests(unittest.TestCase):
    def test_n840_construction_uses_840_unique_or_20_rotated_files(self):
        self.assertEqual(submitter.files_per_event(840, "U"), 840)
        self.assertEqual(submitter.files_per_event(840, "null_b"), 840)
        self.assertEqual(submitter.files_per_event(840, "R"), 20)

    def test_n1260_construction_uses_1260_unique_or_30_rotated_files(self):
        self.assertEqual(submitter.files_per_event(1260, "U"), 1260)
        self.assertEqual(submitter.files_per_event(1260, "null_b"), 1260)
        self.assertEqual(submitter.files_per_event(1260, "R"), 30)

    def test_n1260_test2000_adds_only_the_missing_events(self):
        original_jobs = submitter.jobs_for_events(800)
        extended_jobs = submitter.jobs_for_events(2000)
        missing_jobs_per_class = extended_jobs - original_jobs

        self.assertEqual(original_jobs, 16)
        self.assertEqual(extended_jobs, 40)
        self.assertEqual(missing_jobs_per_class, 24)
        self.assertEqual(missing_jobs_per_class * 3, 72)

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

    def test_larger_validation_split_preserves_frozen_test_cycles(self):
        cycles = list(range(6654))
        original = pools.split_cycles(cycles, 0.15, 0.25)
        enlarged = pools.split_cycles(cycles, 0.25, 0.25)

        self.assertEqual(
            [len(enlarged[name]) for name in ("train", "val", "test")],
            [3326, 1664, 1664],
        )
        self.assertEqual(enlarged["test"], original["test"])
        self.assertTrue(set(original["val"]).issubset(enlarged["val"]))
        self.assertEqual(
            set(enlarged["val"]) - set(original["val"]),
            set(original["train"]) - set(enlarged["train"]),
        )

class RecoFeatureTests(unittest.TestCase):
    def test_only_charge_derived_category_is_a_model_input(self):
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
            (
                "log_pt", "eta", "sin_phi", "cos_phi", "log_energy",
                "charge", "is_charged",
            ),
        )
        charged = trainer.FEATURES.index("is_charged")
        self.assertEqual(float(features[0, 0, charged]), 0.0)
        self.assertEqual(float(features[0, 1, charged]), 1.0)
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

    def test_store_n_files_must_match_requested_study(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_n.h5"
            with h5py.File(path, "w") as h5:
                h5.create_dataset(
                    "particles",
                    data=np.zeros((1, 1, len(trainer.RAW_FEATURES))),
                )
                h5.create_dataset(
                    "source_file", data=np.asarray([b"file.root"])
                )
                h5.create_dataset("source_event", data=np.asarray([0]))
                h5.attrs["features"] = ",".join(trainer.RAW_FEATURES)
                h5.attrs["n_files"] = 420
            with self.assertRaisesRegex(ValueError, "expected N=840"):
                trainer.load_store(path, expected_n_files=840)

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
    def test_label_permutation_null_is_balanced_fixed_and_split_specific(self):
        labels = np.asarray([0] * 100 + [1] * 100, dtype=np.int32)
        train = trainer.permuted_labels(labels, "train")
        repeated = trainer.permuted_labels(labels, "train")
        validation = trainer.permuted_labels(labels, "val")

        np.testing.assert_array_equal(train, repeated)
        self.assertEqual(np.bincount(train).tolist(), [100, 100])
        self.assertFalse(np.array_equal(train, labels))
        self.assertFalse(np.array_equal(train, validation))

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
