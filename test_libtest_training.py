#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

import numpy as np

import libtest_common as lc
from pfn_libtest_train import (
    balanced_chunks,
    binary_cross_entropy,
    initial_state,
    load_or_create_validation_units,
    update_validation_state,
    write_or_validate_config,
)


class LibtestNormalizationTests(unittest.TestCase):
    def test_streaming_moments_remain_correct_beyond_float32_integer_limit(self):
        # Reusing one chunk keeps the test small while the logical particle
        # count exceeds 2**24, where the previous axis-zero float32 reduction
        # stopped incrementing one-hot counts.
        chunk = np.ones((1_000_000, 2), dtype=np.float32)
        mean, std = lc.compute_norm_stats([chunk] * 18)
        np.testing.assert_array_equal(mean, [1.0, 1.0])
        np.testing.assert_array_equal(std, [1.0, 1.0])

    def test_streaming_moments_match_float64_reference(self):
        rng = np.random.default_rng(17)
        arrays = [rng.normal(size=(size, 4)).astype(np.float32)
                  for size in (11, 37, 5)]
        mean, std = lc.compute_norm_stats(arrays)
        reference = np.concatenate(arrays).astype(np.float64)
        np.testing.assert_allclose(mean, reference.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(std, reference.std(axis=0), rtol=1e-6)

    def test_corrupted_pdg_onehot_cache_is_rejected(self):
        names = ["logpt", "theta", "cosphi", "sinphi"] + lc.PDG_ONEHOT
        payload = {
            "names": names,
            "mean": [0.0, 0.0, 0.0, 0.0,
                     0.0672325, 0.0672325, 0.0057, 0.00005, 0.0002],
            "std": [1.0] * len(names),
            "latent_scale": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "norm_stats.json")
            with open(path, "w") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "float32 reduction overflow"):
                lc.load_norm_stats(path)

    def test_valid_pdg_onehot_means_are_accepted(self):
        names = ["logpt", "theta", "cosphi", "sinphi"] + lc.PDG_ONEHOT
        mean = np.asarray([0.0, 0.0, 0.0, 0.0,
                           0.47, 0.52, 0.006, 0.001, 0.003], np.float32)
        lc.validate_norm_stats(mean, np.ones_like(mean), names)


class EarlyStoppingTests(unittest.TestCase):
    def test_minimum_epoch_floor_defers_early_stopping(self):
        state = {"epoch": 31, "best_epoch": 10}
        self.assertFalse(lc.should_early_stop(
            state, patience=20, min_epochs=80))
        state["epoch"] = 80
        self.assertTrue(lc.should_early_stop(
            state, patience=20, min_epochs=80))


class ValidationLossTests(unittest.TestCase):
    def test_binary_cross_entropy_matches_two_class_definition(self):
        labels = np.asarray([0, 1], dtype=np.int32)
        scores = np.asarray([0.25, 0.75], dtype=np.float32)
        self.assertAlmostEqual(binary_cross_entropy(labels, scores), -np.log(0.75))

    def test_patience_still_applies_after_floor(self):
        state = {"epoch": 80, "best_epoch": 70}
        self.assertFalse(lc.should_early_stop(
            state, patience=20, min_epochs=80))
        state["epoch"] = 91
        self.assertTrue(lc.should_early_stop(
            state, patience=20, min_epochs=80))


class OptimizerConfigurationTests(unittest.TestCase):
    class Optimizers:
        class Adam:
            def __init__(self, learning_rate, **kwargs):
                self.learning_rate = learning_rate
                self.kwargs = kwargs

    class Schedules:
        class PolynomialDecay:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class CosineDecay:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    def make_optimizer(self, jit_compile):
        return lc.make_optimizer(
            self.Optimizers, self.Schedules, lr=1e-3,
            warmup_steps=0, clipnorm=0, jit_compile=jit_compile)

    def test_optimizer_jit_is_explicitly_disabled(self):
        optimizer = self.make_optimizer(False)
        self.assertIs(optimizer.kwargs["jit_compile"], False)

    def test_optimizer_jit_follows_requested_model_jit(self):
        optimizer = self.make_optimizer(True)
        self.assertIs(optimizer.kwargs["jit_compile"], True)

    def test_cosine_schedule_contains_warmup_and_floor(self):
        optimizer = lc.make_optimizer(
            self.Optimizers, self.Schedules, lr=3e-4,
            warmup_steps=250, clipnorm=0, decay_steps=5000, min_lr=1e-6)
        schedule = optimizer.learning_rate
        self.assertIsInstance(schedule, self.Schedules.CosineDecay)
        self.assertEqual(schedule.kwargs["warmup_steps"], 250)
        self.assertEqual(schedule.kwargs["decay_steps"], 5000)
        self.assertAlmostEqual(schedule.kwargs["warmup_target"], 3e-4)
        self.assertAlmostEqual(schedule.kwargs["alpha"], 1e-6 / 3e-4)


class SourceSplitTests(unittest.TestCase):
    def test_saved_split_is_shuffled_disjoint_and_reused(self):
        cycles = np.arange(1000, 1100)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "source_split.npz")
            first = lc.load_or_create_cycle_split(
                path, cycles, (0.5, 0.25, 0.25), seed=17)
            second = lc.load_or_create_cycle_split(
                path, cycles, (0.5, 0.25, 0.25), seed=999)
            for name in ("train", "val", "test"):
                np.testing.assert_array_equal(first[name], second[name])
            self.assertFalse(np.array_equal(first["train"], cycles[:50]))
            combined = np.concatenate(list(first.values()))
            np.testing.assert_array_equal(np.sort(combined), cycles)

    def test_validation_units_are_fixed_and_unique_within_event(self):
        cycles = np.arange(100)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "validation_units.npz")
            first = load_or_create_validation_units(
                path, cycles, (12, 3), n_units=20, seed=8)
            second = load_or_create_validation_units(
                path, cycles, (12, 3), n_units=20, seed=999)
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)
                self.assertTrue(all(len(np.unique(row)) == len(row)
                                    for row in a))


class BalancedBatchTests(unittest.TestCase):
    def test_each_batch_has_equal_classes(self):
        definitions = [
            [(class_id, np.asarray([100 * class_id + i]))
             for i in range(12)]
            for class_id in (0, 1)
        ]
        chunks = balanced_chunks(
            definitions, batch_size=4, rng=np.random.default_rng(5))
        self.assertEqual(len(chunks), 6)
        for chunk in chunks:
            self.assertEqual([class_id for class_id, _ in chunk].count(0), 2)
            self.assertEqual([class_id for class_id, _ in chunk].count(1), 2)


class RunMetadataTests(unittest.TestCase):
    def test_only_runtime_config_may_change_on_resume(self):
        base = {
            "config_schema_version": 2, "label": "run", "lr": 1e-4,
            "max_minutes": 2, "progress_every": 25,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.json")
            write_or_validate_config(path, base)
            changed_runtime = dict(base, max_minutes=90, progress_every=10)
            write_or_validate_config(path, changed_runtime)
            with self.assertRaisesRegex(SystemExit, "changed scientific"):
                write_or_validate_config(path, dict(base, lr=3e-4))

    def test_auc_and_loss_extrema_are_independent(self):
        state = initial_state("auc")
        self.assertTrue(update_validation_state(
            state, 0.8, 0.7, 0.01, "auc", 1e-4, 1.0, epoch=0))
        self.assertFalse(update_validation_state(
            state, 0.7, 0.6, 0.01, "auc", 1e-4, 1.0, epoch=1))
        self.assertEqual(state["max_val_auc_epoch"], 0)
        self.assertEqual(state["min_val_loss_epoch"], 1)
        self.assertEqual(state["best_epoch"], 0)


if __name__ == "__main__":
    unittest.main()
