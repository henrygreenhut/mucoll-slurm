#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

import numpy as np

import libtest_common as lc
from pfn_libtest_train import binary_cross_entropy


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


if __name__ == "__main__":
    unittest.main()
