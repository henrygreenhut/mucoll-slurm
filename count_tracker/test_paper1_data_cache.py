#!/usr/bin/env python3
"""Unit tests for the deterministic Paper1 data cache."""

from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

from paper1_data_cache import PreparedPaper1Sampler


class Paper1DataCacheTests(unittest.TestCase):
    @staticmethod
    def _module():
        state = {"make_dataset_calls": 0}

        def make_dataset(path, transform, *, num_classes, is_y_cond, change_val):
            state["make_dataset_calls"] += 1
            return {
                "path": path,
                "transform": transform,
                "num_classes": num_classes,
                "is_y_cond": is_y_cond,
                "change_val": change_val,
            }

        module = types.SimpleNamespace(make_dataset=make_dataset)
        return module, state

    @staticmethod
    def _config():
        return {
            "seed": 42,
            "normalization": "quantile",
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default",
        }

    def test_dataset_is_prepared_once_for_identical_inputs(self):
        module, state = self._module()
        sampler = PreparedPaper1Sampler(module)
        model_params = {"num_classes": 7, "is_y_cond": True}
        config = self._config()

        first = sampler._dataset_for("/tmp/data", "T1", config, model_params, False)
        second = sampler._dataset_for("/tmp/data", "T2", config, model_params, False)

        self.assertIs(first, second)
        self.assertEqual(state["make_dataset_calls"], 1)
        self.assertEqual(sampler.dataset_preparations, 1)

    def test_reuse_with_different_preparation_inputs_is_rejected(self):
        module, _state = self._module()
        sampler = PreparedPaper1Sampler(module)
        model_params = {"num_classes": 7, "is_y_cond": True}
        config = self._config()
        sampler._dataset_for("/tmp/data", "T1", config, model_params, False)

        changed = dict(config, normalization="standard")
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            sampler._dataset_for("/tmp/data", "T2", changed, model_params, False)

    def test_raw_training_features_are_loaded_once(self):
        module, _state = self._module()
        sampler = PreparedPaper1Sampler(module)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "X_num_train.npy"
            expected = np.arange(10, dtype=np.float32).reshape(5, 2)
            np.save(path, expected)
            first = sampler._raw_numerical_features(tmp)
            np.save(path, np.zeros_like(expected))
            second = sampler._raw_numerical_features(tmp)

        self.assertIs(first, second)
        np.testing.assert_array_equal(second, expected)

    def test_discrete_column_scan_is_cached(self):
        module, _state = self._module()
        sampler = PreparedPaper1Sampler(module)
        with tempfile.TemporaryDirectory() as tmp:
            features = np.column_stack([
                np.arange(40, dtype=np.float32),
                np.tile(np.array([0, 1], dtype=np.float32), 20),
            ])
            np.save(Path(tmp) / "X_num_train.npy", features)
            first = sampler._discrete_columns(tmp)
            np.save(Path(tmp) / "X_num_train.npy", np.zeros_like(features))
            second = sampler._discrete_columns(tmp)

        self.assertEqual(first, (1,))
        self.assertEqual(second, first)
        self.assertEqual(sampler.discrete_column_scans, 1)


if __name__ == "__main__":
    unittest.main()
