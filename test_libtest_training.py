#!/usr/bin/env python3
import json
import inspect
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import numpy as np

import libtest_common as lc
import pfn_libtest_train as file_trainer
import pfn_training_engine as engine
import pfn_variable_reuse_train as variable_trainer
from pfn_libtest_train import (
    balanced_chunks,
    class_layout,
    load_or_create_validation_units,
)
from pfn_training_engine import (
    binary_cross_entropy,
    initial_state,
    update_validation_state,
    write_or_validate_config,
)


class LibtestClassLayoutTests(unittest.TestCase):
    def test_main_uses_unique_and_reuse_stores(self):
        unique, reuse = object(), object()
        actual = class_layout(unique, reuse, 420, 5)
        self.assertEqual(actual, (unique, reuse, 420, 84))

    def test_unique_null_is_unique_vs_unique(self):
        unique, reuse = object(), object()
        actual = class_layout(unique, reuse, 420, 7, True, "unique")
        self.assertEqual(actual, (unique, unique, 420, 420))

    def test_reuse_null_is_reuse_vs_reuse(self):
        unique, reuse = object(), object()
        actual = class_layout(unique, reuse, 420, 7, True, "reuse")
        self.assertEqual(actual, (reuse, reuse, 60, 60))


class LibtestNormalizationTests(unittest.TestCase):
    def test_high_energy_muon_cut_is_applied_before_features(self):
        raw = {
            "px": np.arange(4, dtype=np.float32),
            "py": np.arange(4, dtype=np.float32),
            "pz": np.arange(4, dtype=np.float32),
            "E": np.asarray([4.0, 6.0, 8.0, 9.0], dtype=np.float32),
            "t": np.arange(4, dtype=np.float32),
            "vx": np.arange(4, dtype=np.float32),
            "vy": np.arange(4, dtype=np.float32),
            "vz": np.arange(4, dtype=np.float32),
            "pdg": np.asarray([13, -13, 11, 13], dtype=np.int32),
        }

        filtered = file_trainer.exclude_high_energy_muons(raw, 5.0)

        np.testing.assert_array_equal(filtered["E"], [4.0, 8.0])
        np.testing.assert_array_equal(filtered["pdg"], [13, 11])
        self.assertIs(file_trainer.exclude_high_energy_muons(raw, 0.0), raw)

    def test_all_muon_cut_removes_both_charges_at_every_energy(self):
        raw = {
            "E": np.asarray([0.2, 8.0, 4.0], dtype=np.float32),
            "pdg": np.asarray([13, -13, 11], dtype=np.int32),
        }

        filtered = file_trainer.filter_muons(raw, exclude_all=True)

        np.testing.assert_array_equal(filtered["E"], [4.0])
        np.testing.assert_array_equal(filtered["pdg"], [11])

    def test_expanded_no_phi_removes_only_azimuth(self):
        expanded = lc.feature_names("expanded")
        no_phi = lc.feature_names("expanded_no_phi")
        self.assertEqual(
            no_phi,
            [name for name in expanded if name not in ("cosphi", "sinphi")],
        )

    def test_expanded_no_phi_is_invariant_under_z_rotation(self):
        raw = {
            "px": np.asarray([1.0, -2.0], np.float32),
            "py": np.asarray([2.0, 1.0], np.float32),
            "pz": np.asarray([3.0, -4.0], np.float32),
            "E": np.asarray([5.0, 6.0], np.float32),
            "t": np.asarray([7.0, 8.0], np.float32),
            "vx": np.asarray([9.0, -10.0], np.float32),
            "vy": np.asarray([11.0, 12.0], np.float32),
            "vz": np.asarray([13.0, -14.0], np.float32),
            "pdg": np.asarray([22, 11], np.int32),
        }
        angle = 0.73
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated = {name: values.copy() for name, values in raw.items()}
        rotated["px"] = cosine * raw["px"] - sine * raw["py"]
        rotated["py"] = sine * raw["px"] + cosine * raw["py"]
        rotated["vx"] = cosine * raw["vx"] - sine * raw["vy"]
        rotated["vy"] = sine * raw["vx"] + cosine * raw["vy"]
        np.testing.assert_allclose(
            lc.build_features(raw, "expanded_no_phi"),
            lc.build_features(rotated, "expanded_no_phi"),
            rtol=1e-6, atol=1e-6,
        )

    def test_training_normalization_streams_units_without_changing_statistics(self):
        class Store:
            def __init__(self, offset):
                self.arrays = [
                    np.full((size, 3), offset + index, dtype=np.float32)
                    for index, size in enumerate((2, 5, 3, 7, 4, 6))
                ]

            def file_arrays(self, positions):
                return np.concatenate([
                    self.arrays[int(position)] for position in positions
                ])

        samplers = [
            file_trainer.UnitSampler(
                Store(offset), {"train": np.arange(6)}, 2, "expanded")
            for offset in (0, 10)
        ]
        reference_arrays = []
        reference_multiplicities = []
        reference_rng = np.random.default_rng(23)
        for sampler in samplers:
            for _ in range(4):
                positions = sampler.random_unit(reference_rng, "train")
                array = sampler.store.file_arrays(positions)
                reference_arrays.append(array)
                reference_multiplicities.append(len(array))
        reference_mean, reference_std = lc.compute_norm_stats(reference_arrays)

        with mock.patch.object(
                lc, "build_features",
                side_effect=lambda raw, feature_set: raw):
            with mock.patch.object(
                    lc, "compute_norm_stats",
                    wraps=lc.compute_norm_stats) as compute:
                mean, std, multiplicities = (
                    file_trainer.sample_normalization_stats(
                        samplers, 4, np.random.default_rng(23)))

        self.assertNotIsInstance(compute.call_args.args[0], list)
        np.testing.assert_array_equal(mean, reference_mean)
        np.testing.assert_array_equal(std, reference_std)
        self.assertEqual(multiplicities, reference_multiplicities)

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


class GradientAccumulationTests(unittest.TestCase):
    def test_two_microbatches_match_one_full_batch_update(self):
        import tensorflow as tf

        def build_model():
            inputs = tf.keras.Input(shape=(3,))
            outputs = tf.keras.layers.Dense(
                2, activation="softmax",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=7),
                bias_initializer="zeros",
            )(inputs)
            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
                loss="categorical_crossentropy",
            )
            return model

        x = np.asarray([
            [0.2, -0.4, 0.7],
            [1.1, 0.3, -0.2],
            [-0.8, 0.5, 0.6],
            [0.4, 0.9, -1.2],
        ], dtype=np.float32)
        labels = np.asarray([0, 1, 0, 1], dtype=np.int32)
        y = np.eye(2, dtype=np.float32)[labels]

        full = build_model()
        accumulated = build_model()
        accumulated.set_weights(full.get_weights())

        full.train_on_batch(x, y)
        microbatches = iter([
            (x[[0, 1]], y[[0, 1]], labels[[0, 1]]),
            (x[[2, 3]], y[[2, 3]], labels[[2, 3]]),
        ])
        losses = engine.accumulated_train_step(
            accumulated, microbatches, 2, tf)

        self.assertEqual(len(losses), 2)
        self.assertEqual(int(full.optimizer.iterations.numpy()), 1)
        self.assertEqual(int(accumulated.optimizer.iterations.numpy()), 1)
        for observed, expected in zip(
                accumulated.get_weights(), full.get_weights()):
            np.testing.assert_allclose(
                observed, expected, rtol=1e-6, atol=1e-7)


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
            write_or_validate_config(
                path, dict(changed_runtime, gradient_accumulation_steps=1))
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


class SharedTrainingEngineTests(unittest.TestCase):
    def test_both_entry_points_delegate_fitting_to_one_engine(self):
        self.assertIs(file_trainer.run_binary_pfn_training,
                      engine.run_binary_pfn_training)
        self.assertIs(variable_trainer.run_binary_pfn_training,
                      engine.run_binary_pfn_training)
        self.assertNotIn("train_on_batch", inspect.getsource(file_trainer))
        self.assertNotIn(
            "train_on_batch", inspect.getsource(variable_trainer))
        self.assertIn("train_on_batch", inspect.getsource(engine))

    def test_variable_recipe_matches_successful_n420_scaled_run(self):
        args = SimpleNamespace(
            mother_store="/tmp/mothers.h5",
            label="test",
            reuse_k=(1, 10),
            model_seed=1,
            null_test=False,
            max_minutes=1400.0,
            progress_every=25,
        )
        config = variable_trainer.scientific_config(
            args, epochs=80, patience=15, min_epochs=0, units=500,
            val_units=300, test_units=300, norm_units=100)
        expected = {
            "features": "expanded",
            "architecture": "energyflow-scaled-sum",
            "phi_sizes": (100, 100, 128),
            "f_sizes": (200, 200, 200),
            "batch_size": 4,
            "units_per_epoch_per_class": 500,
            "val_units_per_class": 300,
            "test_units_per_class": 300,
            "epochs": 80,
            "patience": 15,
            "learning_rate": 1.0e-4,
            "warmup_epochs": 1,
            "warmup_steps": 250,
            "decay_epochs": 30,
            "decay_steps": 7500,
            "min_learning_rate": 1.0e-6,
            "clipnorm": 0.0,
            "latent_dropout": 0.0,
            "f_dropout": 0.0,
            "phi_l2": 0.0,
            "f_l2": 0.0,
            "jit_compile": False,
            "data_seed": 1701,
            "normalization_seed": 1701,
            "validation_definition_seed": 2700,
            "test_definition_seed": 3727,
            "epoch_seed_formula": "data_seed * 100003 + epoch",
            "model_seed": 1,
            "selection_metric": "validation loss",
            "min_epochs": 0,
            "min_delta": 1.0e-4,
            "min_delta_sigma": 1.0,
            "norm_stat_units_per_class": 100,
            "test_events_overlap_sources": True,
            "materialization_workers": 4,
            "prefetch_batches": 1,
        }
        for key, value in expected.items():
            self.assertEqual(config[key], value, key)

    def test_variable_recipe_can_require_eighty_epochs(self):
        args = SimpleNamespace(
            mother_store="/tmp/mothers.h5",
            label="test_min80",
            reuse_k=(1, 5),
            model_seed=1,
            null_test=False,
            max_minutes=1400.0,
            progress_every=25,
        )
        config = variable_trainer.scientific_config(
            args, epochs=120, patience=15, min_epochs=80, units=500,
            val_units=300, test_units=300, norm_units=100)

        self.assertEqual(config["epochs"], 120)
        self.assertEqual(config["min_epochs"], 80)
        self.assertEqual(config["patience"], 15)

    def test_scaled_energyflow_model_is_built_by_shared_engine(self):
        sentinel = object()
        config = {
            "n_features": 13,
            "latent_scale": 1.0 / 1000.0,
            "phi_sizes": (100, 100, 128),
            "f_sizes": (200, 200, 200),
            "arch": "energyflow",
            "jit": False,
            "lr": 1.0e-4,
            "warmup_steps": 250,
            "decay_steps": 7500,
            "min_lr": 1.0e-6,
            "clipnorm": 0.0,
        }
        with mock.patch.object(
                lc, "build_pfn_energyflow_scaled",
                return_value=sentinel) as builder:
            self.assertIs(engine._build_model(config), sentinel)
        builder.assert_called_once()
        _, latent_scale = builder.call_args.args
        self.assertEqual(latent_scale, config["latent_scale"])
        self.assertFalse(builder.call_args.kwargs["jit_compile"])
        self.assertEqual(builder.call_args.kwargs["warmup_steps"], 250)
        self.assertEqual(builder.call_args.kwargs["decay_steps"], 7500)
        self.assertEqual(builder.call_args.kwargs["clipnorm"], 0.0)

    def test_shared_engine_executes_complete_train_validation_cycle(self):
        class FakeVariable:
            def __init__(self, value, **_):
                self.value = value

            def assign(self, value):
                self.value = value

            def numpy(self):
                return self.value

        class FakeOptimizer:
            def __init__(self):
                self.learning_rate = 1.0e-4
                self.iterations = 0

            def build(self, _):
                pass

        class FakeModel:
            def __init__(self):
                self.optimizer = FakeOptimizer()
                self.trainable_variables = []
                self.loaded = None

            def train_on_batch(self, _x, _y):
                self.optimizer.iterations += 1
                return 0.6

            def save_weights(self, path):
                with open(path, "w") as handle:
                    handle.write("weights")

            def load_weights(self, path):
                self.loaded = path

        class FakeCheckpoint:
            def __init__(self, **_):
                pass

            def restore(self, _):
                return self

            def assert_consumed(self):
                pass

        class FakeManager:
            def __init__(self, *_args, **_kwargs):
                self.latest_checkpoint = None

            def save(self, **_):
                pass

        fake_tf = SimpleNamespace(
            int64="int64",
            float64="float64",
            Variable=FakeVariable,
            keras=SimpleNamespace(
                utils=SimpleNamespace(set_random_seed=lambda _: None)),
            train=SimpleNamespace(
                Checkpoint=FakeCheckpoint,
                CheckpointManager=FakeManager),
        )
        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "result_dir": directory,
                "n_features": 2,
                "latent_scale": 1.0,
                "phi_sizes": (2,),
                "f_sizes": (2,),
                "arch": "energyflow",
                "jit": False,
                "lr": 1.0e-4,
                "warmup_steps": 1,
                "decay_steps": 30,
                "min_lr": 1.0e-6,
                "clipnorm": 0.0,
                "model_seed": 1,
                "select_metric": "loss",
                "min_delta": 0.0,
                "min_delta_sigma": 1.0,
                "epochs": 1,
                "patience": 15,
                "min_epochs": 0,
                "units_per_epoch": 1,
                "batch_size": 2,
                "max_minutes": 0.0,
                "progress_every": 0,
            }

            def train_batches(_epoch):
                yield (
                    np.zeros((2, 1, 2), np.float32),
                    np.eye(2, dtype=np.float32),
                    np.asarray([0, 1], np.int32),
                )

            def predict_validation(_model):
                return (
                    np.asarray([0, 1], np.int32),
                    np.asarray([0.2, 0.8], np.float64),
                )

            with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}), \
                    mock.patch.object(engine, "_build_model",
                                      return_value=model):
                returned, state, complete = engine.run_binary_pfn_training(
                    config, train_batches, predict_validation)

            self.assertIs(returned, model)
            self.assertTrue(complete)
            self.assertTrue(state["done"])
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(state["best_epoch"], 0)
            self.assertEqual(model.loaded, os.path.join(
                directory, "best.weights.h5"))
            self.assertTrue(os.path.isfile(os.path.join(
                directory, "history.csv")))


class VariableReuseDefinitionTests(unittest.TestCase):
    def test_parallel_materialization_preserves_event_order_and_contents(self):
        definitions = [(0, 1, 11), (1, 7, 22), (0, 1, 33), (1, 7, 44)]

        def fake_unit_features(_store, _pool, physical_k, seed):
            time.sleep((50 - seed) / 5000.0)
            return np.full((physical_k + 1, 2), seed, dtype=np.float32)

        with mock.patch.object(
                variable_trainer, "unit_features",
                side_effect=fake_unit_features):
            sequential = variable_trainer.padded_batch(
                definitions, None, None, np.zeros(2), np.ones(2))
            with ThreadPoolExecutor(max_workers=4) as executor:
                parallel = variable_trainer.padded_batch(
                    definitions, None, None, np.zeros(2), np.ones(2),
                    executor=executor)

        for actual, expected in zip(parallel, sequential):
            np.testing.assert_array_equal(actual, expected)

    def test_parallel_materialization_uses_multiple_workers(self):
        definitions = [(0, 1, seed) for seed in range(4)]
        lock = threading.Lock()
        active = 0
        maximum = 0

        def fake_unit_features(_store, _pool, _physical_k, seed):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return np.full((2, 2), seed, dtype=np.float32)

        with mock.patch.object(
                variable_trainer, "unit_features",
                side_effect=fake_unit_features):
            with ThreadPoolExecutor(max_workers=4) as executor:
                variable_trainer.padded_batch(
                    definitions, None, None, np.zeros(2), np.ones(2),
                    executor=executor)

        self.assertGreater(maximum, 1)

    def test_prefetch_preserves_order(self):
        self.assertEqual(
            list(variable_trainer.prefetch_one(iter(range(8)))),
            list(range(8)))

    def test_main_definitions_map_labels_to_physical_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            definitions = variable_trainer.load_or_create_seed_definitions(
                os.path.join(directory, "definitions.npz"),
                reuse_k=(1, 10), units_per_class=20, seed=7,
                null_test=False)
        for label, physical_k, _ in definitions:
            self.assertEqual(physical_k, (1, 10)[label])

    def test_null_keeps_inputs_but_permutes_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            main = variable_trainer.load_or_create_seed_definitions(
                os.path.join(directory, "main.npz"),
                reuse_k=(10, 42), units_per_class=100, seed=9,
                null_test=False)
            null = variable_trainer.load_or_create_seed_definitions(
                os.path.join(directory, "null.npz"),
                reuse_k=(10, 42), units_per_class=100, seed=9,
                null_test=True)
        self.assertEqual(
            [(physical_k, seed) for _, physical_k, seed in main],
            [(physical_k, seed) for _, physical_k, seed in null])
        self.assertEqual([label for label, _, _ in null].count(0), 100)
        self.assertNotEqual(
            [label for label, _, _ in main],
            [label for label, _, _ in null])

    def test_test_summary_records_actual_smoke_sample_size(self):
        definitions = [
            (label, (1, 10)[label], seed)
            for label in (0, 1) for seed in range(4)]
        labels = np.asarray([definition[0] for definition in definitions])
        scores = np.where(labels == 1, 0.8, 0.2)
        state = engine.initial_state("loss")
        state.update(epoch=2, best_epoch=1, max_val_auc=0.75,
                     max_val_auc_epoch=1,
                     min_val_loss=0.6, done=True)
        args = SimpleNamespace(label="smoke", null_test=False)
        with tempfile.TemporaryDirectory() as directory:
            variable_trainer.save_test_outputs(
                directory, definitions, labels, scores, (1, 10), 4,
                state, args)
            with open(os.path.join(directory, "summary.json")) as handle:
                summary = json.load(handle)
        self.assertEqual(summary["test_units_per_class"], 4)
        self.assertEqual(summary["test_mode"],
                         "overlapping held-out events; point estimate only")


if __name__ == "__main__":
    unittest.main()
