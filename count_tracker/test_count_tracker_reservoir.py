"""Checks for conditional empirical-hit reservoir sampling."""

import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

import count_tracker_reservoir as reservoir
import count_tracker_input as writer


def condition(sensor, n):
    return np.tile(np.array([1, 0, 0, 0, sensor], dtype=np.int64), (n, 1))


def event(event_id, split, rows):
    return {"event_id": event_id, "split": split, "conditions": {"VBC": rows}}


class ReservoirTests(unittest.TestCase):
    def test_cellid_keys_preserve_signed_side(self):
        labels = np.array([[2, -1, 7, 2047, 255]], dtype=np.int64)
        expected = 2 | (3 << 5) | (7 << 7) | (2047 << 13) | (255 << 24)
        self.assertEqual(int(reservoir.encode_sensor_keys(labels)[0]), expected)
        with self.assertRaises(ValueError):
            reservoir.encode_sensor_keys(np.array([[1, 0, 0, 0, 256]]))

    def test_no_reuse_is_global_and_sensor_matched(self):
        pool_labels = condition(3, 8)
        pool_keys = reservoir.encode_sensor_keys(pool_labels)
        events = [
            event("a", "train", condition(3, 2)),
            event("b", "test", condition(3, 3)),
        ]
        indices, audit = reservoir.draw_indices(
            pool_keys, events, "VBC", seed=4, reuse_policy="none")
        used = np.concatenate(indices)
        self.assertEqual(len(used), len(np.unique(used)))
        np.testing.assert_array_equal(pool_keys[indices[0]],
                                      reservoir.encode_sensor_keys(condition(3, 2)))
        self.assertEqual(audit[0]["splits"]["test"]["reused"], 0)

    def test_within_split_reuse_never_crosses_splits(self):
        pool_keys = reservoir.encode_sensor_keys(condition(3, 4))
        events = [
            event("train_a", "train", condition(3, 3)),
            event("train_b", "train", condition(3, 3)),
            event("test_a", "test", condition(3, 1)),
        ]
        indices, audit = reservoir.draw_indices(
            pool_keys, events, "VBC", seed=9, reuse_policy="within-split")
        self.assertEqual(len(np.unique(indices[0])), 3)
        self.assertEqual(len(np.unique(indices[1])), 3)
        self.assertTrue(set(indices[0]).isdisjoint(indices[2]))
        self.assertTrue(set(indices[1]).isdisjoint(indices[2]))
        self.assertEqual(audit[0]["splits"]["train"]["reused"], 3)

    def test_capacity_rules_distinguish_the_two_policies(self):
        requested = {7: 6}
        by_split = {
            "train": {7: 4}, "val": {}, "test": {7: 2},
        }
        maxima = {
            "train": {7: 3}, "val": {}, "test": {7: 1},
        }
        self.assertEqual(len(reservoir.capacity_errors(
            {7: 4}, requested, by_split, maxima, "none")), 1)
        self.assertEqual(reservoir.capacity_errors(
            {7: 4}, requested, by_split, maxima, "within-split"), [])
        self.assertEqual(len(reservoir.capacity_errors(
            {7: 3}, requested, by_split, maxima, "within-split")), 1)

    def test_split_pool_allocation_respects_event_minima(self):
        sizes = reservoir.allocate_split_pool_sizes(
            20,
            {"train": 12, "val": 4, "test": 4},
            {"train": 5, "val": 2, "test": 2},
        )
        self.assertEqual(sum(sizes.values()), 20)
        self.assertGreaterEqual(sizes["train"], 5)
        self.assertGreaterEqual(sizes["val"], 2)
        self.assertGreaterEqual(sizes["test"], 2)

    def test_prepare_builds_input_compatible_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            templates = root / "templates"
            training = root / "training"
            event_id = "norm42_cmp_000000"
            event_dir = templates / "test" / event_id
            event_dir.mkdir(parents=True)
            training.mkdir()

            summaries = {}
            verification_collections = {}
            for short, (system, name) in reservoir.COLLECTIONS.items():
                conditions = np.array([[system, 0, 0, 0, 0]], dtype=np.int64)
                path = event_dir / f"{short}_conditions.npy"
                np.save(path, conditions)
                summaries[short] = {
                    "hits": 1, "occupied_sensors": 1,
                    "sha256": reservoir.sha256_file(path),
                }
                pool = np.array([
                    [0.001, 1, 2, 3, 4, system, 0, 0, 0, 0],
                    [0.002, 5, 6, 7, 8, system, 0, 0, 0, 0],
                ], dtype=np.float64)
                pool_path = training / f"{name}_SimTrackerHit_conditional_reco9_0.npy"
                np.save(pool_path, pool)
                verification_collections[short] = {"path": str(pool_path), "rows": 2}

            source_event = {
                "event_id": event_id, "split": "test",
                "sources": {
                    polarity: [
                        {"path": f"/{polarity}/{cycle}.root", "cycle": cycle, "entry": 0}
                        for cycle in range(10)
                    ] for polarity in ("MUPLUS", "MUMINUS")
                },
            }
            template_report = {
                "manifest": {
                    "schema_version": 2, "construction": "norm42",
                    "file_normalization": 42, "n_files_per_polarity": 10,
                    "norm1_equivalents_per_polarity": 420,
                    "cell_id_encoding": reservoir.CELL_ID_ENCODING,
                    "sampling": {"cohort": "cmp"}, "events": [source_event],
                },
                "events": [{"event_id": event_id, "split": "test",
                            "collections": summaries}],
            }
            (templates / "manifest.json").write_text(json.dumps(template_report))
            verification = root / "verification.json"
            verification.write_text(json.dumps({
                "status": "exact ordered match", "selection": reservoir.SELECTION,
                "collections": verification_collections,
            }))
            output = root / "output"
            args = types.SimpleNamespace(
                output=str(output), splits=["test"], training_npys=str(training),
                verification_report=str(verification), templates=str(templates),
                reuse_policy="none", seed=11,
            )
            reservoir.prepare(args)
            new_id = "norm42_reservoir_cmp_000000"
            loaded, expected = writer.load_event(
                output, "test", new_id, "norm42_reservoir")
            self.assertEqual(loaded["template_event_id"], event_id)
            self.assertEqual(sum(map(sum, (counter.values() for counter in expected.values()))), 6)


if __name__ == "__main__":
    unittest.main()
