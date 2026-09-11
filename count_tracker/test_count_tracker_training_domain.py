"""Checks for the event-aware training-domain adapter."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import count_tracker_input as writer
import count_tracker_training_domain as domain


def encode(system, side, layer, module, sensor):
    return system | ((side & 3) << 5) | (layer << 7) | (module << 13) | (sensor << 24)


class TrainingDomainTests(unittest.TestCase):
    def test_training_chunk_requires_exact_values_and_order(self):
        reference = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(domain.compare_training_chunk(reference, 0, reference, "VBC"), 2)
        with self.assertRaisesRegex(ValueError, "row 0, column x"):
            changed = reference.copy()
            changed[0, 1] = 99.0
            domain.compare_training_chunk(reference, 0, changed, "VBC")
        with self.assertRaisesRegex(ValueError, "more rows"):
            domain.compare_training_chunk(reference, 1, reference, "VBC")

    def test_event_partition_is_distinct_deterministic_and_event_level(self):
        counts = Counter({event: event + 1 for event in range(20)})
        first = domain.choose_event_splits(
            counts, {"train": 5, "val": 3, "test": 4}, seed=17)
        second = domain.choose_event_splits(
            counts, {"train": 5, "val": 3, "test": 4}, seed=17)
        self.assertEqual(first, second)
        flattened = sum(first.values(), [])
        self.assertEqual(len(flattened), len(set(flattened)))
        with self.assertRaisesRegex(ValueError, "only 20"):
            domain.choose_event_splits(
                counts, {"train": 10, "val": 10, "test": 1}, seed=17)

    def test_full_scan_checks_order_and_collects_event_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = np.array([
                [0.1, 1, 2, 3, 4, 1, 0, 2, 3, 4],
                [0.2, 5, 6, 7, 8, 1, 0, 2, 3, 5],
            ], dtype=np.float64)
            for short, (_, name) in domain.COLLECTIONS.items():
                values = rows if short == "VBC" else np.empty((0, 10), dtype=np.float64)
                np.save(root / f"{name}_SimTrackerHit_conditional_reco9_0.npy", values)
            chunks = []
            for index in range(2):
                chunks.append(pd.DataFrame({
                    "event": [10 + index], "collection": ["VertexBarrelCollection"],
                    **{column: [rows[index, column_index]]
                       for column_index, column in enumerate(domain.MODEL_COLUMNS)},
                    "cellid0": [encode(1, 0, 2, 3, 4 + index)],
                }))
            with patch("pandas.read_hdf", return_value=iter(chunks)):
                counts, by_collection, _, totals = domain.scan_and_verify(
                    "source.h5", root, 1)
            self.assertEqual(counts, Counter({10: 1, 11: 1}))
            self.assertEqual(by_collection["VBC"], Counter({10: 1, 11: 1}))
            self.assertEqual(totals["VBC"], 2)

    def test_distribution_includes_events_with_zero_collection_hits(self):
        summary = domain.count_distribution(Counter({1: 10, 3: 20}), [1, 2, 3])
        self.assertEqual(summary["min"], 0)
        self.assertEqual(summary["median"], 10.0)
        self.assertEqual(summary["max"], 20)

    def test_event_products_preserve_cellid_and_empty_collections(self):
        cellid = encode(1, 0, 2, 3, 4)
        frame = pd.DataFrame({
            "event": [7], "collection": ["VertexBarrelCollection"],
            "Edep": [0.001], "x": [10.0], "y": [11.0], "z": [12.0], "t": [2.0],
            "system": [1], "side": [0], "layer": [2], "module": [3], "sensor": [4],
            "cellid0": [cellid],
        })
        products = domain.event_products(frame, 7)
        np.testing.assert_array_equal(products["VBC"][0], [[1, 0, 2, 3, 4]])
        self.assertEqual(products["VBC"][1].shape, (1, 11))
        self.assertEqual(int(products["VBC"][1][0, 10]), cellid)
        self.assertEqual(products["OTEC"][0].shape, (0, 5))
        self.assertEqual(products["OTEC"][1].shape, (0, 11))

        bad = frame.copy()
        bad.loc[0, "sensor"] = 5
        with self.assertRaisesRegex(ValueError, "disagree with cellid0"):
            domain.event_products(bad, 7)

    def test_input_loader_accepts_only_verified_training_domain_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_id = "training_domain_000007"
            event_dir = root / "test" / event_id
            event_dir.mkdir(parents=True)
            collections = {}
            arrays = {}
            for short, (system, _) in writer.COLLECTIONS.items():
                conditions = np.empty((0, 5), dtype=np.int64)
                sim = np.empty((0, 11), dtype=np.float64)
                if short == "VBC":
                    conditions = np.array([[system, 0, 2, 3, 4]], dtype=np.int64)
                    sim = np.array([[0.001, 10, 11, 12, 2, system, 0, 2, 3, 4,
                                     encode(system, 0, 2, 3, 4)]], dtype=np.float64)
                condition_path = event_dir / f"{short}_conditions.npy"
                sim_path = event_dir / f"{short}_sim_hits.npy"
                np.save(condition_path, conditions)
                np.save(sim_path, sim)
                collections[short] = {
                    "hits": len(conditions), "occupied_sensors": len(conditions),
                    "sha256": writer.sha256_file(condition_path),
                }
                arrays[short] = {
                    "path": str(sim_path.relative_to(root)), "hits": len(sim),
                    "sha256": writer.sha256_file(sim_path),
                }
            event = {"event_id": event_id, "split": "test", "source_event": 7,
                     "sim_arrays": arrays}
            report = {
                "kind": "count_tracker_training_domain_conditions",
                "manifest": {
                    "schema_version": 2, "construction": "training_domain",
                    "cell_id_encoding": writer.CELL_ID_ENCODING,
                    "selection": domain.SELECTION, "generator_training_holdout": False,
                    "events": [event],
                },
                "events": [{"event_id": event_id, "split": "test",
                            "collections": collections}],
                "training_array_verification": {
                    "status": "exact ordered match", "selection": domain.SELECTION,
                },
            }
            (root / "manifest.json").write_text(json.dumps(report))
            loaded, expected = writer.load_event(root, "test", event_id, "training_domain")
            self.assertEqual(loaded["source_event"], 7)
            self.assertEqual(expected["VBC"], Counter({(1, 0, 2, 3, 4): 1}))

            report["manifest"]["generator_training_holdout"] = True
            (root / "manifest.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "not a holdout"):
                writer.load_event(root, "test", event_id, "training_domain")


if __name__ == "__main__":
    unittest.main()
