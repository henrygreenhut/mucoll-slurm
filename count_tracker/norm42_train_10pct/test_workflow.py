#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_subset
import build_datasets
import prepare_sources


class WorkflowTests(unittest.TestCase):
    def make_sources(self, root, count=200):
        for polarity in prepare_sources.POLARITIES:
            directory = root / polarity
            directory.mkdir(parents=True)
            for cycle in range(count):
                (directory / f"bib_sim_{cycle}.edm4hep.root").write_bytes(
                    f"{polarity}-{cycle}".encode()
                )
        (root / "MUMINUS" / "bib_sim_6291.edm4hep.root").write_bytes(b"invalid")

    def test_source_selection_is_deterministic_and_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "SIM"
            self.make_sources(sources)
            out_a, out_b = root / "a", root / "b"
            for output in (out_a, out_b):
                prepare_sources.prepare(sources, output, seed=42, cohort="tenpct")
            a = json.loads((out_a / "sources.json").read_text())
            b = json.loads((out_b / "sources.json").read_text())
            selected_a = {
                polarity: [row["cycle"] for row in a["sources"][polarity]]
                for polarity in prepare_sources.POLARITIES
            }
            selected_b = {
                polarity: [row["cycle"] for row in b["sources"][polarity]]
                for polarity in prepare_sources.POLARITIES
            }
            self.assertEqual(selected_a, selected_b)
            self.assertNotEqual(selected_a["MUPLUS"], selected_a["MUMINUS"])
            for values in selected_a.values():
                self.assertEqual(len(values), prepare_sources.FILES_PER_POLARITY)
                self.assertEqual(len(values), len(set(values)))

    def test_coverage_audit_finds_missing_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            source_root = root / "source_ids"
            data_root.mkdir()
            source_root.mkdir()
            manifest = {
                "kind": "norm42_10pct_training_development_corpus",
                "selection": {"files_per_polarity": 1},
                "source_inventory": str(root / "inventory.json"),
                "sources": {
                    "MUPLUS": [{"source_id": 6666}],
                    "MUMINUS": [{"source_id": 0}],
                },
            }
            manifest_path = root / "sources.json"
            manifest_path.write_text(json.dumps(manifest))
            (root / "inventory.json").write_text(json.dumps({
                "sources": {
                    "MUPLUS": [{"source_id": 6666}],
                    "MUMINUS": [{"source_id": 0}, {"source_id": 1}],
                }
            }))
            rows = np.array([
                [1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 1, 0, 1, 2, 0],
                [1, 0, 0, 0, 0, 1, 0, 1, 3, 0],
            ], dtype=np.float32)
            source_ids = np.array([0, 6666, 1], dtype=np.int32)
            for collection in audit_subset.COLLECTIONS:
                np.save(data_root / f"{collection}_data.npy", rows)
                np.save(source_root / f"{collection}_source_id.npy", source_ids)
            output = root / "audit"
            audit_subset.audit(
                manifest_path, data_root, source_root, output, chunk_rows=2
            )
            report = json.loads((output / "coverage_audit.json").read_text())
            self.assertFalse(
                report["decision_gate"]["passed_complete_sensor_coverage"]
            )
            self.assertEqual(report["collections"]["VBC"]["selected_hits"], 2)
            self.assertEqual(
                report["collections"]["VBC"]["missing_sensor_conditions"], 1
            )

    def test_reference_split_forces_only_singletons_into_training(self):
        y = np.array([0] * 5 + [1] * 6 + [2], dtype=np.int64)
        train, validation, singleton_classes, forced = (
            build_datasets.split_indices(y, seed=42)
        )
        self.assertEqual(singleton_classes.tolist(), [2])
        self.assertEqual(forced.tolist(), [11])
        self.assertIn(11, train)
        self.assertNotIn(11, validation)
        self.assertEqual(set(train) | set(validation), set(range(len(y))))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(y[train]), {0, 1, 2})

    def test_model_feature_order_and_condition_lookup(self):
        raw = np.array([
            [np.e, 3, 4, 7, 11, 5, 0, 1, 2, 3],
            [1, 5, 12, 8, 13, 5, 0, 1, 4, 5],
        ], dtype=np.float32)

        def mark_local_phi(values, **kwargs):
            result = values.copy()
            result[:, 2] = np.array([0.1, -0.1], dtype=np.float32)
            return result

        x_num, y, lookup = build_datasets.transform_rows(
            raw, "OuterTrackerBarrelCollection", mark_local_phi
        )
        np.testing.assert_allclose(
            x_num,
            np.array([
                [1, 11, 5, 0.1, 7],
                [0, 13, 13, -0.1, 8],
            ], dtype=np.float32),
            rtol=1e-6,
        )
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(
            lookup,
            np.array([[0, 1, 2, 3], [0, 1, 4, 5]], dtype=np.int64),
        )


if __name__ == "__main__":
    unittest.main()
