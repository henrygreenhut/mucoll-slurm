"""Scientific and provenance checks for the direct norm1 mother preparation."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import count_tracker_input as writer
import count_tracker_mother_direct as mother


class MotherDirectTests(unittest.TestCase):
    def test_split_mother_edm4hep_source_schema_uses_lowercase_edep(self):
        self.assertEqual(mother.TRACKER_FIELDS, (
            "eDep", "position.x", "position.y", "position.z", "time", "cellID"))

    def test_cycle_metadata_requires_and_retains_both_polarities(self):
        files = np.array([
            [0, 1, 0, 2, 3, 0, 2], [0, -1, 2, 1, 4, 2, 1],
            [1, 1, 3, 2, 5, 3, 2], [1, -1, 5, 2, 6, 5, 2],
        ], dtype=np.int64)
        records, cycles = mother.cycle_metadata(files)
        self.assertEqual(cycles, [0, 1])
        self.assertEqual(records[(1, 1)]["mother_count"], 2)
        self.assertEqual(records[(1, -1)]["source_tracker_hits"], 6)
        with self.assertRaisesRegex(ValueError, "missing one polarity"):
            mother.cycle_metadata(files[:-1])

    def test_source_selection_is_independent_by_polarity_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = {}
            cycles = list(range(10))
            for cycle in cycles:
                for polarity, code in (("MUPLUS", 1), ("MUMINUS", -1)):
                    path = root / polarity / f"bib_sim_{cycle}.edm4hep.root"
                    path.parent.mkdir(exist_ok=True)
                    path.touch()
                    records[(cycle, code)] = {
                        "mother_count": 1, "source_tracker_hits": 0,
                        "mother_start": cycle, "row_start": cycle, "row_count": 1,
                    }
            first = mother.select_events(cycles, 1, root, records, 12345, "cmp")
            second = mother.select_events(cycles, 1, root, records, 12345, "cmp")
        self.assertEqual(first, second)
        plus = [source["cycle"] for source in first[0]["sources"]["MUPLUS"]]
        minus = [source["cycle"] for source in first[0]["sources"]["MUMINUS"]]
        self.assertLess(len(set(plus)), 420)
        self.assertLess(len(set(minus)), 420)
        self.assertNotEqual(plus, minus)

    def test_classifier_cycle_pools_and_event_ids_are_split_isolated(self):
        cycles = list(range(100))
        pools = mother.split_cycle_pools(cycles, 42)
        self.assertEqual({split: len(pool) for split, pool in pools.items()},
                         {"train": 60, "val": 20, "test": 20})
        self.assertEqual(set().union(*map(set, pools.values())), set(cycles))
        self.assertEqual(sum(map(len, pools.values())), len(cycles))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = {}
            for cycle in cycles:
                for polarity, code in (("MUPLUS", 1), ("MUMINUS", -1)):
                    path = root / polarity / f"bib_sim_{cycle}.edm4hep.root"
                    path.parent.mkdir(exist_ok=True)
                    path.touch()
                    records[(cycle, code)] = {
                        "mother_count": 1, "source_tracker_hits": 0,
                        "mother_start": cycle, "row_start": cycle, "row_count": 1,
                    }
            events = []
            for split in ("train", "val", "test"):
                events.extend(mother.select_events(
                    pools[split], 1, root, records, 42, "overnight", split))
        self.assertEqual(len({event["event_id"] for event in events}), 3)
        for event in events:
            allowed = set(pools[event["split"]])
            for sources in event["sources"].values():
                self.assertTrue({source["cycle"] for source in sources} <= allowed)

    def test_all_stored_hits_are_retained_with_sensor_identity(self):
        labels = np.array([[1, 0, 0, 0, 3]] * 3, dtype=np.int64)
        cell_ids = writer.pack_cell_ids(labels)
        values = [
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-8.0, 1.0e7, 1.0e7 + 1.0]),
            cell_ids,
        ]
        rows, summary = mother.all_tracker_rows(values, 1)
        self.assertEqual(summary, {"all_stored": 3})
        np.testing.assert_array_equal(rows[:, 5:10], labels)
        np.testing.assert_array_equal(rows[:, 4], [-8.0, 1.0e7, 1.0e7 + 1.0])

    def test_training_raw_time_cut_is_applied_before_conditions(self):
        labels = np.array([[1, 0, 0, 0, 3]] * 3, dtype=np.int64)
        values = [
            np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3),
            np.array([-8.0, 1.0e7 - 1.0, 1.0e7]),
            writer.pack_cell_ids(labels),
        ]
        rows, summary = mother.tracker_rows(
            values, 1, mother.TRAINING_RAW_TIME_SELECTION)
        self.assertEqual(summary, {"all_stored": 3, "selected": 2})
        np.testing.assert_array_equal(rows[:, 4], [-8.0, 1.0e7 - 1.0])

    def test_prepared_manifest_is_accepted_by_input_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_id = "norm1_mother_direct_trackcmp_000000"
            event_dir = root / "test" / event_id
            event_dir.mkdir(parents=True)
            arrays, summaries = {}, {}
            for short, (system, _) in writer.COLLECTIONS.items():
                labels = np.array([[system, 0, 0, 0, 1]], dtype=np.int64)
                cell_id = writer.pack_cell_ids(labels).astype(np.float64)
                rows = np.column_stack((np.ones((1, 5)), labels, cell_id))
                path = event_dir / f"{short}_sim_hits.npy"
                np.save(path, rows)
                conditions = event_dir / f"{short}_conditions.npy"
                np.save(conditions, labels)
                arrays[short] = {
                    "path": str(Path("test") / event_id / path.name),
                    "sha256": writer.sha256_file(path), "hits": 1,
                }
                summaries[short] = {
                    "hits": 1, "occupied_sensors": 1,
                    "sha256": writer.sha256_file(conditions), "all_stored": 1,
                }
            sources = {
                polarity: [{"draw": draw, "cycle": draw % 10,
                            "path": f"/{polarity}/{draw % 10}.root",
                            "entries": "all", "mother_count": 1,
                            "source_tracker_hits": 1}
                           for draw in range(420)]
                for polarity in writer.POLARITIES
            }
            event = {"event_id": event_id, "split": "test", "sources": sources,
                     "sim_arrays": arrays}
            report = {
                "kind": "count_tracker_norm1_mother_direct_conditions",
                "hit_selection": "all-stored",
                "manifest": {
                    "schema_version": 2,
                    "construction": mother.CONSTRUCTION,
                    "cell_id_encoding": writer.CELL_ID_ENCODING,
                    "generator_training_holdout": False,
                    "model_split_used": False,
                    "analysis_split_used": False,
                    "classifier_ready": False,
                    "physical_event_boundaries": True,
                    "n_files_per_polarity": 420,
                    "norm1_equivalents_per_polarity": 420,
                    "hit_selection": "all-stored",
                    "source_cycle_pool": {
                        "kind": "all complete cycles in files.npy",
                        "count": 10, "cycles": list(range(10))},
                    "events": [event],
                },
                "events": [{"event_id": event_id, "split": "test",
                            "collections": summaries}],
            }
            (root / "manifest.json").write_text(json.dumps(report))
            loaded, expected = writer.load_event(
                root, "test", event_id, mother.CONSTRUCTION)
            self.assertEqual(loaded["event_id"], event_id)
            self.assertEqual(loaded["_source_domain"], None)
            self.assertEqual(loaded["_source_cycle_pool"]["count"], 10)
            self.assertTrue(all(sum(counts.values()) == 1 for counts in expected.values()))


if __name__ == "__main__":
    unittest.main()
