"""Checks for source isolation and exact, event-local sensor occupancy."""

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import count_tracker_conditions as conditions


def manifest():
    return {
        "schema_version": 2,
        "construction": "norm1",
        "file_normalization": 1,
        "norm1_equivalents_per_polarity": 420,
        "cell_id_encoding": conditions.CELL_ID_ENCODING,
        "n_files_per_polarity": 420,
        "events": [{
            "event_id": "event_0", "split": "train",
            "sources": {
                polarity: [
                    {"path": f"{polarity}/{cycle}.root", "cycle": cycle, "entry": 0}
                    for cycle in range(420)
                ] for polarity in conditions.POLARITIES
            },
        }],
    }


class ConditionTests(unittest.TestCase):
    def test_norm42_normalization_and_source_count_are_explicit(self):
        data = manifest()
        data.update(construction="norm42", file_normalization=42, n_files_per_polarity=10)
        for polarity in conditions.POLARITIES:
            data["events"][0]["sources"][polarity] = data["events"][0]["sources"][polarity][:10]
        conditions.validate_manifest(data, Path("/tmp"))
        data["n_files_per_polarity"] = 420
        with self.assertRaisesRegex(ValueError, "requires n_files_per_polarity=10"):
            conditions.validate_manifest(data, Path("/tmp"))

    def test_ambiguous_old_manifest_is_rejected(self):
        data = manifest()
        data["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "explicit construction"):
            conditions.validate_manifest(data, Path("/tmp"))

    def test_norm42_conditions_are_counted_from_twenty_actual_files(self):
        from count_tracker_input import load_event

        data = manifest()
        data.update(construction="norm42", file_normalization=42, n_files_per_polarity=10)
        data["events"][0]["event_id"] = "norm42_SIM_A_000000"
        for polarity in conditions.POLARITIES:
            data["events"][0]["sources"][polarity] = data["events"][0]["sources"][polarity][:10]
        counts = {short: Counter() for short in conditions.COLLECTIONS}
        counts["VBC"][(1, 0, 0, 0, 0)] = 7
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sources.json"
            source.write_text(json.dumps(data))
            output = Path(directory) / "conditions"
            with patch.object(conditions, "read_source_counts", return_value=counts) as read:
                conditions.prepare(source, output)
            self.assertEqual(read.call_count, 20)
            event, expected = load_event(output, "train", "norm42_SIM_A_000000", "norm42")
            # norm42 file contents already include their normalization: do not
            # multiply their actual hit counts by 42 a second time.
            self.assertEqual(expected["VBC"], Counter({(1, 0, 0, 0, 0): 140}))
            self.assertEqual(len(event["sources"]["MUPLUS"]), 10)
            with self.assertRaisesRegex(ValueError, "different SIM construction"):
                load_event(output, "train", "norm42_SIM_A_000000", "norm1")

    def test_decode_signed_side_and_extreme_sensor_fields(self):
        # Independently constructed known bit patterns, including side=-1.
        ids = np.array([2 | (3 << 5) | (7 << 7) | (2047 << 13) | (255 << 24)], dtype=np.uint64)
        np.testing.assert_array_equal(
            conditions.decode_cell_ids(ids, 2), [[2, -1, 7, 2047, 255]]
        )
        with self.assertRaisesRegex(ValueError, "system"):
            conditions.decode_cell_ids(ids, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            conditions.decode_cell_ids(np.array([1 << 32], dtype=np.uint64), 1)

    def test_in_time_window_uses_flight_correction(self):
        # t=0 @ origin -> tof 0 (in); t=8 @ 1.5 m -> tof ~3.0 (in after correction);
        # t=20 @ origin -> tof 20 (out); huge time -> out (out-of-time tail).
        mask = conditions.in_time_mask(
            np.array([0.0, 8.0, 20.0, 1e6]),
            np.array([0.0, 1500.0, 0.0, 0.0]), np.zeros(4), np.zeros(4))
        np.testing.assert_array_equal(mask, [True, True, False, False])
        # Bounds are inclusive at exactly -0.5 and 15.0 ns.
        edge = conditions.in_time_mask(
            np.array([-0.5, 15.0, -0.51, 15.01]), np.zeros(4), np.zeros(4), np.zeros(4))
        np.testing.assert_array_equal(edge, [True, True, False, False])

    def test_equal_totals_do_not_hide_sensor_reassignment(self):
        expected = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 1, 0]], dtype=np.int64)
        conditions.require_matching_counts(expected, expected[::-1])
        with self.assertRaisesRegex(ValueError, "1 missing and 1 extra"):
            conditions.require_matching_counts(expected, expected[[0, 0]])
        empty = np.empty((0, 5), dtype=np.int64)
        conditions.require_matching_counts(empty, empty)

    def test_flipped_sources_cannot_cross_splits(self):
        data = manifest()
        second = copy.deepcopy(data["events"][0])
        second.update(event_id="event_1", split="test")
        # Use fresh MUPLUS cycles, but reuse flipped training cycles.
        for source in second["sources"]["MUPLUS"]:
            source["cycle"] += 1000
            source["path"] = f"MUPLUS/{source['cycle']}.root"
        data["events"].append(second)
        with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
            conditions.validate_manifest(data, Path("/tmp"))

    def test_duplicate_and_aliased_sources_are_rejected(self):
        data = manifest()
        data["events"][0]["sources"]["MUPLUS"][1] = copy.deepcopy(
            data["events"][0]["sources"]["MUPLUS"][0]
        )
        with self.assertRaisesRegex(ValueError, "Repeated cycle"):
            conditions.validate_manifest(data, Path("/tmp"))
        data = manifest()
        data["events"][0]["sources"]["MUPLUS"][1]["path"] = "MUPLUS/0.root"
        with self.assertRaisesRegex(ValueError, "Conflicting identities"):
            conditions.validate_manifest(data, Path("/tmp"))

    def test_preparation_preserves_events_empty_collections_and_both_polarities(self):
        data = manifest()
        second = copy.deepcopy(data["events"][0])
        second["event_id"] = "event_1"
        for sources in second["sources"].values():
            for source in sources:
                source["entry"] = 1
        data["events"].append(second)

        def read(path, entry):
            result = {short: Counter() for short in conditions.COLLECTIONS}
            if Path(path).stem == "0":
                sensor = 0 if Path(path).parent.name == "MUPLUS" else 1
                result["VBC"][(1, 0, 0, 0, sensor)] = entry + 1
            return result

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "input.json"
            source.write_text(json.dumps(data))
            output = base / "conditions"
            with patch.object(conditions, "read_source_counts", side_effect=read):
                conditions.prepare(source, output)
            for event_id, n in (("event_0", 1), ("event_1", 2)):
                rows = np.load(output / "train" / event_id / "VBC_conditions.npy")
                self.assertEqual(conditions.sensor_counts(rows), Counter({
                    (1, 0, 0, 0, 0): n, (1, 0, 0, 0, 1): n,
                }))
                self.assertEqual(np.load(output / "train" / event_id / "VEC_conditions.npy").shape, (0, 5))
            with self.assertRaises(FileExistsError):
                conditions.prepare(source, output)

    def test_failed_read_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            source.write_text(json.dumps(manifest()))
            output = Path(directory) / "conditions"
            with patch.object(conditions, "read_source_counts", side_effect=ValueError("unreadable")):
                with self.assertRaisesRegex(ValueError, "unreadable"):
                    conditions.prepare(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).glob(".count_conditions_*")), [])


if __name__ == "__main__":
    unittest.main()
