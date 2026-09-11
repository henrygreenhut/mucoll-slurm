"""Scientific input checks that do not require the detector container."""

from collections import Counter
import copy
import unittest
from unittest.mock import patch

import numpy as np

import count_tracker_input as writer


class AssignedHitTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.array([
            [0.0001, 10, 20, -500, -0.5, 2, -1, 0, 0, 3],
            [0.0002, 11, 21, -501, 15.0, 2, -1, 0, 0, 3],
        ], dtype=np.float64)
        self.expected = Counter({(2, -1, 0, 0, 3): 2})

    def test_broad_times_are_retained_without_conversion_or_selection(self):
        original = self.rows.copy()
        labels, stats = writer.validate_count_rows(self.rows, 2, self.expected)
        np.testing.assert_array_equal(self.rows, original)
        self.assertEqual(len(labels), 2)
        np.testing.assert_array_equal(labels, [[2, -1, 0, 0, 3]] * 2)
        self.assertEqual(stats, {"expected": 2, "written": 2, "missing": 0, "extra": 0})

    def test_reassignment_fails_at_default_zero_tolerance(self):
        self.rows[1, 9] = 4
        with self.assertRaisesRegex(ValueError, "1 missing, 1 extra"):
            writer.validate_count_rows(self.rows, 2, self.expected)

    def _uniform_rows(self, n, sensor=3):
        row = [0.001, 10, 20, -500, 0.0, 2, -1, 0, 0, sensor]
        return np.tile(row, (n, 1)).astype(np.float64)

    def test_small_reassignment_within_tolerance_is_recorded_not_raised(self):
        rows = self._uniform_rows(100)
        rows[0, 9] = 4  # one hit moved to a neighbour sensor
        expected = Counter({(2, -1, 0, 0, 3): 100})
        labels, stats = writer.validate_count_rows(rows, 2, expected, tolerance=0.05)
        self.assertEqual(len(labels), 100)
        self.assertEqual(stats, {"expected": 100, "written": 100, "missing": 1, "extra": 1})

    def test_dropped_hits_within_tolerance_are_recorded(self):
        rows = self._uniform_rows(98)  # two hits dropped by CellID assignment
        expected = Counter({(2, -1, 0, 0, 3): 100})
        labels, stats = writer.validate_count_rows(rows, 2, expected, tolerance=0.05)
        self.assertEqual(stats, {"expected": 100, "written": 98, "missing": 2, "extra": 0})

    def test_mismatch_beyond_tolerance_still_raises(self):
        rows = self._uniform_rows(100)
        rows[:10, 9] = 4  # 10% reassigned exceeds the 5% tolerance
        expected = Counter({(2, -1, 0, 0, 3): 100})
        with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
            writer.validate_count_rows(rows, 2, expected, tolerance=0.05)

    def test_out_of_range_and_fractional_ids_cannot_wrap_or_truncate(self):
        for sensor in (256, -1, 3.5):
            with self.subTest(sensor=sensor):
                rows = self.rows.copy()
                rows[0, 9] = sensor
                with self.assertRaises(ValueError):
                    writer.validate_count_rows(rows, 2, self.expected)

    def test_nonfinite_and_negative_energy_are_errors_not_dropped_hits(self):
        for column, value in ((0, -1), (1, np.nan), (4, np.inf)):
            with self.subTest(column=column, value=value):
                rows = self.rows.copy()
                rows[0, column] = value
                with self.assertRaises(ValueError):
                    writer.validate_count_rows(rows, 2, self.expected)

    def test_zero_hit_collection_and_signed_cell_id(self):
        labels, stats = writer.validate_count_rows(np.empty((0, 10)), 2, Counter())
        self.assertEqual(stats, {"expected": 0, "written": 0, "missing": 0, "extra": 0})
        self.assertEqual(writer.pack_cell_ids(labels).shape, (0,))
        cell_id = writer.pack_cell_ids(np.array([[2, -1, 7, 2047, 255]]))[0]
        self.assertEqual(int(cell_id), 2 | (3 << 5) | (7 << 7) | (2047 << 13) | (255 << 24))


class SimCopyTests(unittest.TestCase):
    def test_all_source_hits_are_copied_without_time_selection(self):
        class Hit:
            def __init__(self, time):
                self.time = time
                self.overlay = False
                self.relations = True

            def clone(self, relations):
                result = copy.copy(self)
                result.relations = relations
                return result

            def setOverlay(self, value):
                self.overlay = value

        class Collection(list):
            def push_back(self, value):
                self.append(value)

        hits = {name: [Hit(-10), Hit(0), Hit(20)] for _, name in writer.COLLECTIONS.values()}

        class Frame:
            def get(self, name):
                return hits[name]

        event = {"sources": {
            polarity: [{"path": f"{polarity}.root", "entry": 0}]
            for polarity in writer.POLARITIES
        }}
        output = {short: Collection() for short in writer.COLLECTIONS}
        with patch.object(writer, "read_frame", return_value=(object(), Frame())):
            writer.append_sim_hits(output, event)
        for short, (_, name) in writer.COLLECTIONS.items():
            self.assertEqual([hit.time for hit in output[short]], [-10, 0, 20] * 2)
            self.assertTrue(all(hit.overlay and not hit.relations for hit in output[short]))
            self.assertTrue(all(not hit.overlay and hit.relations for hit in hits[name]))


if __name__ == "__main__":
    unittest.main()
