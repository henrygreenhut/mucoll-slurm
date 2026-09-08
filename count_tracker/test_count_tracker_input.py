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
        labels = writer.validate_count_rows(self.rows, 2, self.expected)
        np.testing.assert_array_equal(self.rows, original)
        self.assertEqual(len(labels), 2)
        np.testing.assert_array_equal(labels, [[2, -1, 0, 0, 3]] * 2)

    def test_reassigned_hits_fail_even_when_total_count_matches(self):
        self.rows[1, 9] = 4
        with self.assertRaisesRegex(ValueError, "1 missing, 1 extra"):
            writer.validate_count_rows(self.rows, 2, self.expected)

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
        labels = writer.validate_count_rows(np.empty((0, 10)), 2, Counter())
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
