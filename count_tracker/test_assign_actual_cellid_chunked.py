#!/usr/bin/env python3
"""Tests for bounded-memory v2.11 CellID assignment output."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from v2_11.assign_actual_cellid_chunked import COUNT_KEYS, convert_collection


class FakeNative:
    @staticmethod
    def convert_rows(rows, _input_format, _system_id, _geom, _k, _ref_lookup):
        rows = np.asarray(rows)
        valid = rows[:, 0] >= 0
        output = np.column_stack((rows[valid], rows[valid, :2]))
        counts = {key: 0 for key in COUNT_KEYS}
        counts["written_rows"] = int(valid.sum())
        counts["invalid_rows"] = int((~valid).sum())
        counts["multi_assigned_rows"] = int(np.count_nonzero(rows[valid, 1] == 2))
        return output.astype(np.float64), counts


class ChunkedCellIDTest(unittest.TestCase):
    def run_conversion(self, rows, chunk_size):
        with tempfile.TemporaryDirectory() as work:
            output = Path(work) / "assigned.npy"
            counts = convert_collection(
                FakeNative, rows, output,
                input_format="10col", system_id=1, geom=None, k=64,
                ref_lookup=None, chunk_size=chunk_size, label="TEST",
            )
            result = np.load(output, allow_pickle=False)
            leftovers = list(Path(work).glob(".*.maximum"))
        return result, counts, leftovers

    def test_chunking_matches_one_shot_with_dropped_rows(self):
        rows = np.arange(40, dtype=np.float64).reshape(5, 8)
        rows[1, 0] = -1
        rows[3, 0] = -1
        expected, expected_counts = FakeNative.convert_rows(
            rows, "10col", 1, None, 64, None
        )
        actual, counts, leftovers = self.run_conversion(rows, chunk_size=2)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(counts, expected_counts)
        self.assertEqual(leftovers, [])

    def test_all_valid_output_preserves_order(self):
        rows = np.arange(48, dtype=np.float64).reshape(6, 8)
        expected, expected_counts = FakeNative.convert_rows(
            rows, "10col", 1, None, 64, None
        )
        actual, counts, leftovers = self.run_conversion(rows, chunk_size=4)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(counts, expected_counts)
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
