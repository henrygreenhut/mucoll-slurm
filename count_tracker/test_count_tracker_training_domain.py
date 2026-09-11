"""Checks for the HDF-to-training-array verifier."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    unittest.main()
