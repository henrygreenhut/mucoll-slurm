#!/usr/bin/env python3
"""Checks for event-weighted AUC and grouped uncertainty calculations."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import count_tracker_metrics as metrics
import count_tracker_multiplicity as multiplicity
from count_tracker_track_features import RAW_FEATURES


class MetricTests(unittest.TestCase):
    def test_auc_handles_direction_ties_and_event_weights(self):
        self.assertAlmostEqual(metrics.weighted_auc([0, 0, 1, 1], [1, 2, 2, 3]), 0.875)
        self.assertAlmostEqual(metrics.weighted_auc([0, 1], [2, 1]), 0.0)
        self.assertAlmostEqual(metrics.weighted_auc([0, 1], [1, 1]), 0.5)
        self.assertAlmostEqual(metrics.weighted_auc([0, 0, 1], [1, 3, 2], [1, 3, 1]), 0.25)
        with self.assertRaises(ValueError):
            metrics.weighted_auc([0, 0], [1, 2])

    def test_pair_bootstrap_keeps_whole_event_groups(self):
        labels = [0, 0, 1, 1]
        scores = [1, 2, 3, 4]
        groups = [0, 1, 2, 3]
        result = metrics.paired_event_bootstrap(
            labels, scores, np.ones(4), groups, n_events=2, n_draws=20)
        self.assertEqual(result["valid_draws"], 20)
        self.assertEqual((result["lower"], result["upper"]), (1.0, 1.0))

    def test_multiplicity_uses_test_events_and_exposes_signed_auc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sample, counts in (("SIM", [1, 2]), ("COUNT", [3, 4])):
                prefix = root / f"norm42_{sample}_test"
                np.savez(prefix.with_suffix(".npz"),
                         tracks=np.zeros((2, 4, len(RAW_FEATURES)), np.float32),
                         n_tracks=np.asarray(counts, np.int64))
                prefix.with_suffix(".json").write_text(json.dumps({
                    "schema_version": 2, "construction": "norm42", "sample": sample,
                    "split": "test", "track_collection": "SiTracks", "track_state": "AtIP",
                    "features": list(RAW_FEATURES), "n_events": 2,
                    "total_tracks": sum(counts), "conditions_manifest_sha256": "shared",
                    "events": [{"event_id": "e0"}, {"event_id": "e1"}],
                }))
            result = multiplicity.evaluate(root, "norm42", n_draws=20)
            self.assertEqual(result["auc_signed"], 1.0)
            self.assertEqual(result["paired_counts"][0]["COUNT"], 3)


if __name__ == "__main__":
    unittest.main()
