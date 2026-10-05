#!/usr/bin/env python3
"""Tests for aggregate paired-cohort reporting."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import count_tracker_cohort_report as report


class CohortReportTests(unittest.TestCase):
    def test_all_combines_classifier_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conditions = root / "conditions"
            conditions.mkdir()
            conditions.joinpath("manifest.json").write_text(json.dumps({
                "manifest": {"construction": "norm42"},
                "events": [
                    {"event_id": "train_event", "split": "train"},
                    {"event_id": "test_event", "split": "test"},
                ],
            }))

            def arm(_):
                collections = {
                    name: {"entering_digitization": 10, "digitized": 4}
                    for name in report.COLLECTIONS
                }
                return {"collections": collections,
                        "total": {"si_tracks": 3}}

            with patch.object(report, "arm_report", side_effect=arm):
                result = report.summarize(conditions, root / "events", "all")
            self.assertEqual(result["events"], 2)
            self.assertEqual(result["split"], "all")
            self.assertEqual(result["aggregate"]["SIM"]["total"]["si_tracks"], 6)
            self.assertEqual(result["aggregate"]["COUNT"]["total"]["si_tracks"], 6)
            self.assertEqual(result["count_over_sim_tracks"], 1.0)


if __name__ == "__main__":
    unittest.main()
