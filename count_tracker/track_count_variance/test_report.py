#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report


class ReportTests(unittest.TestCase):
    def test_paired_track_statistics(self):
        result = report.statistics([1, 3], [2, 6])
        self.assertEqual(result["SIM"]["mean"], 2.0)
        self.assertEqual(result["SIM"]["variance_ddof1"], 2.0)
        self.assertEqual(result["COUNT"]["mean"], 4.0)
        self.assertEqual(result["COUNT"]["variance_ddof1"], 8.0)
        self.assertEqual(result["paired"]["variance_ratio_count_over_sim"], 4.0)
        self.assertAlmostEqual(result["paired"]["pearson_correlation"], 1.0)
        self.assertEqual(
            result["paired"]["difference_count_minus_sim"]["variance_ddof1"], 2.0)

    def test_bootstrap_is_paired_and_reproducible(self):
        first = report.bootstrap_intervals([1, 2, 4], [2, 3, 8], 50, 123)
        second = report.bootstrap_intervals([1, 2, 4], [2, 3, 8], 50, 123)
        self.assertEqual(first, second)
        self.assertEqual(first["method"], "paired event percentile bootstrap")


if __name__ == "__main__":
    unittest.main()
