#!/usr/bin/env python3
"""Tests for the above-one-BX manifest builder."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from prepare import prepare


class HighOccupancyPrepareTest(unittest.TestCase):
    def arguments(self, output, files=8, events=2):
        return SimpleNamespace(
            shared_sim_root="/shared", recovery_muminus_root="/recovered",
            files_per_polarity=files, events=events, seed=12345,
            cohort="high_test", output=str(output),
        )

    @patch("prepare.complete_inventory")
    def test_events_are_independent_polarity_draws_without_replacement(
            self, inventory):
        cycles = list(range(10))
        inventory.return_value = (
            {cycle: Path(f"/plus/{cycle}.root") for cycle in cycles},
            {cycle: Path(f"/minus/{cycle}.root") for cycle in cycles},
            cycles,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "scan"
            prepare(self.arguments(output))
            scan = json.loads((output / "scan.json").read_text())
            manifest = json.loads((output / "n0008/sources.json").read_text())

        self.assertEqual(scan["events_per_point"], 2)
        self.assertEqual(scan["points"][0]["fraction_of_full_bx_reference"], 8 / 1666)
        for event in manifest["events"]:
            for polarity in ("MUPLUS", "MUMINUS"):
                selected = [row["cycle"] for row in event["sources"][polarity]]
                self.assertEqual(len(selected), 8)
                self.assertEqual(len(set(selected)), 8)

    @patch("prepare.complete_inventory")
    def test_refuses_more_files_than_the_repaired_inventory(self, inventory):
        cycles = list(range(10))
        inventory.return_value = ({}, {}, cycles)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "contains only 10"):
                prepare(self.arguments(Path(temporary) / "scan", files=11))


if __name__ == "__main__":
    unittest.main()
