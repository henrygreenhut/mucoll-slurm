#!/usr/bin/env python3
"""Tests for the source-disjoint full-BX manifest builder."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare


class SourceDisjointFullBxTest(unittest.TestCase):
    def test_partition_is_deterministic_disjoint_and_complete(self):
        cycles = list(range(6666))
        first, unused_first = prepare.partition_cycles(cycles, 12345, "cohort")
        second, unused_second = prepare.partition_cycles(cycles, 12345, "cohort")
        self.assertEqual(first, second)
        self.assertEqual(unused_first, unused_second)
        self.assertEqual([len(group) for group in first], [1666] * 4)
        self.assertEqual(len(set().union(*map(set, first))), 6664)
        self.assertEqual(len(unused_first), 2)

    def test_manifest_keeps_polarity_counterparts_in_one_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plus = {
                cycle: root / "MUPLUS" / f"bib_sim_{cycle}.edm4hep.root"
                for cycle in range(6666)
            }
            minus = {
                cycle: root / "MUMINUS" / f"bib_sim_{cycle}.edm4hep.root"
                for cycle in range(6666)
            }
            args = SimpleNamespace(
                shared_sim_root=str(root / "shared"),
                recovery_muminus_root=str(root / "recovery"),
                seed=12345,
                cohort="unit",
                output=str(root / "output"),
            )
            with patch.object(
                    prepare, "complete_inventory",
                    return_value=(plus, minus, list(range(6666)))):
                prepare.prepare(args)

            manifest = json.loads(
                (root / "output/n1666/sources.json").read_text())
            self.assertEqual(
                [event["split"] for event in manifest["events"]],
                ["train", "train", "val", "test"],
            )
            seen = set()
            for event in manifest["events"]:
                plus = [row["cycle"] for row in event["sources"]["MUPLUS"]]
                minus = [row["cycle"] for row in event["sources"]["MUMINUS"]]
                self.assertEqual(plus, minus)
                self.assertFalse(seen.intersection(plus))
                seen.update(plus)
            self.assertEqual(len(seen), 6664)


if __name__ == "__main__":
    unittest.main()
