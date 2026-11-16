#!/usr/bin/env python3

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_track_checkpoint as checkpoint
import track_checkpoint_report as cohort_report


class TrackCheckpointTests(unittest.TestCase):
    def test_selection_excludes_union_and_is_deterministic(self):
        records = {
            polarity: {
                cycle: {"cycle": cycle, "path": f"/{polarity}/{cycle}", "entry": 0}
                for cycle in range(500)
            }
            for polarity in checkpoint.POLARITIES
        }
        excluded = set(range(40))
        first = checkpoint.select_evaluation_sources(records, excluded, 123, "check")
        second = checkpoint.select_evaluation_sources(records, excluded, 123, "check")
        self.assertEqual(first, second)
        for polarity in checkpoint.POLARITIES:
            cycles = [row["cycle"] for row in first[polarity]]
            self.assertEqual(len(cycles), checkpoint.FILES_PER_POLARITY)
            self.assertFalse(set(cycles) & excluded)
            self.assertEqual(len(cycles), len(set(cycles)))
        self.assertNotEqual(
            [row["cycle"] for row in first["MUPLUS"]],
            [row["cycle"] for row in first["MUMINUS"]],
        )
        third = checkpoint.select_evaluation_sources(
            records, excluded, 123, "check", event_index=1
        )
        self.assertNotEqual(first, third)

    def test_training_manifest_loader_rejects_wrong_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.json"
            path.write_text(json.dumps({"kind": "wrong"}))
            with self.assertRaises(ValueError):
                checkpoint.load_training_sources(path)

    def test_prepare_writes_source_disjoint_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory_rows = {}
            for polarity in checkpoint.POLARITIES:
                directory = root / polarity
                directory.mkdir()
                rows = []
                for cycle in range(600):
                    path = directory / f"bib_sim_{cycle}.edm4hep.root"
                    path.write_bytes(b"source")
                    rows.append({
                        "cycle": cycle, "path": str(path), "entry": 0,
                        "size_bytes": path.stat().st_size,
                    })
                inventory_rows[polarity] = rows
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({
                "kind": "norm42-independent-polarity-source-inventory",
                "sources": inventory_rows,
            }))
            digest = hashlib.sha256(inventory.read_bytes()).hexdigest()
            training_cycles = {
                "MUPLUS": list(range(167)),
                "MUMINUS": list(range(100, 267)),
            }
            training = root / "training.json"
            training.write_text(json.dumps({
                "kind": "norm42_10pct_training_development_corpus",
                "selection": {"files_per_polarity": 167},
                "source_inventory": str(inventory),
                "source_inventory_sha256": digest,
                "sources": {
                    polarity: [{"cycle": cycle} for cycle in cycles]
                    for polarity, cycles in training_cycles.items()
                },
            }))
            output = root / "checkpoint"
            checkpoint.prepare(
                training, output, 12345, "test_checkpoint", events=3
            )
            sources = json.loads((output / "n0167" / "sources.json").read_text())
            excluded = set(training_cycles["MUPLUS"]) | set(training_cycles["MUMINUS"])
            self.assertEqual(len(sources["events"]), 3)
            for event in sources["events"]:
                for polarity in checkpoint.POLARITIES:
                    selected = {
                        row["cycle"] for row in event["sources"][polarity]
                    }
                    self.assertEqual(len(selected), 167)
                    self.assertFalse(selected & excluded)
            self.assertEqual(sources["sampling"]["training_overlap"], "zero against union of training cycle IDs")
            self.assertIn("reuse between events is allowed", sources["sampling"]["across_events"])
            overlaps = json.loads((output / "source_overlap.json").read_text())
            self.assertEqual(len(overlaps["event_ids"]), 3)
            self.assertEqual(
                [row[index] for index, row in enumerate(
                    overlaps["pairwise_overlap_by_polarity"]["MUPLUS"]
                )],
                [167, 167, 167],
            )

    def test_report_combines_checkpoint_and_followup(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "checkpoint", Path(tmp) / "followup"]
            for index, root in enumerate(roots):
                point = root / "n0167"
                point.mkdir(parents=True)
                (root / "scan.json").write_text(json.dumps({"index": index}))
                (point / "sources.json").write_text(json.dumps({
                    "n_files_per_polarity": 167,
                    "overlay_normalization": {
                        "fraction_of_full_bx_reference": 167 / 1666,
                    },
                    "events": [{
                        "event_id": f"event_{index}",
                        "sources": {
                            polarity: [
                                {"cycle": cycle + 200 * index}
                                for cycle in range(167)
                            ]
                            for polarity in checkpoint.POLARITIES
                        },
                    }],
                }))

            def fake_read_arm(_point, event_id, arm):
                event_index = int(event_id[-1])
                tracks = {
                    "SIM": 100 + 10 * event_index,
                    "COUNT_FULL": 120 + 10 * event_index,
                    "COUNT_10PCT": 100 + 10 * event_index,
                }[arm]
                return {
                    "entering_digitization": 1000,
                    "digitized": 200,
                    "survival_fraction": 0.2,
                    "si_tracks": tracks,
                    "regional_si_tracks": {
                        region: tracks // 5 for region in cohort_report.REGIONS
                    },
                }

            with patch.object(cohort_report, "read_arm", side_effect=fake_read_arm):
                result = cohort_report.report(roots)
            self.assertEqual(result["n_events"], 2)
            self.assertEqual(result["aggregate_arms"]["SIM"]["si_tracks"], 210)
            self.assertEqual(
                result["track_ratio_summary"]["COUNT_10PCT"]
                ["ratio_of_aggregate_tracks"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
