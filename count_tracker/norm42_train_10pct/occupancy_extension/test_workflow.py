#!/usr/bin/env python3
"""Unit tests for occupancy-extension task expansion and report merging."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import merge_reports
import plan_task
import prepare_reuse
import prepare_priority2_partition


class TestOccupancyExtension(unittest.TestCase):
    def test_priority2_partition_is_disjoint_and_reuses_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            points = []
            for tag, count in (("n0010", 5), ("n0464", 5)):
                point_root = source / tag
                point_root.mkdir()
                (point_root / "sources.json").write_text("{}\n")
                (point_root / "conditions").mkdir()
                (point_root / "count_samples").mkdir()
                points.append({
                    "tag": tag, "reconstruction": "five-region",
                    "events": [
                        {"event_id": f"{tag}_{index}", "split": "test"}
                        for index in range(count)
                    ],
                    "training_overlap": [{"event_id": f"{tag}_{index}"}
                                         for index in range(count)],
                })
            (source / "plan.json").write_text(json.dumps({"points": points}))
            args = type("Args", (), {
                "source_root": source,
                "fast_root": root / "fast",
                "remaining_root": root / "remaining",
            })
            prepare_priority2_partition.prepare(args)
            fast = json.loads((root / "fast" / "plan.json").read_text())
            remaining = json.loads(
                (root / "remaining" / "plan.json").read_text())
            fast_ids = {event["event_id"] for event in fast["points"][-1]["events"]}
            remaining_ids = {
                event["event_id"]
                for event in remaining["points"][0]["events"]
            }
            self.assertEqual(fast_ids, {"n0464_0", "n0464_1"})
            self.assertEqual(remaining_ids, {"n0464_2", "n0464_3", "n0464_4"})
            self.assertTrue(fast_ids.isdisjoint(remaining_ids))
            self.assertTrue((root / "fast" / "n0464" / "conditions").is_symlink())

    def test_high_checkpoint_and_completion_are_disjoint(self):
        first = prepare_reuse.POINTS["high_first"][0]
        remaining = prepare_reuse.POINTS["high_remaining"][0]
        self.assertEqual(first[5], (0,))
        self.assertEqual(remaining[5], (1,))
        self.assertTrue(set(first[5]).isdisjoint(remaining[5]))

    def test_task_expansion(self):
        plan = {
            "points": [
                {
                    "tag": "n0010", "reconstruction": "unsplit",
                    "sim_source_point": "/sim/a",
                    "events": [{"event_id": "a", "split": "test"}],
                },
                {
                    "tag": "n0036", "reconstruction": "five-region",
                    "sim_source_point": "/sim/b",
                    "events": [
                        {"event_id": "b0", "split": "test"},
                        {"event_id": "b1", "split": "test"},
                    ],
                },
            ]
        }
        self.assertEqual(len(plan_task.tasks(plan, "point")), 2)
        self.assertEqual(len(plan_task.tasks(plan, "event")), 3)
        regions = plan_task.tasks(plan, "region")
        self.assertEqual(len(regions), 10)
        self.assertEqual(regions[0]["region_tag"], "theta_000_030")
        self.assertEqual(regions[-1]["region_tag"], "theta_150_180")

    def test_merge_combines_two_fresh_ten_percent_waves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh_paths = []
            for wave, sim, count in ((1, [10, 14], [12, 16]),
                                     (2, [18], [21])):
                path = root / f"fresh{wave}.json"
                path.write_text(json.dumps({"results": [{
                    "fraction_of_full_bx_reference": 0.1,
                    "files_per_polarity": 167,
                    "paired": [
                        {"SIM": {"si_tracks": left},
                         "COUNT": {"si_tracks": right}}
                        for left, right in zip(sim, count)
                    ],
                }]}))
                fresh_paths.append(path)
            reuse = root / "reuse.json"
            reuse.write_text(json.dumps({"results": [{
                "fraction_of_full_bx_reference": 0.01,
                "files_per_polarity": 17,
                "reconstruction": "unsplit",
                "sim": {"values": [2, 4]},
                "count_10pct": {"values": [3, 5]},
            }]}))
            output = root / "data.csv"
            merge_reports.merge(fresh_paths, [reuse], output)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            ten = rows[1]
            self.assertEqual(int(ten["events"]), 3)
            self.assertAlmostEqual(float(ten["paired_sim_mean_tracks"]), 14.0)
            self.assertAlmostEqual(float(ten["count_10pct_mean_tracks"]), 49 / 3)


if __name__ == "__main__":
    unittest.main()
