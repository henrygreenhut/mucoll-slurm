#!/usr/bin/env python3
"""Event-pair grouping and provenance checks for the preliminary split."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import count_tracker_pilot_split as pilot


class PilotSplitTests(unittest.TestCase):
    def test_prepare_keeps_both_arms_together_and_recomputes_pt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "export.json"
            events = []
            for index in range(4):
                arms = {sample: {"path": f"/{sample}/{index}.root", "sha256": f"{sample}{index}",
                                 "helix": [[0.1, 0.0003, 0.0, 0.2, 1.0]]}
                        for sample in ("SIM", "COUNT")}
                events.append({"event_id": f"e{index}", "split": "test", "arms": arms})
            source.write_text(json.dumps({
                "kind": "count_tracker_read_only_helix_export",
                "construction": "norm42_reservoir",
                "conditions_manifest_sha256": "original",
                "state_columns": list(pilot.STATE_COLUMNS),
                "track_collection": "SiTracks", "track_state": "AtIP",
                "events": events,
            }))
            output = root / "stores"
            summary = pilot.prepare(source, output,
                                    {"train": 2, "val": 1, "test": 1}, seed=7)
            self.assertEqual(set(summary["assignment"]), {f"e{i}" for i in range(4)})
            for split, expected in (("train", 2), ("val", 1), ("test", 1)):
                manifests = [json.loads((output / f"norm42_reservoir_preliminary_{sample}_{split}.json").read_text())
                             for sample in ("SIM", "COUNT")]
                self.assertEqual([event["event_id"] for event in manifests[0]["events"]],
                                 [event["event_id"] for event in manifests[1]["events"]])
                self.assertEqual(manifests[0]["n_events"], expected)
                with np.load(output / f"norm42_reservoir_preliminary_SIM_{split}.npz") as data:
                    self.assertAlmostEqual(float(data["tracks"][0, 0, 0]), 5.0)
            with self.assertRaises(FileExistsError):
                pilot.prepare(source, output, {"train": 2, "val": 1, "test": 1}, seed=7)


if __name__ == "__main__":
    unittest.main()
