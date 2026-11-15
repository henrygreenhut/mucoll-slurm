#!/usr/bin/env python3

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import prepare_track_checkpoint as checkpoint


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
            checkpoint.prepare(training, output, 12345, "test_checkpoint")
            sources = json.loads((output / "n0167" / "sources.json").read_text())
            excluded = set(training_cycles["MUPLUS"]) | set(training_cycles["MUMINUS"])
            for polarity in checkpoint.POLARITIES:
                selected = {
                    row["cycle"] for row in sources["events"][0]["sources"][polarity]
                }
                self.assertEqual(len(selected), 167)
                self.assertFalse(selected & excluded)
            self.assertEqual(sources["sampling"]["training_overlap"], "zero against union of training cycle IDs")


if __name__ == "__main__":
    unittest.main()
