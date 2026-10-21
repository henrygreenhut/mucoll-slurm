#!/usr/bin/env python3

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare


class PrepareTests(unittest.TestCase):
    def make_sources(self, root):
        for polarity in ("MUPLUS", "MUMINUS"):
            directory = root / polarity
            directory.mkdir(parents=True)
            for cycle in range(6):
                (directory / f"bib_sim_{cycle}.edm4hep.root").write_bytes(
                    f"{polarity}-{cycle}".encode())
        (root / "MUMINUS" / "bib_sim_6291.edm4hep.root").write_bytes(b"bad")

    def test_batches_are_explicit_deterministic_and_per_polarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources"
            self.make_sources(sources)
            out_a, out_b = root / "a", root / "b"
            for output in (out_a, out_b):
                prepare.prepare(sources, output, events=5, batch_size=2,
                                n_files=2, seed=7, cohort="variance")

            scan = json.loads((out_a / "scan.json").read_text())
            self.assertEqual(scan["planned_events"], 5)
            self.assertEqual([point["events"] for point in scan["points"]], [2, 2, 1])
            inventory = json.loads((out_a / "source_inventory.json").read_text())
            self.assertEqual(inventory["counts"], {"MUPLUS": 6, "MUMINUS": 6})
            self.assertNotIn(
                6291, [row["cycle"] for row in inventory["sources"]["MUMINUS"]])

            manifests_a = [
                json.loads(Path(point["source_manifest"]).read_text())
                for point in scan["points"]
            ]
            scan_b = json.loads((out_b / "scan.json").read_text())
            manifests_b = [
                json.loads(Path(point["source_manifest"]).read_text())
                for point in scan_b["points"]
            ]
            draws_a = [event["sources"] for manifest in manifests_a
                       for event in manifest["events"]]
            draws_b = [event["sources"] for manifest in manifests_b
                       for event in manifest["events"]]
            normalized_a = [{p: [r["cycle"] for r in draw[p]] for p in draw}
                            for draw in draws_a]
            normalized_b = [{p: [r["cycle"] for r in draw[p]] for p in draw}
                            for draw in draws_b]
            self.assertEqual(normalized_a, normalized_b)
            for draw in normalized_a:
                for polarity in ("MUPLUS", "MUMINUS"):
                    self.assertEqual(len(draw[polarity]), 2)
                    self.assertEqual(len(set(draw[polarity])), 2)


if __name__ == "__main__":
    unittest.main()
