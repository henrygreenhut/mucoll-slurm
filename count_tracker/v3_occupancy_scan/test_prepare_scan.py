"""Unit checks for nested Norm42 occupancy manifests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_scan import prepare


class PrepareScanTests(unittest.TestCase):
    @patch("prepare_scan.read_pools")
    def test_each_replicate_is_a_nested_source_prefix(self, read_pools):
        pools = {}
        for split, start in (("train", 0), ("val", 2000), ("test", 4000)):
            pools[split] = {}
            for polarity in ("MUPLUS", "MUMINUS"):
                pools[split][polarity] = {
                    cycle: {"cycle": cycle, "path": f"/{polarity}/{cycle}.root", "entry": 0}
                    for cycle in range(start, start + 2000)
                }
        read_pools.return_value = (pools, {"path": "/pools", "manifest_sha256": "x"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scan"
            prepare("/unused", output, [10, 100, 1666], 5, 12345, "scan")
            scan = json.loads((output / "scan.json").read_text())
            self.assertEqual([point["files_per_polarity"] for point in scan["points"]],
                             [10, 100, 1666])
            manifests = [
                json.loads((output / point["tag"] / "sources.json").read_text())
                for point in scan["points"]
            ]
            for event_index in range(5):
                for polarity in ("MUPLUS", "MUMINUS"):
                    ten = manifests[0]["events"][event_index]["sources"][polarity]
                    hundred = manifests[1]["events"][event_index]["sources"][polarity]
                    full = manifests[2]["events"][event_index]["sources"][polarity]
                    self.assertEqual(ten, hundred[:10])
                    self.assertEqual(hundred, full[:100])


if __name__ == "__main__":
    unittest.main()
