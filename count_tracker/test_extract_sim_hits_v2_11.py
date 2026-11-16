import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "v2_11"))
import extract_sim_hits_v2_11 as extract  # noqa: E402


class FakeBranch:
    def __init__(self, values):
        self.values = values

    def array(self, *, entry_start, entry_stop, library):
        if (entry_start, entry_stop, library) != (0, 1, "ak"):
            raise AssertionError("unexpected uproot branch request")
        return [self.values]


class FakeCollection:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, branch):
        field = branch.removeprefix(f"{self.collection}.")
        offset = extract.FIELDS.index(field)
        return FakeBranch(np.array([offset, offset + 1], dtype=np.float64))


class FakeEvents:
    num_entries = 1

    def __getitem__(self, collection):
        return FakeCollection(collection)


class FakeRoot:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __getitem__(self, key):
        if key != "events":
            raise KeyError(key)
        return FakeEvents()


class ExtractSimHitsV211Test(unittest.TestCase):
    def test_read_entry_opens_source_once_for_all_collections(self):
        opened = []

        def fake_open(path):
            opened.append(path)
            return FakeRoot()

        uproot = types.SimpleNamespace(open=fake_open)
        with patch.dict(sys.modules, {"uproot": uproot}):
            result = extract.read_entry("source.root", 0)

        self.assertEqual(opened, ["source.root"])
        expected = {collection for _, collection in extract.COLLECTIONS.values()}
        self.assertEqual(set(result), expected)
        for values in result.values():
            self.assertEqual(len(values), len(extract.FIELDS))
            np.testing.assert_array_equal(values[0], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
