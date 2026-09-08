"""Source-selection checks for independent polarity draws and stable cohorts."""

import json
from pathlib import Path
import tempfile
import unittest

from count_tracker_manifest import POLARITIES, SPLITS, read_pools, select_events


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.pools = {
            split: {
                polarity: {
                    cycle: {"cycle": cycle, "path": f"/{polarity}/{cycle}.root", "entry": 0}
                    for cycle in range(1000 * index, 1000 * index + 600)
                } for polarity in POLARITIES
            } for index, split in enumerate(SPLITS)
        }
        self.counts = {split: 1 for split in SPLITS}

    def test_polarities_are_independent_draws_without_replacement(self):
        events = select_events(self.pools, self.counts, 17, "SIM_A", "norm1")
        for event in events:
            selected = {}
            for polarity in POLARITIES:
                selected[polarity] = {source["cycle"] for source in event["sources"][polarity]}
                self.assertEqual(len(selected[polarity]), 420)
                self.assertTrue(selected[polarity] <= self.pools[event["split"]][polarity].keys())
            self.assertNotEqual(selected["MUPLUS"], selected["MUMINUS"])

    def test_expanding_a_split_does_not_change_existing_events(self):
        original = select_events(self.pools, self.counts, 17, "SIM_A", "norm1")
        larger = select_events(self.pools, {**self.counts, "train": 3}, 17, "SIM_A", "norm1")
        indexed = {(event["split"], event["event_id"]): event for event in larger}
        for event in original:
            self.assertEqual(event, indexed[(event["split"], event["event_id"])])

    def test_cohort_name_changes_draws_even_with_same_seed(self):
        first = select_events(self.pools, self.counts, 17, "SIM_A", "norm1")
        second = select_events(self.pools, self.counts, 17, "SIM_B", "norm1")
        self.assertNotEqual(first[0]["sources"], second[0]["sources"])

    def test_insufficient_pool_is_an_error_not_sampling_with_replacement(self):
        self.pools["train"]["MUPLUS"] = {}
        with self.assertRaisesRegex(ValueError, "Fewer than 420"):
            select_events(self.pools, self.counts, 17, "SIM_A", "norm1")

    def test_norm42_draws_ten_files_and_has_distinct_event_identity(self):
        unique = select_events(self.pools, self.counts, 17, "SIM_A", "norm1")
        reused = select_events(self.pools, self.counts, 17, "SIM_A", "norm42")
        for left, right in zip(unique, reused):
            self.assertNotEqual(left["event_id"], right["event_id"])
            for polarity in POLARITIES:
                cycles = {source["cycle"] for source in right["sources"][polarity]}
                self.assertEqual(len(cycles), 10)
                self.assertTrue(cycles <= self.pools[right["split"]][polarity].keys())
            self.assertNotEqual(right["sources"]["MUPLUS"], right["sources"]["MUMINUS"])

    def test_both_libraries_must_follow_the_shared_pool_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({
                "splits": {split: [index] for index, split in enumerate(SPLITS)}
            }))
            for library in ("norm1", "norm42"):
                for index, split in enumerate(SPLITS):
                    for polarity in POLARITIES:
                        path = root / library / split / polarity
                        path.mkdir(parents=True)
                        (path / f"cycle_{index:06d}__source.root").touch()
            first, provenance = read_pools(root, "norm1")
            second, other_provenance = read_pools(root, "norm42")
            self.assertEqual(provenance, other_provenance)
            self.assertEqual(set(first["train"]["MUPLUS"]), set(second["train"]["MUPLUS"]))
            path = root / "norm42" / "test" / "MUMINUS"
            (path / "cycle_000000__wrong_split.root").touch()
            with self.assertRaisesRegex(ValueError, "differs from its manifest"):
                read_pools(root, "norm1")


if __name__ == "__main__":
    unittest.main()
