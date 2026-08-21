import unittest

import numpy as np

import bib_split_muon_gen as split


class BibSplitMuonGenTest(unittest.TestCase):
    def test_cycle_filename(self):
        self.assertEqual(split.cycle_from_name("bib_gen_6291.edm4hep.root"), 6291)
        with self.assertRaises(ValueError):
            split.cycle_from_name("bib_gen_cycle_6291.root")

    def test_rotations_are_reproducible_and_all_random(self):
        first = split.rotation_angles("MUPLUS", 12, 4)
        second = split.rotation_angles("MUPLUS", 12, 4)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 42)
        self.assertTrue(np.all(first > 0.0))
        self.assertTrue(np.all(first < 2.0 * np.pi))
        self.assertFalse(np.any(first == 0.0))
        self.assertFalse(np.array_equal(first, split.rotation_angles("MUMINUS", 12, 4)))

    def test_rotation_preserves_radius(self):
        x = np.asarray([1.0, 0.0, 3.0])
        y = np.asarray([0.0, 2.0, 4.0])
        rotated_x, rotated_y = split.rotate_pair(x, y, 1.25)
        self.assertTrue(np.allclose(np.hypot(rotated_x, rotated_y), np.hypot(x, y)))

    def test_summary_requires_exact_partition(self):
        records = [{
            "entries": 5,
            "particles": 20,
            "bulk_entries": 3,
            "bulk_particles": 12,
            "muon_entries": [
                {"entry": 1, "particles": 3},
                {"entry": 4, "particles": 5},
            ],
            "muon_particles": 2,
            "muon_component_particles": 8,
        }]
        summary = split.summarize(records)
        self.assertEqual(summary["muon_component_entries"], 2)
        self.assertEqual(summary["bulk_entries"], 3)
        records[0]["bulk_particles"] = 11
        with self.assertRaises(RuntimeError):
            split.summarize(records)

    def test_muon_groups_are_reproducible_and_independent_by_polarity(self):
        histories = [
            {"cycle": index, "filename": "file", "entry": 0, "particles": 1}
            for index in range(split.MUON_PRODUCING_DECAYS)
        ]
        first = split.muon_group_sources(histories, "MUPLUS", 12)
        second = split.muon_group_sources(histories, "MUPLUS", 12)
        opposite = split.muon_group_sources(histories, "MUMINUS", 12)
        self.assertEqual(first, second)
        self.assertNotEqual(first, opposite)

    def test_muon_group_mean(self):
        self.assertAlmostEqual(split.MUON_GROUP_MEAN, 4.753580517019908)


if __name__ == "__main__":
    unittest.main()
