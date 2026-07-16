#!/usr/bin/env python3
import unittest

import numpy as np

from pfn_synthetic42_validate import paired_definitions


class Synthetic42ValidationTests(unittest.TestCase):
    def test_pairs_share_cycles_and_test_pairs_are_disjoint(self):
        definitions = paired_definitions(
            np.random.default_rng(4), np.arange(100), 5, 4,
            null_test=False, disjoint=True)
        groups = []
        for index in range(0, len(definitions), 2):
            class0, class1 = definitions[index:index + 2]
            self.assertEqual(class0[0], 0)
            self.assertEqual(class1[0], 1)
            np.testing.assert_array_equal(
                class0[2]["cycles"], class1[2]["cycles"])
            self.assertEqual(class0[2]["kind"], "synthetic")
            self.assertEqual(class1[2]["kind"], "original")
            groups.append(set(class0[2]["cycles"]))
        for first in range(len(groups)):
            for second in range(first):
                self.assertFalse(groups[first] & groups[second])

    def test_null_uses_two_independent_synthetic_rotations(self):
        definitions = paired_definitions(
            np.random.default_rng(8), np.arange(20), 1, 5,
            null_test=True)
        self.assertEqual(definitions[0][2]["kind"], "synthetic")
        self.assertEqual(definitions[1][2]["kind"], "synthetic")
        self.assertNotEqual(definitions[0][2]["angle_seed"],
                            definitions[1][2]["angle_seed"])


if __name__ == "__main__":
    unittest.main()
