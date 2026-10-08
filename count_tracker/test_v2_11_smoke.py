import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent / "v2_11"))
from make_smoke_input_v2_11 import evenly_spaced_indices  # noqa: E402


class SmokeInputSelectionTest(unittest.TestCase):
    def test_small_collection_is_kept_completely(self):
        self.assertEqual(list(evenly_spaced_indices(3, 8)), [0, 1, 2])

    def test_large_collection_is_spanned_without_repetition(self):
        indices = list(evenly_spaced_indices(10, 4))
        self.assertEqual(indices, [0, 2, 5, 7])
        self.assertEqual(len(indices), len(set(indices)))
        self.assertTrue(all(0 <= index < 10 for index in indices))


if __name__ == "__main__":
    unittest.main()
