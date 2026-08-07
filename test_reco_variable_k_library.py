import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np

import reco_variable_k_library as library
import submit_reco_variable_k as submitter


class VariableKLibraryTest(unittest.TestCase):
    def test_overlay_counts_preserve_n420_mother_equivalents(self):
        expected = {1: 210, 5: 42, 7: 30, 10: 21, 21: 10}
        for reuse_k, files in expected.items():
            self.assertEqual(submitter.files_per_event(reuse_k), files)
            self.assertEqual(
                files * library.CHUNK_MOTHERS * reuse_k,
                library.MOTHER_EQUIVALENTS,
            )

    def make_bank(self, path, polarity):
        cycles = np.arange(6, dtype=np.int64)
        mothers_per_cycle = 280
        mother_cycles = np.repeat(cycles, mothers_per_cycle)
        mother_local = np.tile(
            np.arange(mothers_per_cycle, dtype=np.int32), len(cycles)
        )
        cycle_offsets = np.arange(
            0, len(mother_cycles) + 1, mothers_per_cycle, dtype=np.int64
        )
        mother_offsets = np.arange(len(mother_cycles) + 1, dtype=np.int64)
        with h5py.File(path, "w") as handle:
            handle.attrs["schema"] = "split-mother-gen-v1"
            handle.attrs["polarity"] = polarity
            handle.create_dataset("cycle_ids", data=cycles)
            handle.create_dataset("cycle_offsets", data=cycle_offsets)
            handle.create_dataset("mother_cycle_ids", data=mother_cycles)
            handle.create_dataset("mother_local_ids", data=mother_local)
            handle.create_dataset("mother_offsets", data=mother_offsets)
            particles = handle.create_group("particles")
            values = np.arange(len(mother_cycles), dtype=np.float32)
            for field in library.PARTICLE_FIELDS:
                particles.create_dataset(
                    field,
                    data=(
                        np.full(len(values), 22, dtype=np.int32)
                        if field == "pdg" else values
                    ),
                )

    def test_manifest_is_cycle_disjoint_and_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plus = root / "plus.h5"
            minus = root / "minus.h5"
            output = root / "manifest"
            self.make_bank(plus, "MUPLUS")
            self.make_bank(minus, "MUMINUS")
            library.prepare_manifest(
                Namespace(
                    muplus_bank=str(plus),
                    muminus_bank=str(minus),
                    outdir=str(output),
                    force=False,
                )
            )

            manifest = json.loads((output / "manifest.json").read_text())
            split_cycles = [
                set(manifest["splits"][split]["cycles"])
                for split in library.SPLITS
            ]
            self.assertFalse(split_cycles[0] & split_cycles[1])
            self.assertFalse(split_cycles[0] & split_cycles[2])
            self.assertFalse(split_cycles[1] & split_cycles[2])
            self.assertEqual(set.union(*split_cycles), set(range(6)))
            self.assertEqual(
                manifest["files_per_event"],
                {"1": 210, "5": 42, "7": 30, "10": 21, "21": 10},
            )
            with np.load(output / "chunks.npz") as arrays:
                for split in library.SPLITS:
                    chunks = arrays["{}_chunks".format(split)]
                    self.assertEqual(chunks.shape[1], 140)
                    self.assertEqual(len(np.unique(chunks)), chunks.size)

            repeated = root / "manifest_repeated"
            library.prepare_manifest(
                Namespace(
                    muplus_bank=str(plus),
                    muminus_bank=str(minus),
                    outdir=str(repeated),
                    force=False,
                )
            )
            repeated_manifest = json.loads(
                (repeated / "manifest.json").read_text()
            )
            self.assertEqual(
                manifest["chunk_arrays_sha256"],
                repeated_manifest["chunk_arrays_sha256"],
            )

    def test_angles_are_nested_and_rotation_preserves_invariants(self):
        cycles = np.asarray([2, 8], dtype=np.int64)
        local = np.asarray([3, 4], dtype=np.int32)
        angles = library.mother_angles(cycles, local)
        self.assertTrue(
            np.array_equal(angles[:, :5], library.mother_angles(cycles, local)[:, :5])
        )

        raw = {
            "px": np.asarray([1.0, 0.0], dtype=np.float32),
            "py": np.asarray([0.0, 2.0], dtype=np.float32),
            "pz": np.asarray([3.0, 4.0], dtype=np.float32),
            "E": np.asarray([5.0, 6.0], dtype=np.float32),
            "t": np.asarray([7.0, 8.0], dtype=np.float32),
            "vx": np.asarray([9.0, 0.0], dtype=np.float32),
            "vy": np.asarray([0.0, 10.0], dtype=np.float32),
            "vz": np.asarray([11.0, 12.0], dtype=np.float32),
            "pdg": np.asarray([11, 22], dtype=np.int32),
        }
        rotated = library.rotate_chunk(
            raw, np.asarray([0, 1]), angles, reuse_k=5
        )
        for field in ("pz", "E", "t", "vz", "pdg"):
            self.assertTrue(np.array_equal(rotated[field][:2], raw[field]))
        base_pt = np.hypot(raw["px"], raw["py"])
        for rotation in range(5):
            start = rotation * 2
            self.assertTrue(
                np.allclose(
                    np.hypot(
                        rotated["px"][start:start + 2],
                        rotated["py"][start:start + 2],
                    ),
                    base_pt,
                )
            )

    def test_k1_is_native_and_unrotated(self):
        raw = {
            "px": np.asarray([1.0, 2.0], dtype=np.float32),
            "py": np.asarray([3.0, 4.0], dtype=np.float32),
            "pz": np.asarray([5.0, 6.0], dtype=np.float32),
            "E": np.asarray([7.0, 8.0], dtype=np.float32),
            "t": np.asarray([9.0, 10.0], dtype=np.float32),
            "vx": np.asarray([11.0, 12.0], dtype=np.float32),
            "vy": np.asarray([13.0, 14.0], dtype=np.float32),
            "vz": np.asarray([15.0, 16.0], dtype=np.float32),
            "pdg": np.asarray([11, 22], dtype=np.int32),
        }
        angles = np.asarray([[0.4], [1.7]])
        native = library.rotate_chunk(
            raw, np.asarray([0, 1]), angles, reuse_k=1
        )
        for field in library.PARTICLE_FIELDS:
            self.assertTrue(np.array_equal(native[field], raw[field]))


if __name__ == "__main__":
    unittest.main()
