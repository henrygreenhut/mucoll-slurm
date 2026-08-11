import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

import reco_cycle_k_library as library
import reco_cycle_k_perlmutter as perlmutter
from gen_mother_make_fixed_reuse_store import cycle_angles as gen_cycle_angles
from submit_reco_cycle_k_unconed import cycle_ids


class CycleKLibraryTest(unittest.TestCase):
    def make_bank(self, path, polarity):
        cycles = np.asarray([10, 20, 30, 40], dtype=np.int64)
        cycle_offsets = np.asarray([0, 2, 5, 7, 10], dtype=np.int64)
        mother_cycles = np.repeat(cycles, np.diff(cycle_offsets))
        local_ids = np.concatenate(
            [np.arange(count, dtype=np.int32) for count in np.diff(cycle_offsets)]
        )
        mother_offsets = np.arange(len(mother_cycles) + 1, dtype=np.int64)
        with h5py.File(path, "w") as handle:
            handle.attrs["schema"] = "split-mother-gen-v1"
            handle.attrs["polarity"] = polarity
            handle.create_dataset("cycle_ids", data=cycles)
            handle.create_dataset("cycle_offsets", data=cycle_offsets)
            handle.create_dataset("mother_cycle_ids", data=mother_cycles)
            handle.create_dataset("mother_local_ids", data=local_ids)
            handle.create_dataset("mother_offsets", data=mother_offsets)
            particles = handle.create_group("particles")
            values = np.arange(len(mother_cycles), dtype=np.float32)
            for field in library.FIELDS:
                particles.create_dataset(
                    field,
                    data=(
                        np.full(len(values), 22, dtype=np.int32)
                        if field == "pdg" else values
                    ),
                )

    def test_manifest_preserves_declared_source_cycles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plus = root / "plus.h5"
            minus = root / "minus.h5"
            source = root / "source.json"
            output = root / "manifest.json"
            self.make_bank(plus, "MUPLUS")
            self.make_bank(minus, "MUMINUS")
            source.write_text(json.dumps({
                "split_seed": 12345,
                "split_fractions": {"train": 0.5, "val": 0.25, "test": 0.25},
                "excluded_cycles": [6291],
                "n_paired_cycles": 4,
                "splits": {
                    "train": [10, 20],
                    "val": [30],
                    "test": [40],
                },
            }))
            library.prepare(Namespace(
                source_pool_manifest=str(source),
                muplus_bank=str(plus),
                muminus_bank=str(minus),
                output=str(output),
            ))
            manifest = json.loads(output.read_text())
            self.assertEqual(manifest["splits"]["train"]["cycles"], [10, 20])
            self.assertEqual(
                manifest["files_per_pseudocrossing_per_polarity"],
                {"7": 60, "21": 20},
            )
            self.assertEqual(manifest["construction"],
                             "one synthetic GEN and SIM file per original FLUKA source cycle")

    def test_frozen_source_split_matches_oscar_baseline(self):
        source = library.frozen_source_split()
        self.assertEqual(source["n_paired_cycles"], 6654)
        self.assertEqual(
            [len(source["splits"][split]) for split in library.SPLITS],
            [3326, 1664, 1664],
        )
        self.assertEqual(
            source["splits"]["train"][:10],
            [6498, 4712, 798, 1281, 6364, 446, 5265, 4984, 3638, 5329],
        )
        self.assertEqual(source["split_sha256"], library.SOURCE_SPLIT_SHA256)
        self.assertNotIn(6291, source["cycles"])

    def test_perlmutter_work_items_cover_every_cycle_and_k(self):
        source = library.frozen_source_split()
        manifest = {
            "splits": {
                split: {"cycles": source["splits"][split]}
                for split in library.SPLITS
            }
        }
        base = Path("/tmp/cycle-k")
        items = list(perlmutter.work_items(base, manifest))
        self.assertEqual(len(items), 6654 * 2 * 2)
        first = items[0]
        self.assertEqual(first[:4], (7, "train", "MUPLUS", 6498))
        self.assertEqual(items[1][:4], (21, "train", "MUPLUS", 6498))
        self.assertEqual(
            first[4],
            base / "GEN/k7/train/MUPLUS/bib_gen_cycle_006498.edm4hep.root",
        )

    def test_nested_coherent_rotations(self):
        particles = {
            "px": np.asarray([1.0, 0.0, 3.0], dtype=np.float32),
            "py": np.asarray([0.0, 2.0, 4.0], dtype=np.float32),
            "pz": np.asarray([5.0, 6.0, 7.0], dtype=np.float32),
            "E": np.asarray([8.0, 9.0, 10.0], dtype=np.float32),
            "t": np.asarray([11.0, 12.0, 13.0], dtype=np.float32),
            "vx": np.asarray([14.0, 0.0, 16.0], dtype=np.float32),
            "vy": np.asarray([0.0, 15.0, 17.0], dtype=np.float32),
            "vz": np.asarray([18.0, 19.0, 20.0], dtype=np.float32),
            "pdg": np.asarray([11, 11, 22], dtype=np.int32),
        }
        owners = np.asarray([0, 0, 1], dtype=np.int64)
        angles = library.cycle_angles(10, 2)
        self.assertTrue(np.array_equal(
            angles[:, :21], gen_cycle_angles(2, 10, 21, library.ROTATION_SEED)
        ))
        k7 = library.rotate(particles, owners, angles, 7)
        k21 = library.rotate(particles, owners, angles, 21)
        for field in library.FIELDS:
            self.assertTrue(np.array_equal(k7[field], k21[field][:len(k7[field])]))
        base_pt = np.hypot(particles["px"], particles["py"])
        for rotation in range(7):
            start = rotation * len(base_pt)
            self.assertTrue(np.allclose(
                np.hypot(k7["px"][start:start + 3], k7["py"][start:start + 3]),
                base_pt,
            ))

    def test_cycle_sim_filename_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for cycle in (4, 18):
                (root / "bib_sim_cycle_{:06d}.edm4hep.root".format(cycle)).touch()
            self.assertEqual(cycle_ids(root), {4, 18})

    def test_completion_marker_is_written_after_empty_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = root / "plus.h5"
            manifest = root / "manifest.json"
            output = root / "output"
            marker = root / "complete.json"
            self.make_bank(bank, "MUPLUS")
            index = library.bank_index(bank)
            manifest.write_text(json.dumps({
                "schema": "reco-cycle-k-v1",
                "source_identity_sha256": index["identity_sha256"],
                "splits": {"train": {"cycles": [10]}},
            }))
            with mock.patch.object(
                library, "particle_properties", return_value={}
            ):
                library.write_gen(Namespace(
                    bank=str(bank),
                    manifest=str(manifest),
                    benchmarks_dir=str(root),
                    split="train",
                    reuse_k=7,
                    output_dir=str(output),
                    shard_index=1,
                    num_shards=2,
                    max_cycles=0,
                    validate=True,
                    completion_marker=str(marker),
                ))
            payload = json.loads(marker.read_text())
            self.assertEqual(payload["assigned_cycles"], 0)
            self.assertEqual(payload["reuse_k"], 7)


if __name__ == "__main__":
    unittest.main()
