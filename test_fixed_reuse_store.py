#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import libtest_common as lc
from gen_mother_make_fixed_reuse_store import (
    build_fixed_reuse_store,
    cycle_angles,
)


class FixedReuseStoreTests(unittest.TestCase):
    def write_source(self, path):
        raw = {
            "px": np.array([1.0, 0.0, 2.0], np.float32),
            "py": np.array([0.0, 1.0, 0.0], np.float32),
            "pz": np.array([3.0, 4.0, 5.0], np.float32),
            "E": np.array([4.0, 5.0, 6.0], np.float32),
            "t": np.array([7.0, 8.0, 9.0], np.float32),
            "vx": np.array([10.0, 0.0, 20.0], np.float32),
            "vy": np.array([0.0, 10.0, 0.0], np.float32),
            "vz": np.array([11.0, 12.0, 13.0], np.float32),
            "pdg": np.array([22, 11, 2112], np.int32),
        }
        with h5py.File(path, "w") as output:
            particles = output.create_group("particles")
            for name, values in raw.items():
                particles.create_dataset(name, data=values)
            output.create_dataset("mother_offsets", data=[0, 2, 3])
            output.create_dataset("mother_cycle_ids", data=[100, 101])
            output.create_dataset("mother_local_ids", data=[0, 0])
            output.create_dataset("cycle_ids", data=[100, 101])
            output.create_dataset("cycle_offsets", data=[0, 1, 2])

    def test_native_mother_bank_is_store_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mothers.h5"
            self.write_source(source)

            store = lc.Store(source)
            np.testing.assert_array_equal(store.cycle_ids, [100, 101])
            np.testing.assert_array_equal(store.offsets, [0, 2, 3])
            np.testing.assert_array_equal(store.raw["px"], [1.0, 0.0, 2.0])

    def test_fixed_k_expands_every_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mothers.h5"
            output = Path(directory) / "k5.h5"
            self.write_source(source)
            build_fixed_reuse_store(source, output, reuse_k=5, seed=9)

            store = lc.Store(output)
            np.testing.assert_array_equal(store.offsets, [0, 10, 15])
            np.testing.assert_array_equal(
                store.raw["pz"], [3.0, 4.0] * 5 + [5.0] * 5
            )
            np.testing.assert_allclose(
                np.hypot(store.raw["px"], store.raw["py"]),
                [1.0, 1.0] * 5 + [2.0] * 5,
                atol=1e-6,
            )

    def test_k5_angles_are_nested_inside_k7(self):
        k5 = cycle_angles(8, cycle=123, reuse_k=5, seed=1701)
        k7 = cycle_angles(8, cycle=123, reuse_k=7, seed=1701)
        np.testing.assert_array_equal(k5, k7[:, :5])

    def test_refuses_to_build_a_k1_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mothers.h5"
            output = Path(directory) / "k1.h5"
            self.write_source(source)
            with self.assertRaises(ValueError):
                build_fixed_reuse_store(source, output, reuse_k=1)


if __name__ == "__main__":
    unittest.main()
