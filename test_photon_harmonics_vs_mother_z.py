#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from gen_fluka_make_mother_store import (
    FLUKA_RECORD,
    convert_cycle,
    group_particles_by_mother,
)
from photon_harmonics_vs_mother_z import (
    bin_edges,
    load_mother_statistics,
    main as analyze,
    summarize,
)
from plot_photon_harmonics_vs_mother_z import main as plot


class MotherPositionTests(unittest.TestCase):
    def test_grouping_preserves_first_appearance_order(self):
        records = np.zeros(3, dtype=FLUKA_RECORD)
        records[0]["z_mu"] = 20.0
        records[2]["z_mu"] = 20.0
        records[1]["z_mu"] = 10.0

        order, counts, positions = group_particles_by_mother(records)

        np.testing.assert_array_equal(order, [0, 2, 1])
        np.testing.assert_array_equal(counts, [2, 1])
        np.testing.assert_array_equal(positions["z_mu"], [20.0, 10.0])

    def test_decay_positions_use_output_units_and_z_convention(self):
        records = np.zeros(1, dtype=FLUKA_RECORD)
        records["fid"] = 1
        records["e_kin"] = 1.0
        records["cx"] = 1.0
        records["x_mu"] = 1.0
        records["y_mu"] = 2.0
        records["z_mu"] = 3.0

        _, positions, _, skipped = convert_cycle(
            records, {1: 22}, {22: ("photon", 0.0)}, invert_z=True
        )

        self.assertEqual(skipped, 0)
        np.testing.assert_array_equal(positions, [[10.0, 20.0, -30.0]])


class PhotonHarmonicProfileTests(unittest.TestCase):
    def write_bank(self, path):
        phi = np.array(
            [
                0.0,
                np.pi,
                0.0,
                np.pi,
                np.pi / 2,
                -np.pi / 2,
                np.pi / 4,
                -3 * np.pi / 4,
            ]
        )
        with h5py.File(path, "w") as output:
            particles = output.create_group("particles")
            particles.create_dataset("pdg", data=np.full(8, 22, dtype=np.int32))
            particles.create_dataset("px", data=np.cos(phi))
            particles.create_dataset("py", data=np.sin(phi))
            output.create_dataset("mother_offsets", data=[0, 2, 4, 6, 8])
            output.create_dataset("mother_cycle_ids", data=[10, 11, 10, 11])
            output.create_dataset(
                "mother_decay_positions",
                data=[[0, 0, -500], [0, 0, -500], [0, 0, 500], [0, 0, 500]],
            )

    def test_profile_uses_photons_from_complete_mothers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.h5"
            self.write_bank(path)
            data = load_mother_statistics(path)
            edges = bin_edges([data], width=1000.0)
            result = summarize(
                data,
                edges,
                bootstrap_samples=20,
                rng=np.random.default_rng(4),
            )

        np.testing.assert_array_equal(result["mother_counts"], [2, 2])
        np.testing.assert_array_equal(result["photon_counts"], [4, 4])
        np.testing.assert_allclose(result["c2"], [1.0, -0.5], atol=1e-12)
        np.testing.assert_allclose(result["s2"], [0.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(result["a2"], [1.0, np.sqrt(0.5)], atol=1e-12)

    def test_command_line_analysis_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bank = directory / "bank.h5"
            output = directory / "results"
            figure = output / "profile"
            self.write_bank(bank)

            with patch(
                "sys.argv",
                [
                    "photon_harmonics_vs_mother_z.py",
                    "--bank",
                    f"MUPLUS={bank}",
                    "--output-directory",
                    str(output),
                    "--bootstrap-samples",
                    "20",
                ],
            ):
                analyze()
            with patch(
                "sys.argv",
                [
                    "plot_photon_harmonics_vs_mother_z.py",
                    str(output / "photon_harmonics_vs_mother_z.npz"),
                    "--output-prefix",
                    str(figure),
                ],
            ):
                plot()

            for suffix in (".npz", ".csv", ".json"):
                self.assertTrue(
                    (output / "photon_harmonics_vs_mother_z")
                    .with_suffix(suffix)
                    .is_file()
                )
            self.assertTrue(figure.with_suffix(".pdf").is_file())
            self.assertTrue(figure.with_suffix(".png").is_file())


if __name__ == "__main__":
    unittest.main()
