import unittest

import numpy as np

from photon_provenance import PHOTON, RECORD, binned, cycle_statistics, summarize


class PhotonProvenanceTests(unittest.TestCase):
    def test_raw_record_layout(self):
        self.assertEqual(RECORD.itemsize, 96)
        self.assertEqual(RECORD.names[-3:], ("x_mu", "y_mu", "z_mu"))

    def test_harmonic_orientation(self):
        phi = np.full(4, np.pi / 4)
        values = [np.ones(4), np.cos(2 * phi), np.sin(2 * phi),
                  np.cos(2 * phi), np.sin(2 * phi)]
        data = cycle_statistics(np.arange(4), np.zeros(4, int), values, 4, 1)
        row = summarize(data)[0]
        self.assertAlmostEqual(row["momentum_c2"], 0)
        self.assertAlmostEqual(row["momentum_s2"], 1)

    def test_bin_counts_and_contributions_are_additive(self):
        photons = np.zeros(16, PHOTON)
        photons["cycle"] = np.repeat(np.arange(4), 4)
        phi = np.tile([0, np.pi / 2, np.pi, -np.pi / 2], 4)
        values = [np.ones(16), np.cos(2 * phi), np.sin(2 * phi),
                  np.cos(2 * phi), np.sin(2 * phi)]
        _, rows, _ = binned(photons, values, [phi],
                            [np.array([-np.inf, 0, np.inf])], 4)
        self.assertEqual(sum(row["photons"] for row in rows), 16)
        self.assertAlmostEqual(sum(row["momentum_c2_contribution"] for row in rows), 0)

    def test_uncertainty_uses_cycles(self):
        values = np.random.default_rng(3).normal(size=20)
        data = np.stack([np.ones(20), values, values, values, values], axis=-1)
        row = summarize(data[:, None, :])[0]
        self.assertAlmostEqual(row["momentum_c2_se"], values.std(ddof=1) / np.sqrt(20))


if __name__ == "__main__":
    unittest.main()
