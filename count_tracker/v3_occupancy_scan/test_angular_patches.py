#!/usr/bin/env python3
"""Unit tests for full-BX angular track patches."""

import unittest
import json
from pathlib import Path
import tempfile
import types
from unittest import mock

import numpy as np

import build_angular_patches as angular
from build_angular_patches import partition_tracks, sector_indices_and_local_phi


class AngularPatchTests(unittest.TestCase):
    def test_sector_boundaries_wrap_and_cover(self):
        phi = np.asarray([-np.pi, -np.pi + 0.01, 0.0, np.pi - 0.01, np.pi, 3 * np.pi])
        sectors, local = sector_indices_and_local_phi(phi, 4)
        np.testing.assert_array_equal(sectors, [0, 0, 2, 3, 0, 0])
        self.assertTrue(np.all(local >= -np.pi / 4))
        self.assertTrue(np.all(local < np.pi / 4))

    def test_partition_preserves_each_track_once_and_localizes_phi(self):
        tracks = np.zeros((13, 6), dtype=np.float32)
        tracks[:, 2] = np.linspace(-np.pi, np.pi, len(tracks), endpoint=False)
        tracks[:, 0] = np.arange(len(tracks)) + 1
        patches = partition_tracks(tracks, 6)
        self.assertEqual(len(patches), 6)
        self.assertEqual(sum(map(len, patches)), len(tracks))
        observed_ids = np.concatenate([patch[:, 0] for patch in patches])
        np.testing.assert_array_equal(np.sort(observed_ids), np.arange(len(tracks)) + 1)
        self.assertTrue(all(np.all(np.abs(patch[:, 2]) <= np.pi / 6 + 1e-6)
                            for patch in patches))

    def test_partition_preserves_reduced_chi2_column(self):
        tracks = np.zeros((3, 7), dtype=np.float32)
        tracks[:, 2] = [-2.0, 0.0, 2.0]
        tracks[:, 6] = [0.5, 1.0, 1.5]
        patches = partition_tracks(
            tracks, 4, angular.RAW_FEATURES_WITH_CHI2_NDF)
        observed = np.concatenate([patch[:, 6] for patch in patches])
        np.testing.assert_allclose(np.sort(observed), [0.5, 1.0, 1.5])

    def test_builder_keeps_parent_splits_paired_and_all_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conditions = root / "scan" / "n1666" / "conditions"
            conditions.mkdir(parents=True)
            conditions.joinpath("manifest.json").write_text(json.dumps({
                "events": [{"event_id": f"event_{index}"} for index in range(5)]
            }))
            one_track = np.asarray([[1.0, 0.2, 0.1, 0.0, 0.0, 0.001]], np.float32)
            args = types.SimpleNamespace(
                scan_root=str(root / "scan"), point="n1666", phi_sectors=4,
                split_counts=[3, 1, 1], output=str(root / "stores"))
            with mock.patch.object(angular, "read_regional_tracks",
                                   return_value=one_track):
                angular.build(args)

            prefix = root / "stores" / "norm42_fullbx_theta_phi4"
            for split, parents in (("train", 3), ("val", 1), ("test", 1)):
                manifests = [json.loads(
                    prefix.with_name(prefix.name + f"_{sample}_{split}.json").read_text())
                    for sample in angular.SAMPLES]
                self.assertEqual(manifests[0]["n_events"], parents * 5 * 4)
                self.assertEqual(manifests[0]["total_tracks"], parents * 5)
                self.assertEqual(manifests[0]["example_parent_ids"],
                                 manifests[1]["example_parent_ids"])
                self.assertEqual(manifests[0]["n_parent_events"], parents)
            summary = json.loads((root / "stores" / "angular_patch_summary.json").read_text())
            self.assertEqual(summary["total_tracks"], {"SIM": 25, "COUNT": 25})

    def test_builder_can_publish_reduced_chi2_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conditions = root / "scan" / "n1666" / "conditions"
            conditions.mkdir(parents=True)
            conditions.joinpath("manifest.json").write_text(json.dumps({
                "events": [{"event_id": f"event_{index}"} for index in range(5)]
            }))
            one_track = np.asarray(
                [[1.0, 0.2, 0.1, 0.0, 0.0, 0.001]], np.float32)
            reduced_chi2 = np.asarray([1.25], np.float32)
            args = types.SimpleNamespace(
                scan_root=str(root / "scan"), point="n1666", phi_sectors=4,
                split_counts=[3, 1, 1], output=str(root / "stores"),
                include_chi2_ndf=True)
            with mock.patch.object(
                    angular, "read_regional_track_data",
                    return_value=(one_track, reduced_chi2)):
                angular.build(args)

            prefix = root / "stores" / "norm42_fullbx_theta_phi4_chi2ndf_SIM_test"
            manifest = json.loads(prefix.with_suffix(".json").read_text())
            self.assertEqual(
                tuple(manifest["features"]), angular.RAW_FEATURES_WITH_CHI2_NDF)
            with np.load(prefix.with_suffix(".npz")) as data:
                self.assertEqual(data["tracks"].shape[-1], 7)
                valid = data["tracks"][:, :, 6][data["tracks"][:, :, 0] > 0]
                np.testing.assert_allclose(valid, 1.25)


if __name__ == "__main__":
    unittest.main()
