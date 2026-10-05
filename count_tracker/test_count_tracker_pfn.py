#!/usr/bin/env python3
"""Local tests for the track-feature extraction and PFN data path (no TF/uproot)."""

import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

import count_tracker_track_features as feat
import count_tracker_features as store
import count_tracker_train as train


class FeatureTests(unittest.TestCase):
    def test_track_row_physics_and_curvature(self):
        row = feat.track_row(0.5, 0.0003, 1.0, 0.1, 2.0)
        self.assertAlmostEqual(row[0], 5.0, places=6)          # pt = 0.0015/|omega|
        self.assertAlmostEqual(row[1], np.arcsinh(1.0), places=6)  # eta
        self.assertEqual(row[2], 0.5)                          # phi
        self.assertEqual((row[3], row[4]), (0.1, 2.0))         # d0, z0
        self.assertEqual(row[5], 0.0003)                       # original signed omega
        self.assertEqual(feat.track_row(0.1, -0.0006, 0.0, 0.0, 0.0)[5], -0.0006)
        with self.assertRaises(ValueError):
            feat.track_row(np.nan, 0.0003, 1.0, 0.1, 2.0)
        with self.assertRaises(ValueError):
            feat.track_row(0.1, 0.0, 1.0, 0.1, 2.0)

    def test_pfn_features_transform_and_padding(self):
        raw = np.array([[[0.5, 0.88, 0.5, 0.1, 2.0, 0.003],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        out = feat.pfn_features(raw, np.array([1]))
        self.assertEqual(out.shape, (1, 2, 6))
        np.testing.assert_allclose(out[0, 0], [np.log(0.5), 0.88, np.sin(0.5),
                                               np.cos(0.5), 0.1, 2.0], rtol=1e-5)
        np.testing.assert_array_equal(out[0, 1], np.zeros(6))  # padding stays zero
        with self.assertRaises(ValueError):
            feat.pfn_features(raw, np.array([2]))  # zero-pT row cannot become a valid track


class StoreTests(unittest.TestCase):
    def test_choose_state_requires_atip(self):
        loc = np.array([3, 1, 2, 1])
        self.assertEqual(store.choose_state(0, 2, loc), 1)   # AtIP at index 1
        self.assertIsNone(store.choose_state(0, 1, loc))     # no AtIP
        self.assertIsNone(store.choose_state(5, 6, loc))     # out of range

    def test_pack_store_pads_and_counts(self):
        per_event = [np.ones((2, 6), np.float32), np.zeros((0, 6), np.float32),
                     np.full((1, 6), 7, np.float32)]
        tracks, counts = store.pack_store(per_event)
        self.assertEqual(tracks.shape, (3, 2, 6))
        np.testing.assert_array_equal(counts, [2, 0, 1])
        np.testing.assert_array_equal(tracks[1], np.zeros((2, 6)))
        np.testing.assert_array_equal(tracks[2, 0], np.full(6, 7))

    def test_read_tracks_from_mock_subset_layout(self):
        class B:
            def __init__(self, data): self.data = data
            def array(self): return self.data

        class G(dict):
            pass

        m = {
            "SiTracks_objIdx": G({"SiTracks_objIdx.index": B([[0, 1]])}),
            "AllTracks": G({"AllTracks.trackStates_begin": B([[0, 2]]),
                            "AllTracks.trackStates_end": B([[2, 4]]),
                            "AllTracks.chi2": B([[6.0, 8.0]]),
                            "AllTracks.ndf": B([[3, 4]])}),
            "_AllTracks_trackStates": G({
                "_AllTracks_trackStates.location": B([[1, 0, 1, 0]]),
                "_AllTracks_trackStates.phi": B([[0.5, 9, 0.7, 9]]),
                "_AllTracks_trackStates.omega": B([[0.0003, 9, -0.0006, 9]]),
                "_AllTracks_trackStates.tanLambda": B([[1.0, 9, 0.0, 9]]),
                "_AllTracks_trackStates.D0": B([[0.1, 9, 0.2, 9]]),
                "_AllTracks_trackStates.Z0": B([[2.0, 9, 3.0, 9]]),
            }),
        }
        events = types.SimpleNamespace(num_entries=1, __getitem__=lambda self, k: m[k])
        # SimpleNamespace can't do item access; use a small object instead.
        class Events:
            num_entries = 1
            def __getitem__(self, k): return m[k]
        per_event = store.read_tracks(Events())
        self.assertEqual(len(per_event), 1)
        rows = per_event[0]
        self.assertEqual(rows.shape, (2, 6))
        np.testing.assert_allclose(rows[0], [5.0, np.arcsinh(1.0), 0.5, 0.1, 2.0, 0.0003], rtol=1e-5)
        np.testing.assert_allclose(rows[1], [2.5, 0.0, 0.7, 0.2, 3.0, -0.0006], rtol=1e-5)

        rows, chi2_ndf = store.read_track_data(Events())
        np.testing.assert_allclose(chi2_ndf[0], [2.0, 2.0])


class TrainDataTests(unittest.TestCase):
    def test_fixed_recipes_use_adam_default_without_schedule(self):
        for name, dropout in (("fixed", 0.0), ("fixed_dropout", 0.1)):
            config = train.recipe_config(name, steps_per_epoch=33)
            self.assertEqual(config["learning_rate"], 1e-3)
            self.assertEqual(config["warmup_steps"], 0)
            self.assertEqual(config["decay_steps"], 0)
            self.assertEqual(config["f_dropout"], dropout)

    def _write_store(self, prefix, n_events, max_tracks):
        tracks = np.zeros((n_events, max_tracks, 6), np.float32)
        tracks[:, 0, 0] = 0.5  # one valid track per event
        np.savez(prefix.with_suffix(".npz"), tracks=tracks,
                 n_tracks=np.ones(n_events, np.int64))
        prefix.with_suffix(".json").write_text(json.dumps(
            {"schema_version": 2, "construction": "norm42", "sample": "SIM",
             "split": "test", "track_collection": "SiTracks", "track_state": "AtIP",
             "features": list(feat.RAW_FEATURES), "n_events": n_events,
             "total_tracks": n_events}))

    def test_combine_labels_and_mismatch(self):
        a = np.zeros((3, 2, 6), np.float32); a[:, 0, 0] = 0.5
        b = np.zeros((3, 2, 6), np.float32); b[:, 0, 0] = 0.5
        counts = np.ones(3, np.int64)
        x, y = train.combine(a, counts, b, counts, width=2)
        self.assertEqual(x.shape, (6, 2, 6))
        np.testing.assert_array_equal(y, [0, 0, 0, 1, 1, 1])
        with self.assertRaises(SystemExit):
            train.combine(a, counts, b[:2], counts[:2], width=2)

    def test_single_track_examples_weight_events_equally(self):
        a = np.zeros((3, 2, 6), np.float32)
        b = np.zeros((3, 2, 6), np.float32)
        a[:, :, 0] = b[:, :, 0] = 0.5
        counts_a = np.array([2, 0, 1])
        counts_b = np.array([1, 1, 0])
        x, y, weights, groups = train.single_track_examples(
            a, counts_a, b, counts_b)
        self.assertEqual(x.shape, (5, 1, 6))
        np.testing.assert_array_equal(y, [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(groups, [0, 0, 2, 3, 4])
        self.assertAlmostEqual(sum(weights[groups == 0]), sum(weights[groups == 2]))
        self.assertAlmostEqual(sum(weights[groups == 3]), sum(weights[groups == 4]))
        self.assertAlmostEqual(sum(weights[y == 0]), sum(weights[y == 1]))

    def test_label_permutation_keeps_tracks_grouped(self):
        labels = np.array([0, 0, 0, 1, 1, 1], np.int32)
        groups = np.array([0, 0, 1, 2, 2, 3], np.int64)
        shuffled = train.grouped_permuted_labels(labels, groups, "train")
        self.assertEqual(shuffled[0], shuffled[1])
        self.assertEqual(shuffled[3], shuffled[4])

    def test_load_store_and_pad_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "norm42_SIM_test"
            self._write_store(prefix, n_events=4, max_tracks=3)
            tracks, counts, manifest = train.load_store(tmp, "norm42", "SIM", "test")
            self.assertEqual(tracks.shape, (4, 3, 6))
            np.testing.assert_array_equal(counts, np.ones(4))
            self.assertEqual(tuple(manifest["features"]), feat.RAW_FEATURES)
            wide = train.pad_width(tracks, 5)
            self.assertEqual(wide.shape, (4, 5, 6))
            np.testing.assert_array_equal(wide[:, 3:], 0)

    def test_permuted_labels_are_deterministic(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        a = train.permuted_labels(y, "train")
        b = train.permuted_labels(y, "train")
        np.testing.assert_array_equal(a, b)
        self.assertEqual(sorted(a), sorted(y))

    def test_sim_count_pair_must_have_same_events_and_conditions(self):
        a = {"events": [{"event_id": "e0"}, {"event_id": "e1"}],
             "conditions_manifest_sha256": "same"}
        b = {"events": [{"event_id": "e0"}, {"event_id": "e1"}],
             "conditions_manifest_sha256": "same"}
        train.validate_sim_count_pair(a, b, "test")
        with self.assertRaises(SystemExit):
            train.validate_sim_count_pair(a, {**b, "events": list(reversed(b["events"]))}, "test")
        with self.assertRaises(SystemExit):
            train.validate_sim_count_pair(a, {**b, "conditions_manifest_sha256": "other"}, "test")


if __name__ == "__main__":
    unittest.main()
