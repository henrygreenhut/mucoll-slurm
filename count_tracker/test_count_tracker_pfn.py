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
    def test_track_row_kinematics_and_charge(self):
        row = feat.track_row(0.5, 0.0003, 1.0, 0.1, 2.0)
        self.assertAlmostEqual(row[0], 0.5, places=6)          # pt = 0.00015/|omega|
        self.assertAlmostEqual(row[1], np.arcsinh(1.0), places=6)  # eta
        self.assertEqual(row[2], 0.5)                          # phi
        self.assertEqual((row[3], row[4]), (0.1, 2.0))         # d0, z0
        self.assertEqual(row[5], 1.0)                          # charge = sign(omega)
        self.assertEqual(feat.track_row(0.1, -0.0006, 0.0, 0.0, 0.0)[5], -1.0)
        self.assertIsNone(feat.track_row(np.nan, 0.0003, 1.0, 0.1, 2.0))

    def test_pfn_features_transform_and_padding(self):
        raw = np.array([[[0.5, 0.88, 0.5, 0.1, 2.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        out = feat.pfn_features(raw)
        self.assertEqual(out.shape, (1, 2, 7))
        np.testing.assert_allclose(out[0, 0], [np.log(0.5), 0.88, np.sin(0.5),
                                               np.cos(0.5), 0.1, 2.0, 1.0], rtol=1e-5)
        np.testing.assert_array_equal(out[0, 1], np.zeros(7))  # padding stays zero


class StoreTests(unittest.TestCase):
    def test_choose_state_prefers_atip_then_begin(self):
        loc = np.array([3, 1, 2, 1])
        self.assertEqual(store.choose_state(0, 2, loc), 1)   # AtIP at index 1
        self.assertEqual(store.choose_state(0, 1, loc), 0)   # no AtIP -> begin
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
                            "AllTracks.trackStates_end": B([[2, 4]])}),
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
        np.testing.assert_allclose(rows[0], [0.5, np.arcsinh(1.0), 0.5, 0.1, 2.0, 1.0], rtol=1e-5)
        np.testing.assert_allclose(rows[1], [0.25, 0.0, 0.7, 0.2, 3.0, -1.0], rtol=1e-5)


class TrainDataTests(unittest.TestCase):
    def _write_store(self, prefix, n_events, max_tracks):
        tracks = np.zeros((n_events, max_tracks, 6), np.float32)
        tracks[:, 0, 0] = 0.5  # one valid track per event
        np.savez(prefix.with_suffix(".npz"), tracks=tracks,
                 n_tracks=np.ones(n_events, np.int64))
        prefix.with_suffix(".json").write_text(json.dumps(
            {"features": list(feat.RAW_FEATURES), "n_events": n_events}))

    def test_combine_labels_and_mismatch(self):
        a = np.zeros((3, 2, 6), np.float32); a[:, 0, 0] = 0.5
        b = np.zeros((3, 2, 6), np.float32); b[:, 0, 0] = 0.5
        x, y = train.combine(a, b, width=2)
        self.assertEqual(x.shape, (6, 2, 7))
        np.testing.assert_array_equal(y, [0, 0, 0, 1, 1, 1])
        with self.assertRaises(SystemExit):
            train.combine(a, b[:2], width=2)

    def test_load_store_and_pad_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "norm42_SIM_test"
            self._write_store(prefix, n_events=4, max_tracks=3)
            tracks, manifest = train.load_store(tmp, "norm42", "SIM", "test")
            self.assertEqual(tracks.shape, (4, 3, 6))
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


if __name__ == "__main__":
    unittest.main()
