#!/usr/bin/env python3
"""Local tests for the COUNT sampling driver, with paper1 primitives mocked.

These exercise the rejection-loop orchestration and the per-event/per-collection
driver without torch, a GPU, or real TabDDPM models. The vendored helpers
(``map_conditions_to_classes``, feature maps) are exercised for real.
"""

import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

import count_tracker_sample as cts


def make_sampler(short, *, num_features=5, y_lookup=None, reject_rows=()):
    """A fake CollectionSampler whose paper1 callables echo the conditions.

    ``reject_rows`` (indices into the first round's ``remaining``) are rejected
    once by the material map to force a second rejection round.
    """
    system_id, collection_name = cts.COLLECTIONS[short]
    state = {"round": 0}

    def tabddpm_sample(*, parent_dir, num_samples, seed, y_to_sample, **_kwargs):
        np.save(Path(parent_dir) / "X_num_train.npy",
                np.full((num_samples, num_features), float(seed), dtype=np.float32))
        np.save(Path(parent_dir) / "y_train.npy", np.asarray(y_to_sample, dtype=np.int64))

    def inverse_geometry_transform(hits, basis, name, order):
        # Our column_stack already is [features..., side, layer, module, sensor].
        return np.asarray(hits, dtype=np.float32)

    def apply_material_map_hybrid(coll_dict, _none, name, indices):
        hits = coll_dict[name]
        mask = np.ones(len(hits), dtype=bool)
        if state["round"] == 0:
            for i in reject_rows:
                if i < len(mask):
                    mask[i] = False
        state["round"] += 1
        return mask

    paper1 = (tabddpm_sample, inverse_geometry_transform, None, None,
              apply_material_map_hybrid)

    sampler = types.SimpleNamespace()
    sampler.short = short
    sampler.system_id = system_id
    sampler.collection_name = collection_name
    sampler.num_features = num_features
    sampler.z_lookup = None
    sampler.device = types.SimpleNamespace(type="cpu")
    sampler.sample_job_common = {"num_numerical_features": num_features}
    sampler.paper1 = paper1
    sampler.y_lookup = y_lookup if y_lookup is not None else np.zeros((0, 4), dtype=np.int64)
    return sampler


class RejectionLoopTests(unittest.TestCase):
    def _conditions(self, short, sensors):
        system_id = cts.COLLECTIONS[short][0]
        return np.array([[system_id, *s] for s in sensors], dtype=np.int64)

    def test_fills_all_conditions_and_preserves_labels(self):
        short = "VBC"
        sensors = [(0, 1, 5, 0), (0, 1, 6, 0), (0, 2, 5, 0)]
        conditions = self._conditions(short, sensors)
        sampler = make_sampler(short, y_lookup=np.array(sensors, dtype=np.int64))
        with tempfile.TemporaryDirectory() as work:
            out = cts.run_rejection_loop(sampler, conditions, seed=7, work_dir=work)
        self.assertEqual(out.shape, (3, 9))
        np.testing.assert_array_equal(out[:, 5:].astype(np.int64), conditions[:, 1:])

    def test_rejected_hits_are_resampled_until_filled(self):
        short = "VEC"
        sensors = [(-1, 0, 0, 3), (1, 0, 0, 4), (-1, 1, 0, 5), (1, 1, 0, 6)]
        conditions = self._conditions(short, sensors)
        sampler = make_sampler(short, y_lookup=np.array(sensors, dtype=np.int64),
                               reject_rows=(1, 3))
        with tempfile.TemporaryDirectory() as work:
            out = cts.run_rejection_loop(sampler, conditions, seed=1, work_dir=work)
        self.assertEqual(len(out), 4)
        np.testing.assert_array_equal(out[:, 5:].astype(np.int64), conditions[:, 1:])

    def test_empty_conditions_return_empty_without_model(self):
        sampler = make_sampler("ITBC")
        empty = np.zeros((0, 5), dtype=np.int64)
        out = cts.run_rejection_loop(sampler, empty, seed=3)
        self.assertEqual(out.shape, (0, 9))

    def test_wrong_system_is_rejected(self):
        sampler = make_sampler("VBC", y_lookup=np.array([[0, 1, 5, 0]], dtype=np.int64))
        bad = np.array([[2, 1, 5, 0]], dtype=np.int64)  # system 2 != VBC system 1
        with self.assertRaises(ValueError):
            cts.run_rejection_loop(sampler, bad, seed=0)

    def test_unfilled_after_max_rounds_raises_with_conditions(self):
        short = "VBC"
        sensors = [(0, 1, 5, 0)]
        conditions = self._conditions(short, sensors)
        sampler = make_sampler(short, y_lookup=np.array(sensors, dtype=np.int64),
                               reject_rows=(0,))
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(RuntimeError) as ctx:
                cts.run_rejection_loop(sampler, conditions, seed=0,
                                       max_rejection_rounds=1, work_dir=work)
        self.assertTrue(hasattr(ctx.exception, "unfilled_conditions"))

    def test_sampler_sample_method_delegates_to_loop(self):
        short = "VBC"
        sensors = [(0, 1, 5, 0), (0, 1, 6, 0)]
        conditions = self._conditions(short, sensors)
        sampler = make_sampler(short, y_lookup=np.array(sensors, dtype=np.int64))
        with tempfile.TemporaryDirectory() as work:
            out = cts.CollectionSampler.sample(sampler, conditions, 5, work_dir=work)
        np.testing.assert_array_equal(out[:, 5:].astype(np.int64), conditions[:, 1:])


class SeedTests(unittest.TestCase):
    def test_seed_is_deterministic_and_varies(self):
        a = cts.event_seed(0, "norm1", "SIM_A", "norm1_SIM_A_000000", "VBC")
        b = cts.event_seed(0, "norm1", "SIM_A", "norm1_SIM_A_000000", "VBC")
        c = cts.event_seed(0, "norm1", "SIM_A", "norm1_SIM_A_000000", "VEC")
        d = cts.event_seed(0, "norm1", "SIM_A", "norm1_SIM_A_000001", "VBC")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertTrue(0 <= a <= 0x7FFFFFFF)


class DriverTests(unittest.TestCase):
    def _make_conditions_dir(self, root, shorts, event_ids):
        report = {"manifest": {"schema_version": 2, "construction": "norm1",
                               "sampling": {"cohort": "SIM_A"}},
                  "events": []}
        for event_id in event_ids:
            report["events"].append({"event_id": event_id, "split": "test",
                                     "collections": {}})
            dest = root / "test" / event_id
            dest.mkdir(parents=True)
            for short in shorts:
                system_id = cts.COLLECTIONS[short][0]
                sensors = np.array([[system_id, 0, 1, 5, 0], [system_id, 0, 1, 6, 0]],
                                   dtype=np.int64)
                np.save(dest / f"{short}_conditions.npy", sensors)
        (root / "manifest.json").write_text(json.dumps(report, indent=2))

    def test_driver_writes_one_file_per_event_and_collection(self):
        shorts = ["VBC", "VEC"]
        event_ids = ["norm1_SIM_A_000000", "norm1_SIM_A_000001"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cond_dir = tmp / "conditions"
            self._make_conditions_dir(cond_dir, shorts, event_ids)
            out_dir = tmp / "count_samples"

            def fake_loop(sampler, conditions, seed, **kwargs):
                labels = np.asarray(conditions)[:, 1:]
                head = np.zeros((len(conditions), 5), dtype=np.float32)
                return np.column_stack([head, labels]).astype(np.float32)

            saved = (cts.load_paper1, cts.resolve_device, cts.resolve_model_dir,
                     cts.CollectionSampler, cts.run_rejection_loop)
            try:
                cts.load_paper1 = lambda root: ("paper1",)
                cts.resolve_device = lambda spec: types.SimpleNamespace(type="cpu")
                cts.resolve_model_dir = lambda model_root, short: Path(model_root) / short
                cts.CollectionSampler = lambda short, *a, **k: types.SimpleNamespace(
                    short=short, y_lookup=np.zeros((0, 4)))
                cts.run_rejection_loop = fake_loop
                args = types.SimpleNamespace(
                    conditions=str(cond_dir), split="test", event_id=None,
                    model_root=str(tmp / "models"), paper1_root=None,
                    collections=shorts, seed_base=0, max_rounds=10,
                    device="cpu", output=str(out_dir))
                cts.sample_events(args)
            finally:
                (cts.load_paper1, cts.resolve_device, cts.resolve_model_dir,
                 cts.CollectionSampler, cts.run_rejection_loop) = saved

            for event_id in event_ids:
                for short in shorts:
                    path = out_dir / "test" / event_id / f"tabddpm_{short}_samples.npy"
                    self.assertTrue(path.is_file(), path)
                    self.assertEqual(np.load(path).shape, (2, 9))
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["construction"], "norm1")
            self.assertEqual(len(manifest["events"]), 2)
            self.assertEqual(set(manifest["events"][0]["collections"]), set(shorts))

    def test_driver_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cond_dir = tmp / "conditions"
            self._make_conditions_dir(cond_dir, ["VBC"], ["norm1_SIM_A_000000"])
            out_dir = tmp / "count_samples"
            out_dir.mkdir()
            args = types.SimpleNamespace(
                conditions=str(cond_dir), split="test", event_id=None,
                model_root=str(tmp / "models"), paper1_root=None,
                collections=["VBC"], seed_base=0, max_rounds=10,
                device="cpu", output=str(out_dir))
            with self.assertRaises(FileExistsError):
                cts.sample_events(args)


if __name__ == "__main__":
    unittest.main()
