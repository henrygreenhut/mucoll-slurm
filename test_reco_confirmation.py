#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate_reco_libtest_confirmation as evaluator
import submit_reco_libtest_confirmation as submitter


class ConfirmationManifestTests(unittest.TestCase):
    def test_manifest_uses_fresh_paired_jobs_and_held_out_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pools = root / "pools"
            output = root / "output"
            for library in ("norm1", "norm42"):
                for polarity in ("MUPLUS", "MUMINUS"):
                    path = pools / library / "test" / polarity
                    path.mkdir(parents=True)
                    (path / "cycle_1.root").touch()

            rows, skipped = submitter.manifest_rows(
                pools, output, events_per_class=75
            )

        self.assertEqual(skipped, 0)
        self.assertEqual(len(rows), 6)
        for sample in submitter.SAMPLES:
            sample_rows = [row for row in rows if row[0] == sample]
            self.assertEqual(
                [int(row[3]) for row in sample_rows], [300000, 300001]
            )
            self.assertEqual(
                [int(row[4]) for row in sample_rows], [50, 25]
            )
            self.assertTrue(all(row[1] == "confirmation" for row in sample_rows))
            self.assertTrue(all("/test/" in row[7] for row in sample_rows))
            self.assertTrue(all("/test/" in row[8] for row in sample_rows))
        reused = [row for row in rows if row[0] == "R"]
        unique = [row for row in rows if row[0] == "U"]
        self.assertTrue(all(int(row[9]) == 10 for row in reused))
        self.assertTrue(all(int(row[9]) == 420 for row in unique))

    def test_followups_require_frozen_checkpoint_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(SystemExit, "missing checkpoint files"):
                submitter.require_checkpoints(root)
            for label in submitter.CHECKPOINTS:
                checkpoint = root / "reco_pfn_results" / label
                checkpoint.mkdir(parents=True)
                (checkpoint / "summary.json").write_text("{}")
                (checkpoint / "best.weights.h5").write_bytes(b"weights")
            submitter.require_checkpoints(root)


class ConfirmationEvaluationTests(unittest.TestCase):
    def test_job_key(self):
        self.assertEqual(
            evaluator.job_key(
                "/scratch/study/confirmation/job_300019/reco_output.root"
            ),
            300019,
        )
        with self.assertRaisesRegex(ValueError, "cannot extract"):
            evaluator.job_key("/scratch/reco_output.root")

    def test_paired_job_bootstrap_is_deterministic(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        metadata = [
            (0, "/x/job_1/output.root", 0),
            (0, "/x/job_1/output.root", 1),
            (0, "/x/job_2/output.root", 0),
            (0, "/x/job_2/output.root", 1),
            (1, "/y/job_1/output.root", 0),
            (1, "/y/job_1/output.root", 1),
            (1, "/y/job_2/output.root", 0),
            (1, "/y/job_2/output.root", 1),
        ]
        first, jobs = evaluator.paired_job_bootstrap(
            labels, scores, metadata, repetitions=20, seed=7
        )
        second, _ = evaluator.paired_job_bootstrap(
            labels, scores, metadata, repetitions=20, seed=7
        )
        self.assertEqual(jobs, 2)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, 1.0)

    def test_crossentropy_and_score_summary(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertLess(evaluator.binary_crossentropy(labels, scores), np.log(2))
        summary = evaluator.score_summary(labels, scores, "R")
        self.assertEqual(summary["U"]["events"], 2)
        self.assertEqual(summary["R"]["events"], 2)


if __name__ == "__main__":
    unittest.main()
