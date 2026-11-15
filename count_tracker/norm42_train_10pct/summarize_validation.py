#!/usr/bin/env python3
"""Collect the six per-collection validation reports into JSON and CSV."""

import argparse
import csv
import json
from pathlib import Path


COLLECTIONS = ("VBC", "VEC", "ITBC", "ITEC", "OTBC", "OTEC")


def summarize(root, output_json, output_csv):
    root = Path(root).resolve()
    reports = []
    rows = []
    for short in COLLECTIONS:
        path = root / short / "validation_report.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed report: {path}")
        report = json.loads(path.read_text())
        if report.get("status") != "complete" or report.get("collection") != short:
            raise ValueError(f"invalid validation report: {path}")
        reports.append(report)
        rows.append({
            "collection": short,
            "checkpoint": report["checkpoint_filename"],
            "requested_hits": report["sampling"]["requested_hits"],
            "generated_hits": report["sampling"]["generated_hits"],
            "unfilled_hits": report["sampling"]["unfilled_hits"],
            "acceptance_fraction": report["sampling"]["acceptance_fraction"],
            "rejection_rounds": report["sampling"]["rejection_rounds"],
            "classifier_auc": report["classifier_two_sample_test"]["roc_auc"],
            "runtime_seconds": report["runtime_seconds"],
        })
    combined = {
        "schema_version": 1,
        "kind": "norm42_10pct_heldout_validation_summary",
        "model_checkpoint": "model.pt",
        "collections": reports,
    }
    Path(output_json).write_text(json.dumps(combined, indent=2) + "\n")
    with Path(output_csv).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    summarize(
        args.validation_root, args.output_json, args.output_csv
    )


if __name__ == "__main__":
    main()
