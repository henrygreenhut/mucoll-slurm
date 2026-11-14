#!/usr/bin/env python3
"""Summarize completion and final loss for the six 10%-corpus models."""

import argparse
import csv
import json
from pathlib import Path

from train_collection import COLLECTIONS, model_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    args = parser.parse_args()
    root = Path(args.model_root)
    rows = []
    for collection in COLLECTIONS:
        directory = root / model_name(collection)
        partial = directory.with_name(directory.name + ".partial")
        row = {"collection": collection, "status": "missing"}
        if directory.is_dir():
            config = directory / "run_config.json"
            loss = directory / "loss.csv"
            if config.is_file() and loss.is_file():
                with loss.open() as stream:
                    history = list(csv.DictReader(stream))
                row.update({
                    "status": json.loads(config.read_text())["status"],
                    "last_step": int(float(history[-1]["step"])),
                    "final_loss": float(history[-1]["loss"]),
                    "directory": str(directory),
                })
            else:
                row["status"] = "invalid_complete_directory"
        elif partial.is_dir():
            row.update({
                "status": "incomplete",
                "directory": str(partial),
            })
        rows.append(row)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
