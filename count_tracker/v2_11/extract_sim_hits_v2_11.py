#!/usr/bin/env python3
"""Extract one prepared SIM BIB event without using podio.

The source ROOT files were written by a newer podio release that v2.11 cannot
read.  Uproot reads the six numerical SimTrackerHit branches and writes the
selected hits as NumPy arrays for the v2.11 BIB-only event writer.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from count_tracker_conditions import (  # noqa: E402
    COLLECTIONS,
    POLARITIES,
    TRAINING_RAW_TIME_SELECTION,
    decode_cell_ids,
    in_time_mask,
    sensor_counts,
    training_raw_time_mask,
)
from count_tracker_input import load_event  # noqa: E402


FIELDS = ("eDep", "position.x", "position.y", "position.z", "time", "cellID")


def read_entry(path, entry, collection):
    import uproot

    with uproot.open(path) as root:
        events = root["events"]
        if entry < 0 or entry >= events.num_entries:
            raise ValueError(f"Entry {entry} does not exist in {path}")
        values = []
        for field in FIELDS:
            branch = events[collection][f"{collection}.{field}"].array(
                entry_start=entry, entry_stop=entry + 1, library="ak"
            )
            values.append(np.asarray(branch[0]))
    if len({len(value) for value in values}) != 1:
        raise ValueError(f"Tracker-hit branch lengths differ in {path}/{collection}")
    return values


def selected_rows(values, system, hit_selection):
    edep, x, y, z, time, cell_ids = map(np.asarray, values)
    if not all(np.all(np.isfinite(value)) for value in (edep, x, y, z, time)):
        raise ValueError("SIM tracker hits contain nonfinite values")
    if np.any(edep < 0):
        raise ValueError("SIM tracker hits contain negative energy deposition")
    keep = np.ones(len(edep), dtype=bool)
    if hit_selection == "flight-corrected":
        keep = in_time_mask(time, x, y, z)
    elif hit_selection == TRAINING_RAW_TIME_SELECTION:
        keep = training_raw_time_mask(time)
    elif hit_selection != "all-stored":
        raise ValueError(f"Unknown hit selection: {hit_selection}")
    labels = decode_cell_ids(cell_ids, system)
    return np.column_stack((edep, x, y, z, time, labels, cell_ids))[keep].astype(np.float64)


def extract(args):
    event, expected = load_event(
        args.conditions, args.split, args.event_id, args.construction
    )
    if "sim_arrays" in event:
        raise ValueError("This bridge is for source-file SIM constructions")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    hit_selection = event.get("_hit_selection", "all-stored")
    totals = {short: [] for short in COLLECTIONS}
    for polarity in POLARITIES:
        for source in event["sources"][polarity]:
            for short, (system, collection) in COLLECTIONS.items():
                rows = selected_rows(
                    read_entry(source["path"], source["entry"], collection),
                    system,
                    hit_selection,
                )
                if len(rows):
                    totals[short].append(rows)

    with tempfile.TemporaryDirectory(prefix=".v2_sim_extract_", dir=output.parent) as work:
        work = Path(work)
        for short, (system, collection) in COLLECTIONS.items():
            rows = np.concatenate(totals[short]) if totals[short] else np.empty((0, 11))
            actual = sensor_counts(rows[:, 5:10].astype(np.int64))
            if actual != expected[short]:
                difference = Counter(actual)
                difference.subtract(expected[short])
                raise ValueError(
                    f"{short} extracted occupancy differs from conditions: {difference}"
                )
            path = work / f"{collection}_SimTrackerHit_conditional_reco9_0.npy"
            np.save(path, rows, allow_pickle=False)
            print(f"{short}: extracted {len(rows):,} hits", flush=True)
        work.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--construction", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
