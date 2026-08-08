#!/usr/bin/env python3

import json
import os
from pathlib import Path


SPLITS = ("train", "val", "test")


def root_count(path):
    return len(list(path.glob("*.root"))) if path.is_dir() else 0


def main():
    base = Path("/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/reco_cycle_k")
    manifest_path = base / "pools" / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("missing {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text())
    for k in (7, 21):
        print("K={}".format(k))
        for split in SPLITS:
            expected = manifest["splits"][split]["count"]
            for polarity in ("MUPLUS", "MUMINUS"):
                gen = root_count(base / "GEN" / "k{}".format(k) / split / polarity)
                sim = root_count(base / "pools" / "k{}".format(k) / split / polarity)
                print(
                    "  {:5s} {:7s} GEN {:4d}/{:4d}  SIM {:4d}/{:4d}".format(
                        split, polarity, gen, expected, sim, expected
                    )
                )

    scratch = Path("/oscar/scratch") / os.environ["USER"] / "mucoll" / "libtest"
    reco = scratch / "reco_cycle_k_n420_unconed"
    print("RECO")
    for k in (7, 21):
        for split in SPLITS:
            complete = len(list(
                (reco / "reco_cycle_n420_k{}".format(k) / split).glob("job_*/complete")
            ))
            print("  K={:<2d} {:5s} {:2d}/40 jobs".format(k, split, complete))


if __name__ == "__main__":
    main()
