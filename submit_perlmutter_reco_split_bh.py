#!/usr/bin/env python3

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("legacy", "bulk_only", "split_bh"), required=True
    )
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--events-per-job", type=int, default=50)
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--qos", default="regular")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--outdir",
        default=None,
    )
    args = parser.parse_args()
    if args.outdir is None:
        args.outdir = os.path.join(
            os.environ.get("PSCRATCH", ""),
            "mucoll/reco_n420_{}_v31".format(args.mode),
        )
    if not 1 <= args.jobs <= 40:
        parser.error("--jobs must be between 1 and 40")
    if args.events_per_job < 1:
        parser.error("--events-per-job must be positive")
    if args.qos == "debug" and args.time > "00:30:00":
        parser.error("debug jobs cannot exceed 00:30:00")
    return args


def require_text(path, text):
    if text not in path.read_text():
        raise SystemExit("{} does not contain {!r}".format(path, text))


def main():
    args = arguments()
    repo = Path(__file__).resolve().parent
    bench = Path(os.environ.get("MUCOLL_BENCHMARKS_PATH", repo.parent / "mucoll-benchmarks"))
    maia = bench / "configs/MAIAConfig"
    source = maia / "MAIAConfig"

    require_text(source / "digi_args.py", "--OverlayBHMuonsSeparately")
    require_text(
        source / "ParticleFlow/pandora.py",
        'RelTrackCollections = ["MergedTrackerHitsRelations"]',
    )
    if subprocess.check_output(
        ["git", "-C", str(maia), "status", "--porcelain"],
        universal_newlines=True,
    ).strip():
        raise SystemExit("MAIAConfig must be clean before submission")
    maia_commit = subprocess.check_output(
        ["git", "-C", str(maia), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()

    output = Path(args.outdir).resolve()
    scratch = Path(os.environ.get("PSCRATCH", "")).resolve()
    try:
        output.relative_to(scratch)
    except ValueError:
        raise SystemExit("output must be under PSCRATCH: {}".format(output))

    rows = []
    skipped = 0
    for job_id in range(args.jobs):
        directory = output / "job_{}".format(job_id)
        reco = directory / "reco_output_{}.edm4hep.root".format(job_id)
        if (directory / "complete").is_file() and reco.is_file() and reco.stat().st_size > 0 and not args.force:
            skipped += 1
            continue
        rows.append((job_id, args.events_per_job, directory))

    if not rows:
        print("All requested outputs are complete; nothing to submit.")
        return

    logs = repo / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = logs / "reco_n420_{}_{}.tsv".format(args.mode, stamp)
    with manifest.open("w") as handle:
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")

    command = [
        "sbatch",
        "--parsable",
        "--qos={}".format(args.qos),
        "--time={}".format(args.time),
        "--ntasks={}".format(len(rows)),
        "--job-name=reco420_{}".format(args.mode),
        "--output=logs/reco420_{}_%j.out".format(args.mode),
        "--error=logs/reco420_{}_%j.err".format(args.mode),
        "--export=ALL,EXPECTED_MAIA_COMMIT={},RECO_BIB_MODE={}".format(
            maia_commit, args.mode
        ),
        str(repo / "submit_perlmutter_reco_split_bh.slurm"),
        str(manifest),
    ]
    print("manifest: {}".format(manifest))
    print("outputs: {}".format(output))
    print("tasks: {} ({} complete tasks skipped)".format(len(rows), skipped))
    print("events: {} total ({} per task)".format(
        args.jobs * args.events_per_job, args.events_per_job
    ))
    if args.mode == "legacy":
        print("N=420 legacy: 10 inclusive norm42 files per polarity from 6654 common valid cycles")
    elif args.mode == "bulk_only":
        print("N=420 bulk only: 10 muon-removed norm42 files per polarity")
    else:
        print("N=420 split BH: 10 bulk norm42 + 10 grouped BH files per polarity")
    print("MAIAConfig: {}".format(maia_commit))
    print(" ".join(command))
    if args.dry_run:
        return
    result = subprocess.run(
        command,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("submitted job {}".format(result.stdout.strip().split(";", 1)[0]))


if __name__ == "__main__":
    main()
