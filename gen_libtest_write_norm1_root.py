#!/usr/bin/env python3
"""Reconstruct a genuine norm1 GEN library as real per-cycle EDM4hep ROOT
files (not the HDF5 store gen_libtest_reconstruct_unrotated.py produces),
so ddsim can run on them to generate real norm1 SIM data.

Background: a genuine norm1 SIM library DOES exist -- reco_libtest_prepare_
pools.py's separate required --norm1-sim/--norm42-sim arguments and
config.sh's BIB_DIR=.../bib-v3p0-fmt2-norm1/SIM confirm Perlmutter had a
real, distinct norm1 SIM/DIGI/RECO chain (that's what the original reco
experiment ran against). It just isn't reachable right now: Perlmutter is
down for maintenance, and OSCAR's copy of the BIB tree only ever received
norm42-RandomRot -- confirmed by MD5, bib/SIM/MUPLUS/bib_sim_0... is
byte-identical to bib-v3p0-fmt2-norm42-RandomRot/SIM/MUPLUS/bib_sim_0....
This script is the "keep working now" path rather than waiting ~2 weeks
for Perlmutter and rsyncing the real thing: dedupe the norm42 GEN files
back to their unique mothers (same method as gen_libtest_reconstruct_
unrotated.py -- rotation-invariant key on pdg + |p|/theta/vz/t, first-
encountered representative kept per 42-clone group), write each cycle out
as a real single-event EDM4hep GEN file, then run ddsim on those. The
result is a faithful reconstruction (the equivalent GEN-level dedup was
validated earlier against Perlmutter's real norm1 store, exact agreement
on all 7 summary statistics checked), not guaranteed byte-identical to
whatever originally produced any historical reco-level results.

Each source file is confirmed (inspect_gen_rotation.py, and directly here)
to be exactly one event with no parent/daughter genealogy (all parents_/
daughters_begin/end are 0) and uniform generatorStatus=1/simulatorStatus=0
-- flat, independent primaries, nothing to preserve beyond the per-
particle kinematics/PDG/charge/mass/time/vertex themselves.

Must run INSIDE the mucoll-sim container with /opt/setup_mucoll.sh
sourced first -- podio/edm4hep/cppyy (the EDM4hep write API) only exist
there, not in the plain uproot/awkward venv used for the HDF5-store path:

    apptainer exec /oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif \\
        bash -c "source /opt/setup_mucoll.sh && python3 \\
        gen_libtest_write_norm1_root.py \\
        --input-dir /oscar/data/mleblan6/mucoll/bib/bib-v3p0-fmt2-norm42-RandomRot/GEN/MUPLUS \\
        --output-dir /oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/GEN/MUPLUS \\
        --max-files 5"

Drop --max-files (or 0) for the full run once a small sample's output has
been sanity-checked (read back + confirmed via a smoke-test ddsim run --
both done, see gen_libtest_write_norm1_root validation notes).

Deliberately serial WITHIN one process, not multiprocessing: cppyy/ROOT's
global state has known issues with fork-based parallelism. For the full
6666-file production run, parallelize at the SLURM-job level instead
(--shard-index/--num-shards below), which sidesteps that entirely via OS-
level process isolation -- see submit_norm1_gen_write.slurm, a 64-way
sharded array job (64 = this account's normal-QOS MaxTRESPU cpu cap on
the batch partition).
"""

import argparse
import glob
import os
import sys
import time

import awkward as ak
import numpy as np
import uproot

DECIMALS = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=0,
                        help="process only the first N files (by cycle "
                             "order) -- for validating output before "
                             "committing to a full run (0 = all files); "
                             "applied AFTER sharding if --num-shards is set")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="this task's shard, 0-indexed (for SLURM array "
                             "parallelism -- see submit_norm1_gen_write.slurm)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="total shards; files are assigned by "
                             "index %% num_shards, so each shard gets an "
                             "interleaved (naturally load-balanced) subset "
                             "rather than a contiguous block")
    parser.add_argument("--expected-group-size", type=int, default=42,
                        help="warn if a file's median group size deviates "
                             "from this (default 42, the confirmed value)")
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    return args


def sorted_by_cycle(paths):
    """Sort paths by cycle id (the varying integer token in the basename).
    Inlined rather than imported from libtest_common: that module isn't
    on this container's PYTHONPATH (it's outside the key4hep stack), and
    this is a small enough piece of logic not to be worth wiring in
    cross-environment."""
    import re
    tokens = [re.findall(r"\d+", os.path.basename(p)) for p in paths]
    n_tok = min(len(t) for t in tokens)
    if n_tok == 0:
        raise ValueError("filenames contain no integer tokens")
    best_pos, best_distinct = 1, 0
    for pos in range(1, n_tok + 1):
        distinct = len({t[-pos] for t in tokens})
        if distinct > best_distinct:
            best_pos, best_distinct = pos, distinct
    ids = [int(t[-best_pos]) for t in tokens]
    order = np.argsort(ids)
    return [paths[i] for i in order], [ids[i] for i in order]


def dedup_one_file(path, expected_group_size):
    """Read one norm42 GEN file (single event), return deduplicated
    float64 per-particle arrays (px,py,pz,mass,charge,t,vx,vy,vz,pdg) --
    one representative per rotation group, first-encountered kept."""
    with uproot.open(path) as f:
        tree = f["events"]
        branches = {
            "pdg": "MCParticles.PDG",
            "charge": "MCParticles.charge",
            "mass": "MCParticles.mass",
            "px": "MCParticles.momentum.x",
            "py": "MCParticles.momentum.y",
            "pz": "MCParticles.momentum.z",
            "t": "MCParticles.time",
            "vx": "MCParticles.vertex.x",
            "vy": "MCParticles.vertex.y",
            "vz": "MCParticles.vertex.z",
        }
        arrays = tree.arrays(list(branches.values()), library="ak")

    raw = {}
    for key, br in branches.items():
        flat = ak.to_numpy(ak.flatten(arrays[br], axis=None))
        raw[key] = flat.astype(np.int64 if key == "pdg" else np.float64)

    p2 = raw["px"] ** 2 + raw["py"] ** 2 + raw["pz"] ** 2
    p = np.sqrt(p2)
    pt = np.hypot(raw["px"], raw["py"])
    theta = np.arctan2(pt, raw["pz"])

    # Same rotation-invariant key as gen_libtest_reconstruct_unrotated.py:
    # a pure z-axis rotation only touches (px,py)/(vx,vy) -- pdg, |p|,
    # theta, vz, t are all exactly preserved across the 42 clones.
    key_arr = np.stack([
        raw["pdg"].astype(np.float64),
        np.round(p, DECIMALS),
        np.round(theta, DECIMALS),
        np.round(raw["vz"], DECIMALS),
        np.round(raw["t"], DECIMALS),
    ], axis=1)
    _, first_idx, group_sizes = np.unique(
        key_arr, axis=0, return_index=True, return_counts=True)
    keep = np.sort(first_idx)

    median_gs = np.median(group_sizes)
    if abs(median_gs - expected_group_size) > 2:
        print(f"  WARNING: {os.path.basename(path)} median group size "
              f"{median_gs:.1f} deviates from expected {expected_group_size}"
              " -- inspect this file before trusting its output",
              flush=True)

    return {k: v[keep] for k, v in raw.items()}, group_sizes


def write_norm1_file(dedup, output_path):
    """Write one single-event EDM4hep GEN file from deduplicated arrays.
    Mirrors mucoll-benchmarks/generation/pgun/pgun_edm4hep.py's write
    pattern -- same MCParticleCollection/Writer/Frame API, adapted from
    "sample from a distribution" to "copy these known particles"."""
    import cppyy
    import edm4hep
    import podio
    from podio.root_io import Writer

    n = len(dedup["pdg"])
    writer = Writer(output_path)

    col = edm4hep.MCParticleCollection()
    evt = podio.Frame()
    evt.put_parameter("eventNumber", "0")
    for i in range(n):
        mcp = col.create()
        mcp.setPDG(int(dedup["pdg"][i]))
        mcp.setGeneratorStatus(1)
        mcp.setCharge(float(dedup["charge"][i]))
        mcp.setMass(float(dedup["mass"][i]))
        mcp.setTime(float(dedup["t"][i]))
        mcp.getMomentum().x = float(dedup["px"][i])
        mcp.getMomentum().y = float(dedup["py"][i])
        mcp.getMomentum().z = float(dedup["pz"][i])
        mcp.getVertex().x = float(dedup["vx"][i])
        mcp.getVertex().y = float(dedup["vy"][i])
        mcp.getVertex().z = float(dedup["vz"][i])
    evt.put(cppyy.gbl.std.move(col), "MCParticles")
    writer.write_frame(evt, "events")


def main():
    args = parse_args()
    files = glob.glob(os.path.join(args.input_dir, "*.root"))
    if not files:
        sys.exit(f"no *.root files in {args.input_dir}")
    files, cycles = sorted_by_cycle(files)
    if args.num_shards > 1:
        # Interleaved, not a contiguous block: naturally load-balances even
        # if particle-count-per-file (and hence dedup cost) varies across
        # the directory, since neighboring cycle IDs aren't assumed similar.
        files = files[args.shard_index::args.num_shards]
        cycles = cycles[args.shard_index::args.num_shards]
    if args.max_files:
        files = files[:args.max_files]
        cycles = cycles[:args.max_files]
    shard_note = (f" (shard {args.shard_index}/{args.num_shards})"
                 if args.num_shards > 1 else "")
    print(f"{len(files)} files from {args.input_dir}"
          f"{' (sampled)' if args.max_files else ''}{shard_note}")

    os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    all_group_sizes = []
    n_kept_total = 0
    for i, (path, cycle) in enumerate(zip(files, cycles)):
        dedup, group_sizes = dedup_one_file(path, args.expected_group_size)
        out_name = f"bib_gen_{cycle}.edm4hep.root"
        write_norm1_file(dedup, os.path.join(args.output_dir, out_name))
        n_kept_total += len(dedup["pdg"])
        all_group_sizes.append(group_sizes)
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            rate = (i + 1) / (time.time() - start)
            print(f"  {i + 1}/{len(files)} files, {n_kept_total:,} "
                  f"particles written, {rate:.2f} files/s", flush=True)

    all_group_sizes = np.concatenate(all_group_sizes) if all_group_sizes else np.array([])
    print(f"\ndone: {len(files)} files -> {args.output_dir}")
    if len(all_group_sizes):
        print(f"group size distribution: min={all_group_sizes.min()} "
              f"median={np.median(all_group_sizes):.1f} "
              f"mean={all_group_sizes.mean():.2f} max={all_group_sizes.max()}")


if __name__ == "__main__":
    main()
