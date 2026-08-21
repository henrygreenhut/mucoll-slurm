#!/bin/bash
#SBATCH -J bib_muon_groups
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH -t 00:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/bib_muon_groups_%j.out
#SBATCH -e logs/bib_muon_groups_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

IMAGE=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
export BIB_MUON_GROUPS=8

srun \
    --ntasks=8 \
    --cpus-per-task=2 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    shifter --image="$IMAGE" \
    bash "$SLURM_SUBMIT_DIR/run_perlmutter_bib_muon_groups.sh"

BASE=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-split-muon-v1/decays-containing-muon-poisson-norot
srun \
    --ntasks=1 \
    --cpus-per-task=2 \
    shifter --image="$IMAGE" \
    python3 - "$BASE" <<'PY'
import glob
import sys
import uproot

base = sys.argv[1]
for polarity in ("MUPLUS", "MUMINUS"):
    paths = sorted(glob.glob(base + "/SIM/" + polarity + "/bib_sim_*.edm4hep.root"))
    assert len(paths) >= 8, (polarity, len(paths))
    for path in paths[:8]:
        root = uproot.open(path)
        assert "podio_metadata" in root, path
        assert root["events"].num_entries == 1, path
    print(polarity, "valid SIM files:", len(paths[:8]))
PY
