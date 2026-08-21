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
#SBATCH --image=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
#SBATCH -o logs/bib_muon_groups_%j.out
#SBATCH -e logs/bib_muon_groups_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export BIB_MUON_GROUPS=8

srun \
    --ntasks=8 \
    --cpus-per-task=2 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    shifter \
    bash "$SLURM_SUBMIT_DIR/run_perlmutter_bib_muon_groups.sh"

BASE=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-split-muon-v1/decays-containing-muon-poisson-norot
for polarity in MUPLUS MUMINUS; do
    count=$(find "$BASE/SIM/$polarity" -name 'bib_sim_*.edm4hep.root' -type f | wc -l)
    if [ "$count" -lt "$BIB_MUON_GROUPS" ]; then
        echo "expected $BIB_MUON_GROUPS SIM files for $polarity; found $count" >&2
        exit 1
    fi
    echo "$polarity valid SIM files: $count"
done
