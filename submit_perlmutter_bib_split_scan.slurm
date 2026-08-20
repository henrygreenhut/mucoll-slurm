#!/bin/bash
#SBATCH -J bib_split_scan
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 00:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH -o logs/bib_split_scan_%j.out
#SBATCH -e logs/bib_split_scan_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

IMAGE=docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
SOURCE=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-norm1-split-mother-norot/GEN
OUTPUT=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-split-muon-v1/manifests/partition.json

mkdir -p logs

shifter --image="$IMAGE" bash -c "
    source /opt/setup_mucoll.sh
    python3 -u '$SLURM_SUBMIT_DIR/bib_split_muon_gen.py' scan \
        --source '$SOURCE' \
        --output '$OUTPUT' \
        --workers 64
"

python3 -m json.tool "$OUTPUT" >/dev/null
echo "$OUTPUT"
