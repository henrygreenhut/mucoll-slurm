#!/bin/bash
# Perlmutter paths shared by the simulation and BIB-reuse workflows.

# mucoll-slurm and mucoll-benchmarks are sister repositories.
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export WORK_DIR
export MUCOLL_BENCHMARKS_PATH="$WORK_DIR/mucoll-benchmarks"

export DATA_GROUP_DIR="/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data"
export IMAGE="docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0"
export GEOM_NAME="MAIA_v0"
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PSCRATCH:-$SCRATCH}/mucoll/output}"

# Default overlay library. RECO study jobs override these paths with their
# immutable train/validation/test pools.
export BIB_DIR="${BIB_DIR:-$DATA_GROUP_DIR/bib-v3p0-fmt2-norm1/SIM}"
export BIB_MUPLUS="${BIB_MUPLUS:-$BIB_DIR/MUPLUS/}"
export BIB_MUMINUS="${BIB_MUMINUS:-$BIB_DIR/MUMINUS/}"
export BIB_NUMBER="${BIB_NUMBER:-6665}"
