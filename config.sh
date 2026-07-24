#!/bin/bash
# OSCAR paths shared by the simulation and BIB-reuse workflows.
# Ported from the original Perlmutter config.sh (shifter/CFS/PSCRATCH) for
# the reco-level particle-gun + BIB-overlay experiment redo on OSCAR.

# mucoll-slurm and mucoll-benchmarks are sister repositories.
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export WORK_DIR
export MUCOLL_BENCHMARKS_PATH="$WORK_DIR/mucoll-benchmarks"

export DATA_GROUP_DIR="/oscar/data/mleblan6/mucoll"
# apptainer .sif, not a shifter docker: reference -- same image already used
# by every other GPU/CPU mucoll-sim step this session (submit_norm1_ddsim.slurm
# et al.), so DIGI/RECO here use the identical software stack as the SIM
# library they overlay.
export IMAGE="$DATA_GROUP_DIR/mucoll-sim-ubuntu24:v3.0.sif"
export GEOM_NAME="MAIA_v0"
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/oscar/scratch/$USER/mucoll/output}"

# Default overlay library: this session's regenerated norm1 SIM (genuine
# GEN-level dedup + ddsim, both polarities, 6666/6666 files each -- see
# gen_libtest_write_norm1_root.py / submit_norm1_ddsim.slurm). RECO study
# jobs override these paths with their immutable train/validation/test pools.
export BIB_DIR="${BIB_DIR:-/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/SIM}"
export BIB_MUPLUS="${BIB_MUPLUS:-$BIB_DIR/MUPLUS/}"
export BIB_MUMINUS="${BIB_MUMINUS:-$BIB_DIR/MUMINUS/}"
export BIB_NUMBER="${BIB_NUMBER:-6665}"
