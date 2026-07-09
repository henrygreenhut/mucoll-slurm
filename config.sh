#!/bin/bash
# =============================================================================
#  mucoll-slurm : single configuration file
# =============================================================================
#  This is the main file you should normally need to edit.
#  It is sourced both on the login node (by the Python submit scripts) and
#  inside the container (by the chain scripts), so stick to just `export`s.
#
#  Quick start:
#    1. Make sure your output area exists:  the scripts create it for you under
#       OUTPUT_BASE_DIR below, but you need write access to DATA_GROUP_DIR.
#    2. (One time) pull the v3.0 image to a shared SIF -- see README.md.
#    3. Run:  python submit_pgun.py        (particle gun, no BIB)
#       or flip BIB=True inside submit_pgun.py for "with BIB".
# =============================================================================

# --- Repository layout (auto-detected) ---------------------------------------
# WORK_DIR = the directory that contains both mucoll-slurm/ and mucoll-benchmarks/.
# Derived from this file's own location so the checkout can live anywhere.
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export WORK_DIR
export MUCOLL_BENCHMARKS_PATH="$WORK_DIR/mucoll-benchmarks"

# --- Shared group storage -----------------------------------------------------
# Perlmutter trial setup.
export DATA_GROUP_DIR="/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data"

# Previous Oscar setup:
# export DATA_GROUP_DIR="/oscar/data/mleblan6/mucoll"

# --- Container image (mucoll-spack v3.0, sim layer, ubuntu24) -----------------
# Perlmutter trial setup. This image string is meant for Shifter-based jobs.
export IMAGE="docker:ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0"
export WHIZARD_IMAGE="$IMAGE"

# Previous Oscar setup:
# export IMAGE="$DATA_GROUP_DIR/mucoll-sim-ubuntu24:v3.0.sif"
# export WHIZARD_IMAGE="$DATA_GROUP_DIR/mucoll-sim-ubuntu24:main.sif"

# --- Detector geometry -------------------------------------------------------
export GEOM_NAME="MAIA_v0"

# --- Output ------------------------------------------------------------------
# Perlmutter trial setup.
export OUTPUT_BASE_DIR="${PSCRATCH:-$SCRATCH}/mucoll/output"

# Previous Oscar setup:
# export OUTPUT_BASE_DIR="$DATA_GROUP_DIR/$USER/output"

# --- Container bind mounts ----------------------------------------------------
# Perlmutter trial setup.
export DATA_BIND="/global/cfs/cdirs/m5197,${PSCRATCH:-$SCRATCH},$HOME"

# Previous Oscar setup:
# export DATA_BIND="$DATA_GROUP_DIR"

# --- Beam-Induced Background (BIB) overlay samples ---------------------------
# Used only when a job is launched "with BIB". Each path is a DIRECTORY of
# *.edm4hep.root files (trailing slash matters); the overlay enumerates them.
export BIB_DIR="$DATA_GROUP_DIR/bib-v3p0-fmt2-norm1/SIM"
export BIB_MUPLUS="$BIB_DIR/MUPLUS/"
export BIB_MUMINUS="$BIB_DIR/MUMINUS/"
export BIB_NUMBER=6665

# Broken Perlmutter benchmark setup:
# export BIB_DIR="$DATA_GROUP_DIR/bib-v3p0-benchmark/SIM"
# export BIB_MUPLUS="$BIB_DIR/MUPLUS/"
# export BIB_MUMINUS="$BIB_DIR/MUMINUS/"
# export BIB_NUMBER=812

# Previous Oscar setup:
# export BIB_DIR="$DATA_GROUP_DIR/bib/Feb12/10TeV_MAIA_edm4hep"
# export BIB_MUPLUS="$BIB_DIR/MUPLUS/"
# export BIB_MUMINUS="$BIB_DIR/MUMINUS/"
# export BIB_NUMBER=812
