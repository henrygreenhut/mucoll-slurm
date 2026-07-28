#!/bin/bash
#SBATCH -J reco420_plots
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --output=reco420_plots_%j.out
#SBATCH --error=reco420_plots_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ENV_DIR=/oscar/scratch/$USER/mucoll/envs/reco-store
STORE_DIR=${STORE_DIR:-/oscar/scratch/$USER/mucoll/libtest/reco_n420_pfn_stores_simple}
OUTDIR=${OUTDIR:-"$SLURM_SUBMIT_DIR/plots/reco_n420_directlog_whole_distributions"}

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "ERROR: missing $ENV_DIR; run ./setup_oscar_reco_envs.sh first"
    exit 1
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$MPLCONFIGDIR"

"$ENV_DIR/bin/python" plot_reco_distributions.py \
    --store-dir "$STORE_DIR" \
    --outdir "$OUTDIR"
