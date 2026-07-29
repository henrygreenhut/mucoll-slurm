#!/bin/bash
#SBATCH -J reco_dist
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --output=reco_distributions_%j.out
#SBATCH --error=reco_distributions_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ENV_DIR=/oscar/scratch/$USER/mucoll/envs/reco-store
BASE=/oscar/scratch/$USER/mucoll/libtest
REQUESTED_SIZE=${1:-all}

case "$REQUESTED_SIZE" in
    all) SIZES=(420 840) ;;
    420|840) SIZES=("$REQUESTED_SIZE") ;;
    *)
        echo "usage: sbatch $0 [all|420|840]" >&2
        exit 2
        ;;
esac

if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "ERROR: missing $ENV_DIR; run ./setup_oscar_reco_envs.sh first"
    exit 1
fi

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$MPLCONFIGDIR"

for n_files in "${SIZES[@]}"; do
    store_dir=$BASE/reco_n${n_files}_pfn_stores_trackfix_val25
    outdir=$SLURM_SUBMIT_DIR/plots/reco_n${n_files}_trackfix_val25_whole_distributions
    echo "N=$n_files store=$store_dir output=$outdir"
    "$ENV_DIR/bin/python" plot_reco_distributions.py \
        --n-files "$n_files" \
        --store-dir "$store_dir" \
        --outdir "$outdir"
done
