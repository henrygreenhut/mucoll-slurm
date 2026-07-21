#!/bin/bash
#SBATCH -J reco420_pfn
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH --output=reco420_pfn_%j.out
#SBATCH --error=reco420_pfn_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

STORE_DIR=${STORE_DIR:-$PSCRATCH/mucoll/libtest/reco_n420_pfn_stores_simple}

python train_reco_libtest_pfn.py \
    --store-dir "$STORE_DIR" --class-a U --class-b R \
    --label "reco_n420_simple_U_vs_R"

python train_reco_libtest_pfn.py \
    --store-dir "$STORE_DIR" --class-a U --class-b null_b \
    --label "reco_n420_simple_null"
