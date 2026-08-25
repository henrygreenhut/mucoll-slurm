#!/bin/bash
#SBATCH -J reco420_bh_mult
#SBATCH -A m5197
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o logs/reco420_bh_mult_%j.out
#SBATCH -e logs/reco420_bh_mult_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load python
conda activate "$PSCRATCH/mucoll/envs/mucoll-inspect"

python plot_reco_split_bh_multiplicity.py \
    --legacy-dir "$PSCRATCH/mucoll/reco_n420_legacy_no6291_v31" \
    --split-dir "$PSCRATCH/mucoll/reco_n420_split_bh_v31" \
    --outdir plots/reco_n420_split_bh_comparison/multiplicities
