#!/bin/bash
#SBATCH -J reco420_stores
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A m5197
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --output=reco420_stores_%j.out
#SBATCH --error=reco420_stores_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
module load python

conda run -p "$PSCRATCH/mucoll/envs/mucoll-inspect" \
    python make_reco_libtest_stores.py --n-files 420
