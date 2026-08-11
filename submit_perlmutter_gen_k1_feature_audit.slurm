#!/bin/bash
#SBATCH -J gen_k1_features
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o perlmutter_gen_k1_features_%j.out
#SBATCH -e perlmutter_gen_k1_features_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MODE=${1:-full}
case "$MODE" in
    smoke)
        TEST_UNITS=4
        LABEL=pm_n420_k1_feature_audit_smoke
        ;;
    full)
        TEST_UNITS=300
        LABEL=pm_n420_k1_feature_audit_v1
        ;;
    *)
        echo "usage: $0 [smoke|full]" >&2
        exit 2
        ;;
esac

module load tensorflow/2.15.0

python -u audit_gen_k1_features.py \
    --test-units "$TEST_UNITS" \
    --label "$LABEL"
