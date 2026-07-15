#!/bin/bash
#SBATCH -J reco420_pfn
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:30:00
#SBATCH --output=reco420_pfn_%j.out
#SBATCH --error=reco420_pfn_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

STORE_DIR=${STORE_DIR:-$PSCRATCH/mucoll/libtest/reco_n420_pfn_stores}

pids=()
for seed in 1 2; do
    srun --exclusive --exact -N 1 -n 1 -c 32 --gpus-per-task=1 --gpu-bind=single:1 \
        python train_reco_libtest_pfn.py \
            --store-dir "$STORE_DIR" --class-a U --class-b R \
            --seed "$seed" --label "reco_n420_U_vs_R_seed${seed}" &
    pids+=("$!")
done

for seed in 1 2; do
    srun --exclusive --exact -N 1 -n 1 -c 32 --gpus-per-task=1 --gpu-bind=single:1 \
        python train_reco_libtest_pfn.py \
            --store-dir "$STORE_DIR" --class-a U --class-b null_b \
            --seed "$seed" --label "reco_n420_null_seed${seed}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
