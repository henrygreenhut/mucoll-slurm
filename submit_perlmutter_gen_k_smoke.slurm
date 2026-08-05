#!/bin/bash
#SBATCH -J gen_k_smoke
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:30:00
#SBATCH -o perlmutter_gen_k_smoke_%j.out
#SBATCH -e perlmutter_gen_k_smoke_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CONFIG="perlmutter_gen_k_smoke_${SLURM_JOB_ID}.conf"
cat > "$CONFIG" <<EOF
0 env DIRECT_RUN=1 bash submit_perlmutter_gen_k.slurm 5 main smoke
1 env DIRECT_RUN=1 bash submit_perlmutter_gen_k.slurm 42-synthetic main smoke
2 env DIRECT_RUN=1 bash submit_perlmutter_gen_k.slurm 42-production main smoke
3 env DIRECT_RUN=1 bash submit_perlmutter_gen_k.slurm 42-synthetic null smoke
EOF

srun --cpu-bind=cores --multi-prog "$CONFIG"
