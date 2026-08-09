#!/bin/bash
#SBATCH -J gen_k1_audit_smoke
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH -t 00:30:00
#SBATCH -o perlmutter_gen_k1_audit_smoke_%j.out
#SBATCH -e perlmutter_gen_k1_audit_smoke_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CONFIG="perlmutter_gen_k1_audit_smoke_${SLURM_JOB_ID}.conf"
cat > "$CONFIG" <<EOF
0 bash submit_perlmutter_gen_k1_seed.slurm baseline 2 main smoke
1 bash submit_perlmutter_gen_k1_seed.slurm no-phi 1 main smoke
2 bash submit_perlmutter_gen_k1_seed.slurm baseline 1 null smoke
3 bash submit_perlmutter_gen_k1_seed.slurm no-phi 1 null smoke
EOF

srun --cpu-bind=cores --multi-prog "$CONFIG"
