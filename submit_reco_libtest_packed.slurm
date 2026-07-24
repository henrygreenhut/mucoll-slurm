#!/bin/bash
#SBATCH -J reco_libtest_packed
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH -t 08:00:00
#SBATCH --array=0-63
#SBATCH -o reco_libtest_packed_%x_%A_%a.out
#SBATCH -e reco_libtest_packed_%x_%A_%a.err

# Ported from the original Perlmutter version, which requested N whole CPU
# nodes (--nodes, --ntasks-per-node=64, --cpus-per-task=4 -- up to 256
# hardware threads/node) and used srun to launch one shifter rank per
# manifest row, indexed by SLURM_PROCID. OSCAR's batch partition caps this
# account at MaxTRESPU=cpu=64 total regardless of how many nodes are
# requested, so that model doesn't translate -- this uses the same 64-way
# interleaved-shard pattern as submit_norm1_ddsim.slurm/
# submit_norm42_muminus_ddsim.slurm instead: 64 single-CPU array tasks, each
# looping sequentially over its assigned manifest rows and calling
# run_reco_libtest_task.sh once per row with an explicit line number.
#
# Idempotent per row (run_reco_libtest_task.sh skips existing reco_output
# files), so a resubmit after a TIMEOUT only redoes what's missing.
#
# Submit (manifest path required):
#   sbatch submit_reco_libtest_packed.slurm logs/reco_n420_<timestamp>.tsv

set -e
cd "$SLURM_SUBMIT_DIR"

MANIFEST=$1
if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
    echo "usage: sbatch submit_reco_libtest_packed.slurm <manifest.tsv>" >&2
    exit 1
fi

NUM_SHARDS=64
TASKS=$(wc -l < "$MANIFEST")
echo "shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS: manifest has $TASKS total rows"

# This shard's interleaved 1-indexed line numbers (sed -n 'Np' is 1-indexed):
# shard_index+1, shard_index+1+64, ... -- same load-balancing rationale as
# every other 64-way sharded script this session.
for ((line = SLURM_ARRAY_TASK_ID + 1; line <= TASKS; line += NUM_SHARDS)); do
    bash "$SLURM_SUBMIT_DIR/run_reco_libtest_task.sh" "$MANIFEST" "$line"
done

echo "shard $SLURM_ARRAY_TASK_ID done"
