#!/bin/bash
#SBATCH -J phi_half_long
#SBATCH -q shared
#SBATCH -C gpu
#SBATCH -A m5197_g
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -c 16
#SBATCH --gpus-per-task=1
#SBATCH --mem=100G
#SBATCH -t 07:00:00
#SBATCH --output=phi_half_long_%x_%j.out
#SBATCH --error=phi_half_long_%x_%j.err

# Longer continuation of raw_large + scaled_large (halved Phi) on the
# regular/shared GPU queue -- the debug queue's 30-min cap doesn't allow
# a real training window. 7-hour first window; if training hasn't
# early-stopped (--patience 15, default) or hit the 200-epoch cap by
# then, just resubmit this same script -- checkpoint/resume is automatic
# via the shared --label (see phi_half_long_task.sh).
#
# -C gpu (not gpu&hbm80g): the ~3800-4300 B/particle memory constant was
# measured for the ORIGINAL full-width Phi (200,200,256); halved Phi
# (100,100,128, summed layer widths 328 vs 656) should cost roughly half
# that per particle, since Phi's retained-for-backprop activations
# dominate this cost. Even a pessimistic worst-case-tail batch (N~2.5-3M,
# beyond anything observed in the 30-draw sample so far) lands around
# 19-22 GB -- comfortably under 40GB. Not directly benchmarked for this
# specific halved config, so a rare extreme-N batch OOMing on a 40GB node
# is possible; recoverable via the same checkpoint/resume as any other
# crash, just resubmit. Dropping hbm80g trades that small risk for a much
# larger node pool and shorter queue wait on the regular QOS.
#
# -q shared (not regular): verified directly against `sacctmgr show qos`
# -- "regular" + -C gpu resolves to the gpu_regular QOS, which is the
# FULL-NODE queue (exclusive access to all 4 GPUs regardless of how many
# are actually requested). "shared" + -C gpu resolves to gpu_shared
# (MaxNodes=1), the QOS actually meant for partial-node requests like
# this one (2 of 4 GPUs) -- bills for what's requested, not a full node.
#
# No --gpu-bind=single:1 under -q shared: that binding mode is documented
# for EXCLUSIVE-node GPU jobs and produced "Invalid generic resource (gres)
# specification" when combined with the shared QOS's sub-node GRES model
# (confirmed empirically -- first submission attempt failed on exactly
# this). --gpus-per-task=1 alone is sufficient; SLURM handles the binding.
#
# --mem=100G: shared QOS jobs need memory requested explicitly for
# accounting (exclusive queues default to "whole node," shared ones
# don't). Real need is non-trivial here: libtest_common.Store eagerly
# loads the FULL norm42 store into host RAM per task (~30GB, all 9 raw
# fields x ~835M particles), and each of the 2 tasks on this node loads
# its own independent copy -- roughly 65-70GB combined before any
# TF/Python overhead. 100G leaves margin; not empirically verified yet,
# watch the first job for an OOM-kill if it's still too tight.
#
# Lightweight logging: one line per epoch (~240s/epoch observed in the
# debug scan), same format as before. Check progress live with:
#   tail -f phi_half_long_halfphi_raw_large_<jobid>.out
#   tail -f phi_half_long_halfphi_scaled_large_<jobid>.out
# or watch history.csv update every epoch:
#   watch -n 30 'tail -3 pfn_results/halfphi_raw_large/history.csv \
#                pfn_results/halfphi_scaled_large/history.csv'

set -e
cd "$SLURM_SUBMIT_DIR"
module load tensorflow

export N_FILES=${N_FILES:-420}
export UNITS_PER_EPOCH=${UNITS_PER_EPOCH:-500}
export MAX_MINUTES=${MAX_MINUTES:-400}

# --kill-on-bad-exit=0: one rank crashing must not take the other,
# still-informative rank down with it.
srun --cpu-bind=cores --kill-on-bad-exit=0 bash phi_half_long_task.sh || \
    echo "one or more ranks exited non-zero; check individual .out files"

echo ""
echo "=== per-rank history tail (last 5 epochs) ==="
for label in halfphi_raw_large halfphi_scaled_large; do
    hist="pfn_results/${label}/history.csv"
    if [ -f "$hist" ]; then
        echo "-- $label --"
        tail -5 "$hist"
    else
        echo "-- $label -- NO history.csv"
    fi
done
