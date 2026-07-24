#!/bin/bash
#SBATCH -J norm1_muplus_ddsim_fixups
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH -t 02:00:00
#SBATCH -o norm1_muplus_ddsim_fixups_%x_%j.out
#SBATCH -e norm1_muplus_ddsim_fixups_%x_%j.err

# One-off fix: a size-outlier sweep (median-relative, not the flat 1MB
# threshold that gave false positives against norm1's naturally smaller
# files) found 16 files in norm1's regenerated SIM/MUPLUS that are all
# exactly 404 bytes -- empty ROOT file shells (ROOT opens them fine but
# reports "has no keys"), same failure signature as the corrupted norm42
# MUMINUS file a user report caught. Root cause: submit_norm1_ddsim.slurm's
# idempotent skip-if-exists check only looks at file *existence*, not
# validity -- if ddsim crashed/was killed mid-write leaving a stub, a
# resubmit would never retry it. These 16 cycles' GEN inputs are all
# confirmed present and normal-sized, so this is a plain re-run, not a
# GEN-level problem.
#
# All 16 explicitly deleted first (not just relying on ddsim's
# --outputFile overwrite) so this stays consistent with the idempotent
# skip-if-exists pattern if it's ever rerun.
#
# Submit once:
#   sbatch submit_norm1_muplus_ddsim_fixups.slurm

set -e
cd "$SLURM_SUBMIT_DIR"

BENCH=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks
CONTAINER=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
GEN_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/GEN/MUPLUS
SIM_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/SIM/MUPLUS

CYCLES=(4262 4344 4872 5127 5210 5253 5332 5334 5778 6193 6196 6268 6369 6379 6380 6523)

WORKDIR=$(mktemp -d "/oscar/scratch/$USER/norm1_muplus_ddsim_fixups.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

LOOP_SCRIPT="$WORKDIR/run_fixups.sh"
{
    echo "source /opt/setup_mucoll.sh"
    echo "source '$BENCH/setup_config.sh' '$BENCH' MAIA_v0"
    echo "cd '$WORKDIR'"
    for cycle in "${CYCLES[@]}"; do
        input_file="$GEN_DIR/bib_gen_${cycle}.edm4hep.root"
        output_file="$SIM_DIR/bib_sim_${cycle}.edm4hep.root"
        echo "rm -f '$output_file'"
        echo "echo '--- cycle $cycle ---'"
        echo "ddsim --steeringFile '$BENCH/simulation/steer_baseline.py' \\"
        echo "    --numberOfEvents 1 \\"
        echo "    --inputFiles '$input_file' \\"
        echo "    --outputFile '$output_file'"
    done
} > "$LOOP_SCRIPT"

apptainer exec --bind /oscar:/oscar "$CONTAINER" bash "$LOOP_SCRIPT"

echo "=== verifying output sizes ==="
for cycle in "${CYCLES[@]}"; do
    f="$SIM_DIR/bib_sim_${cycle}.edm4hep.root"
    stat -c "%s %n" "$f"
done
echo "fixups done"
