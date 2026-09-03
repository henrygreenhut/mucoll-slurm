#!/bin/bash
#SBATCH -J photon_z_harmonics
#SBATCH -p batch
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16g
#SBATCH -t 01:00:00
#SBATCH --exclude=node2336
#SBATCH -o photon_z_harmonics_%j.out
#SBATCH -e photon_z_harmonics_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load ngc-tensorflow-container/25.02-tf2-py3-azkb

STORE_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/libtest/stores
OUTPUT_DIR=${MOTHER_Z_OUTPUT_DIR:-$STORE_DIR/photon_harmonics_vs_mother_z}
BIN_WIDTH_MM=${BIN_WIDTH_MM:-5000}
MINIMUM_PHOTONS=${MINIMUM_PHOTONS:-10000}
MAXIMUM_DISTANCE_M=${MAXIMUM_DISTANCE_M:-60}

MUPLUS=${MUPLUS_BANK:-$STORE_DIR/gen_split_mothers_MUPLUS.h5}
if [ ! -s "$MUPLUS" ]; then
    echo "ERROR: missing $MUPLUS" >&2
    exit 1
fi

apptainer exec --bind /oscar:/oscar "$NGC_TENSORFLOW_CONTAINER" \
    python -u photon_harmonics_vs_mother_z.py \
        "$MUPLUS" \
        --output-directory "$OUTPUT_DIR" \
        --bin-width-mm "$BIN_WIDTH_MM"

apptainer exec --bind /oscar:/oscar "$NGC_TENSORFLOW_CONTAINER" \
    python plot_photon_harmonics_vs_mother_z.py \
        "$OUTPUT_DIR/photon_harmonics_vs_mother_z.npz" \
        --output-prefix "$OUTPUT_DIR/photon_harmonics_vs_mother_z" \
        --minimum-photons "$MINIMUM_PHOTONS" \
        --maximum-distance-m "$MAXIMUM_DISTANCE_M"
