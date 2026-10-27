#!/bin/bash
# Queue the reduced-chi2 repeat of the source-disjoint angular-patch PFN.

set -euo pipefail

: "${CT_DIR:?export CT_DIR as the count_tracker directory}"
: "${SCAN_ROOT:?export SCAN_ROOT as the source-disjoint cohort directory}"

PATCH_ROOT_CHI2="${PATCH_ROOT_CHI2:-$SCAN_ROOT/n1666/angular_patches_phi72_chi2ndf}"
PFN_OUT_CHI2="${PFN_OUT_CHI2:-$SCAN_ROOT/n1666/angular_patch_pfn_source_disjoint_chi2ndf}"

if [ -e "$PATCH_ROOT_CHI2/angular_patch_summary.json" ]; then
    echo "Refusing existing chi2 patch store: $PATCH_ROOT_CHI2" >&2
    exit 1
fi
if [ -d "$PFN_OUT_CHI2/source_disjoint_fullbx_theta_phi72_chi2ndf" ]; then
    echo "Refusing existing chi2 PFN result" >&2
    exit 1
fi

mkdir -p "$PATCH_ROOT_CHI2/logs" "$PFN_OUT_CHI2/logs"

patch_job=$(sbatch --parsable \
    --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",PATCH_ROOT="$PATCH_ROOT_CHI2",POINT=n1666,PHI_SECTORS=72,INCLUDE_CHI2_NDF=1,SPLIT_COUNTS='2 1 1' \
    -o "$PATCH_ROOT_CHI2/logs/build_%j.out" \
    -e "$PATCH_ROOT_CHI2/logs/build_%j.err" \
    "$CT_DIR/v3_occupancy_scan/submit_angular_patches.slurm")

train_job=$(sbatch --parsable \
    --dependency="afterok:$patch_job" \
    --export=ALL,CT_DIR="$CT_DIR",PATCH_ROOT="$PATCH_ROOT_CHI2",PFN_OUT="$PFN_OUT_CHI2",CONSTRUCTION=norm42_fullbx_theta_phi72_chi2ndf,LABEL=source_disjoint_fullbx_theta_phi72_chi2ndf,EPOCHS=1000,PATIENCE=100 \
    -o "$PFN_OUT_CHI2/logs/train_%j.out" \
    -e "$PFN_OUT_CHI2/logs/train_%j.err" \
    "$CT_DIR/v3_occupancy_scan/submit_angular_pfn.slurm")

printf 'chi2 patches=%s chi2 training=%s\n' "$patch_job" "$train_job"
