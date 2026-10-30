#!/bin/bash
# Queue an already-prepared single-point high-occupancy scan.

set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${PAPER1_ROOT:?}"
: "${GENBIB_DIR:?}"; : "${GEOMAP:?}"; : "${IMAGE_V2_11:?}"
: "${IMAGE:?}"; : "${BENCHMARK_DIR:?}"; : "${EVENTS_PER_POINT:?}"

SHARDS="${SHARDS:-1}"
HEAVY_MEM="${HEAVY_MEM:-180G}"
RECO_MEM="${RECO_MEM:-120G}"
mkdir -p "$SCAN_ROOT/logs"
test -f "$SCAN_ROOT/scan.json"

COND=$(sbatch --parsable --array=0-0 --mem="$RECO_MEM" --time=2-00:00:00 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT" \
  -o "$SCAN_ROOT/logs/conditions_%A_%a.out" \
  -e "$SCAN_ROOT/logs/conditions_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_conditions.slurm")

SAMPLE=$(sbatch --parsable --dependency="afterok:$COND" \
  --array="0-$((SHARDS - 1))%2" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",MODEL_ROOT="$MODEL_ROOT",PAPER1_ROOT="$PAPER1_ROOT",SHARDS="$SHARDS",SEED_BASE=12345,MAX_ROUNDS=100,UNFILLED_POLICY=drop \
  -o "$SCAN_ROOT/logs/sample_%A_%a.out" \
  -e "$SCAN_ROOT/logs/sample_%A_%a.err" \
  "$CT_DIR/high_occupancy_scan/submit_sampling.slurm")

INPUT=$(sbatch --parsable --dependency="afterok:$SAMPLE" \
  --array="0-$((EVENTS_PER_POINT - 1))%1" --mem="$HEAVY_MEM" --time=2-00:00:00 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",GENBIB_DIR="$GENBIB_DIR",GEOMAP="$GEOMAP",IMAGE_V2_11="$IMAGE_V2_11",EVENTS_PER_POINT="$EVENTS_PER_POINT" \
  -o "$SCAN_ROOT/logs/input_%A_%a.out" \
  -e "$SCAN_ROOT/logs/input_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_inputs.slurm")

DIGI=$(sbatch --parsable --dependency="afterok:$INPUT" \
  --array="0-$((2 * EVENTS_PER_POINT - 1))%1" --mem="$HEAVY_MEM" --time=2-00:00:00 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR",EVENTS_PER_POINT="$EVENTS_PER_POINT" \
  -o "$SCAN_ROOT/logs/digi_%A_%a.out" \
  -e "$SCAN_ROOT/logs/digi_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_digi.slurm")

SPLIT=$(sbatch --parsable --dependency="afterok:$DIGI" \
  --array="0-$((10 * EVENTS_PER_POINT - 1))%2" --mem="$HEAVY_MEM" --time=2-00:00:00 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",IMAGE="$IMAGE",EVENTS_PER_POINT="$EVENTS_PER_POINT" \
  -o "$SCAN_ROOT/logs/split_%A_%a.out" \
  -e "$SCAN_ROOT/logs/split_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_split.slurm")

RECO=$(sbatch --parsable --dependency="afterok:$SPLIT" \
  --array="0-$((10 * EVENTS_PER_POINT - 1))%5" --mem="$RECO_MEM" --time=2-00:00:00 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR",EVENTS_PER_POINT="$EVENTS_PER_POINT" \
  -o "$SCAN_ROOT/logs/reco_%A_%a.out" \
  -e "$SCAN_ROOT/logs/reco_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_reco.slurm")

printf 'conditions=%s sampling=%s inputs=%s digitization=%s split=%s reco=%s\n' \
  "$COND" "$SAMPLE" "$INPUT" "$DIGI" "$SPLIT" "$RECO"
