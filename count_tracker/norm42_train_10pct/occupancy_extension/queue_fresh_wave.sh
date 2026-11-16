#!/usr/bin/env bash
set -euo pipefail
for name in CT_DIR SCAN_ROOT MODEL_10PCT PAPER1_ROOT GENBIB_DIR GEOMAP \
            IMAGE_V2_11 IMAGE BENCHMARK_DIR; do
  [ -n "${!name:-}" ] || { echo "export $name before queueing" >&2; exit 1; }
done
EVENTS_PER_POINT="${EVENTS_PER_POINT:-5}"; mkdir -p "$SCAN_ROOT/logs"
DEPENDENCY_ARGS=(); [ -z "${AFTEROK:-}" ] || DEPENDENCY_ARGS=(--dependency="afterok:$AFTEROK")

COND=$(sbatch --parsable "${DEPENDENCY_ARGS[@]}" --array=0-0 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT" \
  -o "$SCAN_ROOT/logs/conditions_%A_%a.out" -e "$SCAN_ROOT/logs/conditions_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_conditions.slurm")
SAMPLE=$(sbatch --parsable --dependency="afterok:$COND" --array=0-1%2 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",MODEL_ROOT="$MODEL_10PCT",PAPER1_ROOT="$PAPER1_ROOT",SHARDS=2,SEED_BASE="${SEED_BASE:-12345}",MAX_ROUNDS="${MAX_ROUNDS:-100}",UNFILLED_POLICY=drop \
  -o "$SCAN_ROOT/logs/sample_%A_%a.out" -e "$SCAN_ROOT/logs/sample_%A_%a.err" \
  "$CT_DIR/high_occupancy_scan/submit_sampling.slurm")
INPUT=$(sbatch --parsable --dependency="afterok:$SAMPLE" --array="0-$((2 * EVENTS_PER_POINT - 1))%6" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",EVENTS_PER_POINT="$EVENTS_PER_POINT",GENBIB_DIR="$GENBIB_DIR",GEOMAP="$GEOMAP",IMAGE_V2_11="$IMAGE_V2_11" \
  -o "$SCAN_ROOT/logs/input_%A_%a.out" -e "$SCAN_ROOT/logs/input_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_pair_input.slurm")
DIGI=$(sbatch --parsable --dependency="afterok:$INPUT" --array="0-$((2 * EVENTS_PER_POINT - 1))%6" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",EVENTS_PER_POINT="$EVENTS_PER_POINT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR" \
  -o "$SCAN_ROOT/logs/digi_%A_%a.out" -e "$SCAN_ROOT/logs/digi_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_digi.slurm")
SPLIT=$(sbatch --parsable --dependency="afterok:$DIGI" --array="0-$((10 * EVENTS_PER_POINT - 1))%10" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",EVENTS_PER_POINT="$EVENTS_PER_POINT",IMAGE="$IMAGE" \
  -o "$SCAN_ROOT/logs/split_%A_%a.out" -e "$SCAN_ROOT/logs/split_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_split.slurm")
RECO=$(sbatch --parsable --dependency="afterok:$SPLIT" --array="0-$((10 * EVENTS_PER_POINT - 1))%10" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",EVENTS_PER_POINT="$EVENTS_PER_POINT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR" \
  -o "$SCAN_ROOT/logs/reco_%A_%a.out" -e "$SCAN_ROOT/logs/reco_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_reco.slurm")
REPORT=$(sbatch --parsable --dependency="afterok:$RECO" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT" \
  -o "$SCAN_ROOT/logs/report_%j.out" -e "$SCAN_ROOT/logs/report_%j.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_fresh_report.slurm")
printf 'conditions=%s sampling=%s inputs=%s digitization=%s split=%s reco=%s report=%s\n' \
  "$COND" "$SAMPLE" "$INPUT" "$DIGI" "$SPLIT" "$RECO" "$REPORT"
