#!/usr/bin/env bash
# Queue input through report for a prepared reuse plan after sampling completes.
set -euo pipefail

for name in CT_DIR EXT_ROOT AFTEROK GENBIB_DIR GEOMAP IMAGE_V2_11 IMAGE BENCHMARK_DIR; do
  [ -n "${!name:-}" ] || { echo "export $name before queueing" >&2; exit 1; }
done
PLAN="$EXT_ROOT/plan.json"; test -f "$PLAN"; mkdir -p "$EXT_ROOT/logs"
HELPER="$CT_DIR/norm42_train_10pct/occupancy_extension/plan_task.py"
EVENTS=$(python3 "$HELPER" --plan "$PLAN" --stage event --index 0 --count)
REGIONS=$(python3 "$HELPER" --plan "$PLAN" --stage region --index 0 --count)

INPUT=$(sbatch --parsable --dependency="afterok:$AFTEROK" \
  --array="0-$((EVENTS - 1))%${INPUT_CONCURRENCY:-6}" \
  --mem="${INPUT_MEM:-64G}" --time="${INPUT_TIME:-24:00:00}" \
  --export=ALL,CT_DIR="$CT_DIR",EXT_ROOT="$EXT_ROOT",GENBIB_DIR="$GENBIB_DIR",GEOMAP="$GEOMAP",IMAGE_V2_11="$IMAGE_V2_11" \
  -o "$EXT_ROOT/logs/input_%A_%a.out" -e "$EXT_ROOT/logs/input_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_input.slurm")

DIGI=$(sbatch --parsable --dependency="afterok:$INPUT" \
  --array="0-$((EVENTS - 1))%${DIGI_CONCURRENCY:-6}" \
  --mem="${DIGI_MEM:-64G}" --time="${DIGI_TIME:-24:00:00}" \
  --export=ALL,CT_DIR="$CT_DIR",EXT_ROOT="$EXT_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR" \
  -o "$EXT_ROOT/logs/digi_%A_%a.out" -e "$EXT_ROOT/logs/digi_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_digi.slurm")

SPLIT=$(sbatch --parsable --dependency="afterok:$DIGI" \
  --array="0-$((REGIONS - 1))%${SPLIT_CONCURRENCY:-10}" \
  --mem="${SPLIT_MEM:-96G}" --time="${SPLIT_TIME:-24:00:00}" \
  --export=ALL,CT_DIR="$CT_DIR",EXT_ROOT="$EXT_ROOT",IMAGE="$IMAGE" \
  -o "$EXT_ROOT/logs/split_%A_%a.out" -e "$EXT_ROOT/logs/split_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_split.slurm")

RECO5=$(sbatch --parsable --dependency="afterok:$SPLIT" \
  --array="0-$((REGIONS - 1))%${RECO_CONCURRENCY:-10}" \
  --mem="${RECO_MEM:-96G}" --time="${RECO_TIME:-2-00:00:00}" \
  --export=ALL,CT_DIR="$CT_DIR",EXT_ROOT="$EXT_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR" \
  -o "$EXT_ROOT/logs/reco5_%A_%a.out" -e "$EXT_ROOT/logs/reco5_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_reco_region.slurm")

RECOU=$(sbatch --parsable --dependency="afterok:$DIGI" \
  --array="0-$((EVENTS - 1))%${RECO_CONCURRENCY:-10}" \
  --mem="${RECO_MEM:-96G}" --time="${RECO_TIME:-2-00:00:00}" \
  --export=ALL,CT_DIR="$CT_DIR",EXT_ROOT="$EXT_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR" \
  -o "$EXT_ROOT/logs/recou_%A_%a.out" -e "$EXT_ROOT/logs/recou_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_reco_unsplit.slurm")

REPORT=$(sbatch --parsable --dependency="afterok:$RECO5:$RECOU" \
  --export=ALL,CT_DIR="$CT_DIR",EXT_ROOT="$EXT_ROOT" \
  -o "$EXT_ROOT/logs/report_%j.out" -e "$EXT_ROOT/logs/report_%j.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_report.slurm")
printf 'inputs=%s digitization=%s split=%s reco5=%s recou=%s report=%s\n' \
  "$INPUT" "$DIGI" "$SPLIT" "$RECO5" "$RECOU" "$REPORT"
