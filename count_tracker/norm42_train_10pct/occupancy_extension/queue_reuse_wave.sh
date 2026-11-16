#!/usr/bin/env bash
set -euo pipefail

for name in CT_DIR EXT_ROOT MODEL_10PCT PAPER1_ROOT GENBIB_DIR GEOMAP \
            IMAGE_V2_11 IMAGE BENCHMARK_DIR; do
  [ -n "${!name:-}" ] || { echo "export $name before queueing" >&2; exit 1; }
done
PLAN="$EXT_ROOT/plan.json"; test -f "$PLAN"; mkdir -p "$EXT_ROOT/logs"
HELPER="$CT_DIR/norm42_train_10pct/occupancy_extension/plan_task.py"
POINTS=$(python3 "$HELPER" --plan "$PLAN" --stage point --index 0 --count)
EVENTS=$(python3 "$HELPER" --plan "$PLAN" --stage event --index 0 --count)
REGIONS=$(python3 "$HELPER" --plan "$PLAN" --stage region --index 0 --count)
SAMPLE_SHARDS="${SAMPLE_SHARDS:-1}"
(( SAMPLE_SHARDS > 0 )) || { echo "SAMPLE_SHARDS must be positive" >&2; exit 2; }
SAMPLE_TASKS=$((POINTS * SAMPLE_SHARDS))
DEPENDENCY_ARGS=()
[ -z "${AFTEROK:-}" ] || DEPENDENCY_ARGS=(--dependency="afterok:$AFTEROK")

CONDITIONS=$(sbatch --parsable "${DEPENDENCY_ARGS[@]}" \
  --array="0-$((POINTS - 1))%${CONDITION_CONCURRENCY:-5}" \
  --mem="${CONDITION_MEM:-64G}" --time="${CONDITION_TIME:-2-00:00:00}" \
  --export=ALL,CT_DIR="$CT_DIR",EXT_ROOT="$EXT_ROOT" \
  -o "$EXT_ROOT/logs/conditions_%A_%a.out" -e "$EXT_ROOT/logs/conditions_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_conditions.slurm")

SAMPLE=$(sbatch --parsable --dependency="afterok:$CONDITIONS" \
  --array="0-$((SAMPLE_TASKS - 1))%${SAMPLE_CONCURRENCY:-2}" \
  --export=ALL,CT_DIR="$CT_DIR",EXT_ROOT="$EXT_ROOT",MODEL_10PCT="$MODEL_10PCT",PAPER1_ROOT="$PAPER1_ROOT",SEED_BASE="${SEED_BASE:-12345}",MAX_ROUNDS="${MAX_ROUNDS:-100}",SHARDS_PER_POINT="$SAMPLE_SHARDS" \
  -o "$EXT_ROOT/logs/sample_%A_%a.out" -e "$EXT_ROOT/logs/sample_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_sampling.slurm")

INPUT=$(sbatch --parsable --dependency="afterok:$SAMPLE" \
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
printf 'conditions=%s sampling=%s inputs=%s digitization=%s split=%s reco5=%s recou=%s report=%s\n' \
  "$CONDITIONS" "$SAMPLE" "$INPUT" "$DIGI" "$SPLIT" "$RECO5" "$RECOU" "$REPORT"
