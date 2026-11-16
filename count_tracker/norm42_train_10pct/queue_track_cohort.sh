#!/usr/bin/env bash
set -euo pipefail

for name in CT_DIR SCAN_ROOT MODEL_FULL MODEL_10PCT PAPER1_ROOT GENBIB_DIR \
            GEOMAP IMAGE_V2_11 IMAGE BENCHMARK_DIR; do
  [ -n "${!name:-}" ] || { echo "export $name before queueing" >&2; exit 1; }
done

EVENTS_PER_POINT="$(python3 -c '
import json, os, pathlib
scan = json.loads((pathlib.Path(os.environ["SCAN_ROOT"]) / "scan.json").read_text())
print(scan["events_per_point"])
')"
[[ "$EVENTS_PER_POINT" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid events_per_point: $EVENTS_PER_POINT" >&2
  exit 1
}
mkdir -p "$SCAN_ROOT/logs"

CONDITIONS=$(sbatch --parsable --array=0-0 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT" \
  -o "$SCAN_ROOT/logs/conditions_%A_%a.out" \
  -e "$SCAN_ROOT/logs/conditions_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_conditions.slurm")

SAMPLE=$(sbatch --parsable --dependency="afterok:$CONDITIONS" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",MODEL_FULL="$MODEL_FULL",MODEL_10PCT="$MODEL_10PCT",PAPER1_ROOT="$PAPER1_ROOT",SEED_BASE="${SEED_BASE:-12345}",MAX_ROUNDS="${MAX_ROUNDS:-100}" \
  -o "$SCAN_ROOT/logs/sample_%A_%a.out" \
  -e "$SCAN_ROOT/logs/sample_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_sampling.slurm")

INPUT_MAX=$((EVENTS_PER_POINT * 3 - 1))
INPUT=$(sbatch --parsable --dependency="afterok:$SAMPLE" \
  --array="0-${INPUT_MAX}%${INPUT_CONCURRENCY:-6}" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",EVENTS_PER_POINT="$EVENTS_PER_POINT",GENBIB_DIR="$GENBIB_DIR",GEOMAP="$GEOMAP",IMAGE_V2_11="$IMAGE_V2_11" \
  -o "$SCAN_ROOT/logs/input_%A_%a.out" \
  -e "$SCAN_ROOT/logs/input_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_inputs_parallel.slurm")

DIGI_MAX=$INPUT_MAX
DIGI=$(sbatch --parsable --dependency="afterok:$INPUT" \
  --array="0-${DIGI_MAX}%${DIGI_CONCURRENCY:-6}" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",EVENTS_PER_POINT="$EVENTS_PER_POINT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR" \
  -o "$SCAN_ROOT/logs/digi_%A_%a.out" \
  -e "$SCAN_ROOT/logs/digi_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_digi.slurm")

REGION_MAX=$((EVENTS_PER_POINT * 15 - 1))
SPLIT=$(sbatch --parsable --dependency="afterok:$DIGI" \
  --array="0-${REGION_MAX}%${SPLIT_CONCURRENCY:-10}" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",EVENTS_PER_POINT="$EVENTS_PER_POINT",IMAGE="$IMAGE" \
  -o "$SCAN_ROOT/logs/split_%A_%a.out" \
  -e "$SCAN_ROOT/logs/split_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_split.slurm")

RECO=$(sbatch --parsable --dependency="afterok:$SPLIT" \
  --array="0-${REGION_MAX}%${RECO_CONCURRENCY:-10}" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",EVENTS_PER_POINT="$EVENTS_PER_POINT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR" \
  -o "$SCAN_ROOT/logs/reco_%A_%a.out" \
  -e "$SCAN_ROOT/logs/reco_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_reco.slurm")

printf 'events=%s conditions=%s sampling=%s inputs=%s digitization=%s split=%s reco=%s\n' \
  "$EVENTS_PER_POINT" "$CONDITIONS" "$SAMPLE" "$INPUT" "$DIGI" "$SPLIT" "$RECO"
