#!/usr/bin/env bash
# Resume all events in POINT_TAG, then queue the prepared plan downstream.
set -euo pipefail

for name in CT_DIR EXT_ROOT POINT_TAG MODEL_10PCT PAPER1_ROOT GENBIB_DIR \
            GEOMAP IMAGE_V2_11 IMAGE BENCHMARK_DIR; do
  [ -n "${!name:-}" ] || { echo "export $name before queueing" >&2; exit 1; }
done
PLAN="$EXT_ROOT/plan.json"; test -f "$PLAN"; mkdir -p "$EXT_ROOT/logs"
EVENTS=$(python3 - "$PLAN" "$POINT_TAG" <<'PY'
import json, sys
point = next(row for row in json.load(open(sys.argv[1]))["points"]
             if row["tag"] == sys.argv[2])
print(len(point["events"]))
PY
)
DEPENDENCY_ARGS=()
[ -z "${AFTEROK:-}" ] || DEPENDENCY_ARGS=(--dependency="afterok:$AFTEROK")

SAMPLE=$(sbatch --parsable "${DEPENDENCY_ARGS[@]}" \
  --array="0-$((EVENTS - 1))%${SAMPLE_CONCURRENCY:-2}" \
  --export=ALL,CT_DIR="$CT_DIR",EXT_ROOT="$EXT_ROOT",POINT_TAG="$POINT_TAG",MODEL_10PCT="$MODEL_10PCT",PAPER1_ROOT="$PAPER1_ROOT",SEED_BASE="${SEED_BASE:-12345}",MAX_ROUNDS="${MAX_ROUNDS:-100}" \
  -o "$EXT_ROOT/logs/resume_%A_%a.out" -e "$EXT_ROOT/logs/resume_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/occupancy_extension/submit_resume_sampling.slurm")

DOWNSTREAM=$(AFTEROK="$SAMPLE" \
  bash "$CT_DIR/norm42_train_10pct/occupancy_extension/queue_reuse_downstream.sh")
printf 'sampling=%s %s\n' "$SAMPLE" "$DOWNSTREAM"
