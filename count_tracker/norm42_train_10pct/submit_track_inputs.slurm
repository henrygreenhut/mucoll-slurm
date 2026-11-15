#!/usr/bin/env bash
#SBATCH --job-name=count10_track_input
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${SCAN_ROOT:?}"; : "${GENBIB_DIR:?}"; : "${GEOMAP:?}"
POINT="$SCAN_ROOT/n0167"
EVENT_ID="$(python3 "$CT_DIR/v3_occupancy_scan/scan_point.py" \
  --scan "$SCAN_ROOT/scan.json" --index 0 --field event_id --event-index 0)"
OUTPUT="$POINT/inputs/$EVENT_ID"
PARTIAL="$OUTPUT.partial"
if [ -f "$OUTPUT/SIM/input/input.edm4hep.root" ] && \
   [ -f "$OUTPUT/COUNT_FULL/input/input.edm4hep.root" ] && \
   [ -f "$OUTPUT/COUNT_10PCT/input/input.edm4hep.root" ]; then
  echo "$EVENT_ID inputs already exist"
  exit 0
fi
[ ! -e "$OUTPUT" ] || { echo "refusing incomplete output $OUTPUT" >&2; exit 1; }
[ ! -e "$PARTIAL" ] || { echo "refusing stale partial output $PARTIAL" >&2; exit 1; }
mkdir -p "$PARTIAL"

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib
export ASSIGN_PYTHON="$(command -v python)"
conda deactivate
conda activate count-hdf
export EXTRACT_PYTHON="$(command -v python)"

export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"
bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" \
  --conditions "$POINT/conditions" --construction norm42 \
  --split test --event-id "$EVENT_ID" \
  --count-samples "$POINT/count_samples_full" --geomap "$GEOMAP" \
  --stop-after input --output "$PARTIAL/full"

bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" \
  --conditions "$POINT/conditions" --construction norm42 \
  --split test --event-id "$EVENT_ID" \
  --count-samples "$POINT/count_samples_10pct" --geomap "$GEOMAP" \
  --only COUNT --stop-after input --output "$PARTIAL/tenpct"

mkdir "$PARTIAL/published"
mv "$PARTIAL/full/SIM" "$PARTIAL/published/SIM"
mv "$PARTIAL/full/COUNT" "$PARTIAL/published/COUNT_FULL"
mv "$PARTIAL/tenpct/COUNT" "$PARTIAL/published/COUNT_10PCT"
PUBLISHED="$PARTIAL/published" EVENT_ID="$EVENT_ID" \
python3 -c 'import json, os, pathlib; p=pathlib.Path(os.environ["PUBLISHED"]); (p/"three_arm_manifest.json").write_text(json.dumps({"event_id": os.environ["EVENT_ID"], "arms": {"SIM": "empirical source-disjoint SIM", "COUNT_FULL": "existing full-corpus Norm42 model", "COUNT_10PCT": "new 10%-corpus Norm42 model"}}, indent=2)+"\n")'
mv "$PARTIAL/published" "$OUTPUT"
rm -rf -- "$PARTIAL"
echo "prepared three-arm inputs -> $OUTPUT"
