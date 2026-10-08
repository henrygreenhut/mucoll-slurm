#!/bin/bash
# =============================================================================
#  Per-event v2.11 SIM/COUNT reco as a SLURM CPU array -- the v2.11 analogue of
#  ../submit_count_tracker_reco.slurm.  Each array task processes one
#  interleaved shard of a split's events through run_count_tracker_event_v2_11.sh
#  (extract SIM BIB -> assign v2.11 CellIDs for SIM and COUNT -> write BIB-only
#  inputs with the v2.11 runtime -> digitize and reconstruct both arms under the
#  old Marlin/ACTS stack).  Resumable: events with both reco outputs are skipped
#  and any partial event directory is cleared and redone.
#
#  This path is BIB-only -- no neutrino signal is injected (it has no tracker
#  hits), so unlike the v3 array there is no SIGNAL/SIGNAL_ENTRY.
#
#  The GEOMAP must already exist so parallel tasks do not race to build it:
#      IMAGE_V2_11=... bash v2_11/build_geomap_v2_11.sh "$GEOMAP"
#
#  Write all splits into one EVENTS_OUT (event IDs are unique across splits) so
#  that ../count_tracker_cohort_report.py --events-root "$EVENTS_OUT" --split ...
#  aggregates them exactly as it does for the v3 cohort.
#
#  Submit one array per split, e.g. for the 20 test events with 10 shards:
#    sbatch --array=0-9 \
#      --export=ALL,CT_DIR=...,GENBIB_DIR=...,CONDITIONS=...,CONSTRUCTION=norm42,\
#SPLIT=test,COUNT_SAMPLES=...,GEOMAP=...,EVENTS_OUT=...,IMAGE_V2_11=... \
#      v2_11/submit_count_tracker_reco_v2_11.slurm
# =============================================================================
#SBATCH --job-name=count_reco_v2_11
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -euo pipefail

: "${CT_DIR:?set CT_DIR to the count_tracker directory}"
: "${GENBIB_DIR:?set GENBIB_DIR to the GenBIB-ML checkout}"
: "${CONDITIONS:?set CONDITIONS to the prepared conditions directory}"
: "${CONSTRUCTION:?set CONSTRUCTION to the conditions construction}"
: "${SPLIT:?set SPLIT to train, val, or test}"
: "${COUNT_SAMPLES:?set COUNT_SAMPLES to the generated arrays root}"
: "${GEOMAP:?set GEOMAP to a prebuilt v2.11 geometry map}"
: "${EVENTS_OUT:?set EVENTS_OUT to the v2.11 per-event output root}"
export IMAGE_V2_11="${IMAGE_V2_11:-/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif}"

[ -f "$GEOMAP" ] || { echo "GEOMAP does not exist; build it once before submitting: $GEOMAP" >&2; exit 1; }

CONDA_MODULE="${CONDA_MODULE:-miniforge3/25.3.0-3-a6hh}"
ASSIGN_ENV="${ASSIGN_ENV:-genbib}"
EXTRACT_ENV="${EXTRACT_ENV:-count-hdf}"

module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
conda activate "$ASSIGN_ENV"
export ASSIGN_PYTHON="$(command -v python)"
conda deactivate
conda activate "$EXTRACT_ENV"
export EXTRACT_PYTHON="$(command -v python)"

NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

# Event identities for this split, in manifest order (same source the v3 array
# uses); the signal-entry column is intentionally ignored for the BIB-only path.
EVENT_IDS=()
while IFS= read -r event_id; do
    EVENT_IDS+=("$event_id")
done < <(python3 -c '
import json, sys
manifest, split = sys.argv[1:]
report = json.load(open(manifest))
for event in report["events"]:
    if event["split"] == split:
        print(event["event_id"])
' "$CONDITIONS/manifest.json" "$SPLIT")

TOTAL="${#EVENT_IDS[@]}"
if [ -n "${NEV:-}" ]; then
    [ "$NEV" -le "$TOTAL" ] || { echo "NEV=$NEV exceeds $TOTAL manifest events" >&2; exit 1; }
    TOTAL="$NEV"
fi
[ "$TOTAL" -gt 0 ] || { echo "No $SPLIT events in conditions manifest" >&2; exit 1; }

i="$SHARD"
while [ "$i" -lt "$TOTAL" ]; do
    eid="${EVENT_IDS[$i]}"
    out="$EVENTS_OUT/$eid"
    if [ -f "$out/SIM/reco/reco_output.edm4hep.root" ] && \
       [ -f "$out/COUNT/reco/reco_output.edm4hep.root" ]; then
        echo "skip $eid (already reconstructed)"
    else
        [ -d "$out" ] && rm -rf "$out"   # clear any partial attempt; fully regenerable
        echo "=== v2.11 reco $eid ==="
        bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" \
            --conditions "$CONDITIONS" --construction "$CONSTRUCTION" \
            --split "$SPLIT" --event-id "$eid" \
            --count-samples "$COUNT_SAMPLES" --geomap "$GEOMAP" \
            --output "$out"
    fi
    i=$((i + NUM_SHARDS))
done
