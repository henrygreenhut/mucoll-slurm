#!/bin/bash
# =============================================================================
#  Per-event reco as a SLURM CPU array. Each array task reconstructs one
#  interleaved shard of the cohort's events (SIM + COUNT) via
#  run_count_tracker_event.sh. Resumable: fully-reconstructed events are skipped
#  and any partial event directory is cleared and redone. The geomap must exist
#  already (reuse the smoke geomap) so parallel tasks don't race to build it.
#
#  Chain it after the sampling array so it runs unattended:
#    SAMP=$(sbatch --parsable ... submit_count_tracker_sample.slurm)
#    ... export CONDITIONS/CONSTRUCTION/SPLIT/SIGNAL/COUNT_SAMPLES/GEOMAP/EVENTS_OUT/CT_DIR
#    sbatch --dependency=afterok:$SAMP --array=0-9 submit_count_tracker_reco.slurm
# =============================================================================
#SBATCH --job-name=count_reco
#SBATCH --partition=batch
#SBATCH --exclude=node2336
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -euo pipefail

: "${CT_DIR:?}"; : "${CONDITIONS:?}"; : "${CONSTRUCTION:?}"; : "${SPLIT:?}"
: "${SIGNAL:?}"; : "${COUNT_SAMPLES:?}"
: "${GEOMAP:?}"; : "${EVENTS_OUT:?}"
export IMAGE="${IMAGE:?}" BENCHMARK_DIR="${BENCHMARK_DIR:?}" GENBIB_DIR="${GENBIB_DIR:?}"

NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

# Read identities from the conditions manifest. This supports source-event IDs
# such as training_domain_000123 and avoids reconstructing a guessed filename.
mapfile -t EVENT_IDS < <(python3 -c '
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
        echo "=== reco $eid (signal entry $i) ==="
        bash "$CT_DIR/run_count_tracker_event.sh" \
            --conditions "$CONDITIONS" --construction "$CONSTRUCTION" \
            --split "$SPLIT" --event-id "$eid" \
            --signal "$SIGNAL" --signal-entry "$i" \
            --count-samples "$COUNT_SAMPLES" --geomap "$GEOMAP" \
            --output "$out"
    fi
    i=$((i + NUM_SHARDS))
done
