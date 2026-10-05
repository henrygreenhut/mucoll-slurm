#!/bin/bash
#SBATCH --job-name=count_analysis
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

: "${BASE:?Set BASE to the completed cohort directory}"
: "${CT_DIR:?Set CT_DIR to mucoll-slurm/count_tracker}"
: "${IMAGE:?Set IMAGE to the v3 detector container}"
: "${SAMPLING_JOBS:?Set SAMPLING_JOBS to comma-separated SPLIT=JOB_ID values}"

CONDITIONS="${CONDITIONS:-$BASE/conditions}"
EVENTS_ROOT="${EVENTS_ROOT:-$BASE/events}"
ANALYSIS="${ANALYSIS:-$BASE/analysis}"
CONSTRUCTION="${CONSTRUCTION:-norm42}"
TITLE="${TITLE:-Direct norm42: SIM vs COUNT}"
EVENTS_PER_SAMPLE_TASK="${EVENTS_PER_SAMPLE_TASK:-5}"

if [[ -e "$ANALYSIS" ]]; then
    echo "Refusing to replace existing analysis directory: $ANALYSIS" >&2
    exit 1
fi
mkdir -p "$ANALYSIS/reports" "$ANALYSIS/track_stores" "$ANALYSIS/plots"

IFS=',' read -r -a sampling_jobs <<< "$SAMPLING_JOBS"
sampling_args=()
for job in "${sampling_jobs[@]}"; do
    sampling_args+=(--job "$job")
done
python3 "$CT_DIR/count_tracker_sampling_time.py" \
    "${sampling_args[@]}" \
    --events-per-task "$EVENTS_PER_SAMPLE_TASK" \
    --output "$ANALYSIS/reports/sampling_time.json"

for split in all train val test; do
    apptainer exec --bind /oscar:/oscar "$IMAGE" \
        python "$CT_DIR/count_tracker_cohort_report.py" \
        --conditions "$CONDITIONS" --events-root "$EVENTS_ROOT" \
        --split "$split" --output "$ANALYSIS/reports/cohort_${split}.json"

    for sample in SIM COUNT; do
        apptainer exec --bind /oscar:/oscar "$IMAGE" \
            python "$CT_DIR/count_tracker_features.py" \
            --conditions "$CONDITIONS" --events-root "$EVENTS_ROOT" \
            --sample "$sample" --split "$split" \
            --output "$ANALYSIS/track_stores/${CONSTRUCTION}_${sample}_${split}"
    done

    apptainer exec --bind /oscar:/oscar "$IMAGE" \
        python "$CT_DIR/plot_count_tracker_track_features.py" \
        --store-dir "$ANALYSIS/track_stores" \
        --construction "$CONSTRUCTION" --split "$split" \
        --output-dir "$ANALYSIS/plots/$split" \
        --title "$TITLE ($split)"
done

echo "Analysis complete: $ANALYSIS"
