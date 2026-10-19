#!/bin/bash
# Re-simulate the twelve absent or empty Norm42 MUMINUS cycles. Submit --array=0-11.
#SBATCH --job-name=count_n42_recover
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail

: "${GEN_ROOT:?}"; : "${RECOVERY_ROOT:?}"; : "${IMAGE:?}"; : "${BENCHMARK_DIR:?}"
CYCLES=(100 101 105 106 118 143 144 146 147 148 149 6291)
CYCLE="${CYCLES[${SLURM_ARRAY_TASK_ID:?submit --array=0-11}]}"
INPUT="$GEN_ROOT/MUMINUS/bib_gen_${CYCLE}.edm4hep.root"
DESTINATION="$RECOVERY_ROOT/MUMINUS/bib_sim_${CYCLE}.edm4hep.root"
[ -f "$INPUT" ] || { echo "Missing GEN input: $INPUT" >&2; exit 1; }
mkdir -p "$RECOVERY_ROOT/MUMINUS" "$RECOVERY_ROOT/metadata"

validate_output() {
    apptainer exec --bind /oscar:/oscar "$IMAGE" \
        bash -c 'source /opt/setup_mucoll.sh; python - "$1"' bash "$1" <<'PY'
import ROOT
import sys

path = sys.argv[1]
root = ROOT.TFile.Open(path, "READ")
if not root or root.IsZombie():
    raise SystemExit(f"unreadable ROOT output: {path}")
events = root.Get("events")
if not events or events.GetEntries() != 1:
    raise SystemExit(f"expected one events entry in {path}")
for collection in (
        "VertexBarrelCollection", "VertexEndcapCollection",
        "InnerTrackerBarrelCollection", "InnerTrackerEndcapCollection",
        "OuterTrackerBarrelCollection", "OuterTrackerEndcapCollection"):
    if not events.GetBranch(collection):
        raise SystemExit(f"missing {collection} in {path}")
root.Close()
PY
}

if [ -f "$DESTINATION" ]; then
    validate_output "$DESTINATION"
    echo "cycle $CYCLE already recovered and valid"
    exit 0
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/count-n42-recover-${CYCLE}.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
OUTPUT="$WORK/bib_sim_${CYCLE}.edm4hep.root"
cat > "$WORK/run.sh" <<EOF
set -euo pipefail
source /opt/setup_mucoll.sh
source '$BENCHMARK_DIR/setup_config.sh' '$BENCHMARK_DIR' MAIA_v0
cd '$WORK'
ddsim --steeringFile '$BENCHMARK_DIR/simulation/steer_baseline.py' \\
    --numberOfEvents 1 \\
    --inputFiles '$INPUT' \\
    --outputFile '$OUTPUT'
EOF
apptainer exec --bind /oscar:/oscar "$IMAGE" bash "$WORK/run.sh"
validate_output "$OUTPUT"
mv "$OUTPUT" "$DESTINATION"

cat > "$RECOVERY_ROOT/metadata/cycle_${CYCLE}.txt" <<EOF
cycle=$CYCLE
input=$INPUT
input_sha256=$(sha256sum "$INPUT" | awk '{print $1}')
output=$DESTINATION
output_sha256=$(sha256sum "$DESTINATION" | awk '{print $1}')
image=$IMAGE
benchmark=$BENCHMARK_DIR
benchmark_commit=$(git -C "$BENCHMARK_DIR" rev-parse HEAD)
slurm_job_id=${SLURM_JOB_ID:-unknown}
EOF
echo "Recovered MUMINUS cycle $CYCLE -> $DESTINATION"
