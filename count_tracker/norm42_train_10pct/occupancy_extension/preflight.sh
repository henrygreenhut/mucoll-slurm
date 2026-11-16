#!/usr/bin/env bash
set -euo pipefail

for name in CT_DIR STUDY_ROOT WAVE1_ROOT LOW_ROOT WAVE3_ROOT HIGH_ROOT \
            MODEL_10PCT PAPER1_ROOT GENBIB_DIR GEOMAP IMAGE_V2_11 IMAGE \
            BENCHMARK_DIR; do
  [ -n "${!name:-}" ] || { echo "preflight: export $name" >&2; exit 1; }
done
for command in sbatch squeue python3; do
  command -v "$command" >/dev/null || {
    echo "preflight: missing command: $command" >&2; exit 1;
  }
done

required_files=(
  "$WAVE1_ROOT/scan.json"
  "$LOW_ROOT/plan.json"
  "$WAVE3_ROOT/scan.json"
  "$HIGH_ROOT/plan.json"
  "$PAPER1_ROOT/diffusion/tabddpm_official/scripts/sample.py"
  "$GEOMAP"
  "$IMAGE_V2_11"
  "$IMAGE"
  "$BENCHMARK_DIR/setup_config.sh"
  "$GENBIB_DIR/reco/assign_actual_cellid.py"
)
for path in "${required_files[@]}"; do
  [ -f "$path" ] || { echo "preflight: missing file: $path" >&2; exit 1; }
done

model_count=$(find "$MODEL_10PCT" -mindepth 1 -maxdepth 1 -type d \
  -name '*_TABDDPM_*' | wc -l)
[ "$model_count" -eq 6 ] || {
  echo "preflight: expected six model directories, found $model_count" >&2
  exit 1
}

completing_gpu_jobs=$(squeue -h -u "$USER" -p gpu -t CG -o '%i' | xargs)
if [ -n "$completing_gpu_jobs" ]; then
  echo "preflight: warning: GPU job(s) stuck/completing: $completing_gpu_jobs" >&2
  echo "preflight: these may temporarily consume a GPU allocation" >&2
fi

echo "preflight: runtime paths, six models, and plans passed"
