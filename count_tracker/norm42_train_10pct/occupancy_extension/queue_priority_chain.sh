#!/usr/bin/env bash
set -euo pipefail

for name in CT_DIR WAVE1_ROOT LOW_ROOT WAVE3_ROOT HIGH_ROOT; do
  [ -n "${!name:-}" ] || { echo "export $name before queueing" >&2; exit 1; }
done
bash "$CT_DIR/norm42_train_10pct/occupancy_extension/preflight.sh"
extract_report() {
  sed -n 's/.*report=\([0-9][0-9]*\).*/\1/p' <<<"$1"
}

echo "Priority 1: first five paired 10%-BX events"
OUT1=$(SCAN_ROOT="$WAVE1_ROOT" EVENTS_PER_POINT=5 \
  bash "$CT_DIR/norm42_train_10pct/occupancy_extension/queue_fresh_wave.sh")
echo "$OUT1"; REPORT1="$(extract_report "$OUT1")"; test -n "$REPORT1"

echo "Priority 2: reused-SIM low/intermediate occupancy points"
OUT2=$(AFTEROK="$REPORT1" EXT_ROOT="$LOW_ROOT" \
  bash "$CT_DIR/norm42_train_10pct/occupancy_extension/queue_reuse_wave.sh")
echo "$OUT2"; REPORT2="$(extract_report "$OUT2")"; test -n "$REPORT2"

echo "Priority 3: five additional paired 10%-BX events"
OUT3=$(AFTEROK="$REPORT2" SCAN_ROOT="$WAVE3_ROOT" EVENTS_PER_POINT=5 \
  bash "$CT_DIR/norm42_train_10pct/occupancy_extension/queue_fresh_wave.sh")
echo "$OUT3"; REPORT3="$(extract_report "$OUT3")"; test -n "$REPORT3"

echo "Priority 4: 100% and 200% occupancy points"
OUT4=$(AFTEROK="$REPORT3" EXT_ROOT="$HIGH_ROOT" \
  SAMPLE_SHARDS=5 SAMPLE_CONCURRENCY=2 \
  INPUT_CONCURRENCY=1 DIGI_CONCURRENCY=1 SPLIT_CONCURRENCY=2 RECO_CONCURRENCY=5 \
  INPUT_MEM=220G DIGI_MEM=160G SPLIT_MEM=120G RECO_MEM=64G \
  INPUT_TIME=2-00:00:00 DIGI_TIME=2-00:00:00 SPLIT_TIME=2-00:00:00 RECO_TIME=2-00:00:00 \
  bash "$CT_DIR/norm42_train_10pct/occupancy_extension/queue_reuse_wave.sh")
echo "$OUT4"; REPORT4="$(extract_report "$OUT4")"; test -n "$REPORT4"

printf 'priority_reports=%s,%s,%s,%s\n' "$REPORT1" "$REPORT2" "$REPORT3" "$REPORT4"
