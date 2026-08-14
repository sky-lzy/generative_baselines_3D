#!/bin/bash
# Drive the tail of the fair-NVS pipeline: score cells as they complete, then build the report.
# Submits ONLY scoring jobs (never inference), so it cannot amplify a mistake in the fan-out.
# Idempotent: submit_score skips cells that already have a CSV and cells still generating.
#
#   nvs/slurm/drive_tail.sh [max_minutes]
set -uo pipefail
ROOT=/n/lab_storage/ydu_lab/Lab/akiruga/world_model_4d/standardized_eval_new
PY=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python
MAXMIN=${1:-210}
cd "$ROOT" || exit 1
END=$(( $(date +%s) + MAXMIN * 60 ))

while [ "$(date +%s)" -lt "$END" ]; do
  NCSV=$(ls nvs/results_fair/*.csv 2>/dev/null | grep -vc TUNE || true)
  INF=$(squeue -u "$USER" -h -o "%j" | grep -c "^fnvs_" || true)
  SCO=$(squeue -u "$USER" -h -o "%j" | grep -c "^fscore_" || true)
  echo "[$(date '+%T')] scored=$NCSV/36  inference_jobs=$INF  scoring_jobs=$SCO"

  timeout 900 "$PY" nvs/slurm/submit_score.py 2>&1 | grep -E "^score:|^  [0-9]+  score|FAIL" | head -20

  # done when every cell has a CSV and nothing is left in the queue
  if [ "$NCSV" -ge 36 ] && [ "$INF" -eq 0 ] && [ "$SCO" -eq 0 ]; then
    echo "[$(date '+%T')] ALL CELLS SCORED — building report"
    timeout 3000 "$PY" nvs/report/render_all.py --per_bench 20 2>&1 | tail -25
    timeout 600 "$PY" nvs/report/build_report.py 2>&1 | tail -3
    echo "[$(date '+%T')] REPORT BUILT"
    exit 0
  fi
  sleep 240
done
echo "[$(date '+%T')] time budget reached; report NOT auto-built"
