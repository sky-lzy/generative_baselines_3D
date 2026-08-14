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
  echo "[$(date '+%T')] scored=$NCSV  inference_jobs=$INF  scoring_jobs=$SCO"

  # self-heal first: a cell short of scenes with NO live jobs would otherwise never get a CSV,
  # because --require_complete blocks it forever and nothing else notices.
  timeout 900 "$PY" nvs/slurm/topup_fair.py 2>&1 | grep -E "RESUBMIT|resubmitted" | tail -8

  OUT=$(timeout 900 "$PY" nvs/slurm/submit_score.py 2>&1)
  echo "$OUT" | grep -E "^score:|^  [0-9]+  score|FAIL" | head -20
  READY=$(echo "$OUT" | sed -n 's/^score: \([0-9]*\) cell.*/\1/p')

  # Done when the cluster is idle AND submit_score found nothing further to do. Keyed on the QUEUE
  # rather than a hardcoded cell count, which went stale the moment the scale grids were widened.
  if [ "$INF" -eq 0 ] && [ "$SCO" -eq 0 ] && [ "${READY:-1}" -eq 0 ]; then
    echo "[$(date '+%T')] ALL CELLS SCORED — building report"
    # four benchmarks rendered concurrently -- ~80 videos at ~30s each is 40 minutes serially
    for B in re10k128_50f:2 re10k128_4dim:1 re10k128_4dim:2 re10k128_50f:1; do
      timeout 3000 "$PY" nvs/report/render_all.py --per_bench 20 --only_bench "$B" \
        > "nvs/report/render_${B/:/_}.log" 2>&1 &
    done
    wait
    grep -hcE "^  ok " nvs/report/render_*.log | paste -sd+ | bc | sed 's/^/videos rendered: /' 
    timeout 600 "$PY" nvs/report/build_report.py 2>&1 | tail -3
    echo "[$(date '+%T')] REPORT BUILT"
    exit 0
  fi
  sleep 240
done
echo "[$(date '+%T')] time budget reached; report NOT auto-built"
