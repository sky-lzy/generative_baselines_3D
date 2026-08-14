#!/bin/bash
# Score the hg=0.5 A/B pass as its cells complete. Writes ONLY into the hg05 tree -- the original
# hg=1.0 preds, CSVs, videos and raw arrays are never read for writing or touched.
#
# The A/B being run: identical scene sets, identical scale grids, identical scorer, one variable
# changed (hist_guidance 1.0 -> 0.5, i.e. conditioning-CFG scale 2.0 -> 1.5). The 10-scene tuning
# grid put hg0.5 ahead by +0.717 dB at 4DiM single-view on all three metrics, and behind at
# two-view, so this pass tests whether that transfers to all 128 scenes in every regime.
set -uo pipefail
ROOT=/n/lab_storage/ydu_lab/Lab/akiruga/world_model_4d/standardized_eval_new
PY=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python
R=$ROOT/nvs
# select.methods.seva=false is LOAD-BEARING. topup_fair enumerates every SELECTED method, so without
# it the top-up "discovers" that 16 SEVA cells are missing from the hg05 tree and resubmits 160
# shards to re-run a baseline that does not even read hist_guidance -- ~45 GPU-h of pure waste.
# It did exactly that once; those jobs were cancelled.
OV="run.hist_guidance=0.5 run.cell_suffix=__hg0.5 select.methods.seva=false preds.fair=$R/preds_fair_hg05 results.fair=$R/results_fair_hg05"
MAXMIN=${1:-180}
cd "$ROOT" || exit 1
END=$(( $(date +%s) + MAXMIN * 60 ))

while [ "$(date +%s)" -lt "$END" ]; do
  N=$(ls "$R"/results_fair_hg05/*.csv 2>/dev/null | wc -l)
  INF=$(squeue -u "$USER" -h -o "%j" | grep -c "hg0.5" || true)
  SCO=$(squeue -u "$USER" -h -o "%j" | grep -c "^fscore_.*hg0.5" || true)
  echo "[$(date '+%T')] hg05 scored=$N/32  inference=$INF  scoring=$SCO"

  timeout 900 "$PY" nvs/slurm/topup_fair.py --overrides $OV 2>&1 | grep -E "resubmitted" | tail -2
  OUT=$(timeout 900 "$PY" nvs/slurm/submit_score.py --overrides $OV 2>&1)
  echo "$OUT" | grep -E "^score:|score .*hg0.5" | head -12
  READY=$(echo "$OUT" | sed -n 's/^score: \([0-9]*\) cell.*/\1/p')

  if [ "$INF" -eq 0 ] && [ "$SCO" -eq 0 ] && [ "${READY:-1}" -eq 0 ]; then
    echo "[$(date '+%T')] hg05 PASS COMPLETE — $N cells scored"
    exit 0
  fi
  sleep 240
done
echo "[$(date '+%T')] hg05 time budget reached"
