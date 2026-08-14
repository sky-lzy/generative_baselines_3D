#!/bin/bash
# Score the 10-scene deep-scale-search pass. Subset semantics: --expect_scenes 10 (a cell is complete
# at 10, not 128) and --allow_partial (the scorer sees only the chosen scenes by design, so the other
# 118 are not missing data). Without both, these cells would be skipped forever.
set -uo pipefail
ROOT=/n/lab_storage/ydu_lab/Lab/akiruga/world_model_4d/standardized_eval_new
PY=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python
R=$ROOT/nvs
OV="run.cell_suffix=__ss10 select.methods.mot13b_full=false preds.fair=$R/preds_ss10 results.fair=$R/results_ss10"
cd "$ROOT" || exit 1
END=$(( $(date +%s) + ${1:-180} * 60 ))
while [ "$(date +%s)" -lt "$END" ]; do
  N=$(ls "$R"/results_ss10/*__ss10.csv 2>/dev/null | wc -l)
  J=$(squeue -u "$USER" -h -o "%j" | grep -c "__ss10" || true)
  echo "[$(date '+%T')] ss10 scored=$N/80  jobs=$J"
  OUT=$(timeout 900 "$PY" nvs/slurm/submit_score.py --overrides $OV \
        --expect_scenes 10 --allow_partial 2>&1)
  echo "$OUT" | grep -E "^score:|score .*__ss10" | head -10
  READY=$(echo "$OUT" | sed -n 's/^score: \([0-9]*\) cell.*/\1/p')
  if [ "$J" -eq 0 ] && [ "${READY:-1}" -eq 0 ]; then
    echo "[$(date '+%T')] ss10 COMPLETE ($N cells)"; "$PY" nvs/report/compare_ss10.py; exit 0
  fi
  sleep 240
done
echo "[$(date '+%T')] ss10 time budget reached"
