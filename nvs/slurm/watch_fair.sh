#!/bin/bash
# Health tick for the fair-NVS fleet. REPORT ONLY — never submits, never cancels, never deletes.
# The agent reads this and decides. Run it bare: nvs/slurm/watch_fair.sh
set -uo pipefail
ROOT=/n/lab_storage/ydu_lab/Lab/akiruga/world_model_4d/standardized_eval_new/nvs
PRED=$ROOT/preds_fair
LOGS=$ROOT/slurm/logs
SCENES=/n/netscratch/ydu_lab/Lab/akiruga/vwm_eval_data/svc_bench/scenes_fair

date "+=== tick %F %T"

# ---- queue -------------------------------------------------------------------------------
squeue -u "$USER" -h -o "%T %P" | sort | uniq -c | awk '{printf "queue : %-4s %s %s\n",$1,$2,$3}'
R=$(squeue -u "$USER" -h -t RUNNING -o "%j" | grep -c fnvs_ || true)
P=$(squeue -u "$USER" -h -t PENDING -o "%j" | grep -c fnvs_ || true)
echo "fleet : $R running, $P pending"

# oldest pending reason — the usual cause of a stalled fan-out
squeue -u "$USER" -h -t PENDING -o "%R" | sort | uniq -c | sort -rn | head -3 | sed 's/^/pend  : /'

# ---- progress per cell -------------------------------------------------------------------
echo "cells :"
for d in "$PRED"/*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  case "$n" in
    *seva*|TUNE__seva*) c=$(find "$d" -name "*.png" -path "*samples-rgb*" 2>/dev/null | \
                            awk -F/ '{print $(NF-2)}' | sort -u | wc -l) ;;
    *)                  c=$(find "$d" -name rgb_metrics.json 2>/dev/null | wc -l) ;;
  esac
  printf "        %-58s %4s scenes\n" "$n" "$c"
done

# ---- failures ----------------------------------------------------------------------------
echo "errors:"
bad=0
for f in "$LOGS"/*.out; do
  [ -f "$f" ] || continue
  # only complain about logs that ENDED badly; a live log with no EXIT line is just running
  if grep -q "EXIT [1-9]" "$f" 2>/dev/null; then
    bad=$((bad+1))
    echo "  FAIL $(basename "$f")"
    grep -aE "Error|Traceback|CUDA out of memory|No such file|assert" "$f" | tail -2 | cut -c1-160 | sed 's/^/       /'
  fi
done
[ "$bad" -eq 0 ] && echo "  none"

# ---- preemption (requeue partitions) -----------------------------------------------------
PRE=$(sacct -u "$USER" -S "$(date -d '6 hours ago' +%F-%H:%M)" -X -n -o State 2>/dev/null | \
      grep -cE "PREEMPTED|REQUEUE|NODE_FAIL" || true)
echo "preempt: $PRE job(s) preempted/requeued in the last 6h"
