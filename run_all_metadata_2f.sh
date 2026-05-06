#!/usr/bin/env bash
# Run all selected metadata two-frame datasets locally in GPU-sized waves.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-"$ROOT/results/metadata_2f"}
LOGDIR=${LOGDIR:-"$ROOT/logs/metadata_2f"}
GPUS_STR=${GPUS:-"0 1 2 3"}
MAX_SAMPLES=${MAX_SAMPLES:-}

DATASETS=(
    aria
    dl3dv
    dl3dv_test
    re10k
    scannetpp
    scenenet_depth
    spatialvid_nvs
    tanksandtemples
    vkitti2
    7scenes
    tum
    bonn
)

read -r -a GPUS_ARR <<< "$GPUS_STR"
if [ "${#GPUS_ARR[@]}" -eq 0 ]; then
    echo "GPUS must contain at least one GPU id" >&2
    exit 2
fi

mkdir -p "$OUT" "$LOGDIR"

echo "[run_all_metadata_2f] start $(date)"
for i in "${!DATASETS[@]}"; do
    dataset=${DATASETS[$i]}
    gpu=${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}
    log="$LOGDIR/${dataset}.log"
    echo "[run_all_metadata_2f] launch $dataset on gpu $gpu -> $log"
    if [ -n "$MAX_SAMPLES" ]; then
        bash "$ROOT/run_dataset_metadata_2f.sh" "$dataset" "$gpu" "$MAX_SAMPLES" > "$log" 2>&1 &
    else
        bash "$ROOT/run_dataset_metadata_2f.sh" "$dataset" "$gpu" > "$log" 2>&1 &
    fi

    if [ $(( (i + 1) % ${#GPUS_ARR[@]} )) -eq 0 ]; then
        wait
    fi
done
wait

mamba run -n wan python "$ROOT/collect_results_metadata_2f.py" --root "$OUT"
echo "[run_all_metadata_2f] done $(date)"
