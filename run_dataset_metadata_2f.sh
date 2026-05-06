#!/usr/bin/env bash
# Usage: ./run_dataset_metadata_2f.sh <dataset> <gpu_id> [max_samples]
#
# First/last-frame metadata evaluation for pose_depth baselines only.

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <dataset> <gpu_id> [max_samples]" >&2
    exit 2
fi

DATASET=$1
GPU=$2
MAX_SAMPLES=${3:-}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-"$ROOT/results/metadata_2f"}
INPUT_DIR="$OUT/$DATASET/inputs_pose_depth_2f"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=${WANDB_MODE:-offline}
export GEO4D_DIR=${GEO4D_DIR:-/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/Geo4D}
export GEO4D_CKPT=${GEO4D_CKPT:-"$ROOT/geo4d_ckpt/model.ckpt"}
export GEO4D_CONFIG=${GEO4D_CONFIG:-"$GEO4D_DIR/configs/inference_geo4d.yaml"}
export RAYDIFF_DIR=${RAYDIFF_DIR:-/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/generative_baselines/RayDiffusion/models/co3d_diffusion}
export RAYDIFFUSION_DIR=${RAYDIFFUSION_DIR:-"$ROOT/RayDiffusion"}

mkdir -p "$OUT/$DATASET"

run_py() {
    echo "[$DATASET gpu$GPU] mamba run -n wan python $*"
    mamba run -n wan python "$@"
}

MAX_ARGS=()
if [ -n "$MAX_SAMPLES" ]; then
    MAX_ARGS=(--max_samples "$MAX_SAMPLES")
fi

if [ "$DATASET" = "aria" ]; then
    RAY_DATASET=aria
else
    RAY_DATASET=re10k
fi

echo "[$DATASET gpu$GPU] start $(date)"

run_py "$ROOT/materialize_metadata_2f.py" \
    --dataset "$DATASET" \
    --results_root "$OUT" \
    "${MAX_ARGS[@]}"

run_py "$ROOT/geo4d/eval_geo4d_pose_v3.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUT/$DATASET/geo4d_pose" \
    --ckpt_path "$GEO4D_CKPT" \
    --config "$GEO4D_CONFIG" \
    --custom_run_name "metadata_2f_${DATASET}_geo4d_pose" \
    "${MAX_ARGS[@]}"

run_py "$ROOT/geo4d/eval_geo4d_depth_v3.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUT/$DATASET/geo4d_depth" \
    --ckpt_path "$GEO4D_CKPT" \
    --config "$GEO4D_CONFIG" \
    --dataset "$DATASET" \
    --custom_run_name "metadata_2f_${DATASET}_geo4d_depth" \
    "${MAX_ARGS[@]}"

run_py "$ROOT/RayDiffusion/eval_raydiffusion_pose_v3.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUT/$DATASET/raydiffusion_pose" \
    --model_dir "$RAYDIFF_DIR" \
    --max_frames 2 \
    --dataset "$RAY_DATASET" \
    --custom_run_name "metadata_2f_${DATASET}_raydiffusion_pose" \
    "${MAX_ARGS[@]}"

echo "[$DATASET gpu$GPU] done $(date)"
