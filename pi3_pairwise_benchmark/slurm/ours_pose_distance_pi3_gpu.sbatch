#!/bin/bash
#SBATCH --job-name=ours-pi3-dist
#SBATCH --output=pi3_pairwise_benchmark/outputs/slurm/%x-%j.out
#SBATCH --error=pi3_pairwise_benchmark/outputs/slurm/%x-%j.err
#SBATCH --account=kempner_rcai_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VIDEO_WORLD_MODEL_ROOT="${VIDEO_WORLD_MODEL_ROOT:-/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model}"
OURS_CKPT="${OURS_CKPT:-/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt}"
PI3_ROOT="${PI3_ROOT:-${ROOT}/external/Pi3}"
DEVICE="${DEVICE:-cuda}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
POSE_EVAL_STRIDE="${POSE_EVAL_STRIDE:-1}"
START_INDEX="${START_INDEX:-0}"
NUM_FRAMES="${NUM_FRAMES:-}"
MAX_SEQUENCES="${MAX_SEQUENCES:-1}"
OVERRIDE_N_FRAMES="${OVERRIDE_N_FRAMES:-auto}"
OVERRIDE_HEIGHT="${OVERRIDE_HEIGHT:-}"
OVERRIDE_WIDTH="${OVERRIDE_WIDTH:-}"
IMAGE_PREPROCESS_MODE="${IMAGE_PREPROCESS_MODE:-direct_resize}"
SEQUENCES="${SEQUENCES:-}"

if [[ -z "${DATASET:-}" ]]; then
  echo "DATASET must be set to sintel or tum" >&2
  exit 2
fi
if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "DATA_ROOT must be set" >&2
  exit 2
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR must be set" >&2
  exit 2
fi

cd "${ROOT}"
mkdir -p "${ROOT}/outputs/slurm" "${OUTPUT_DIR}/${DATASET}"

if [[ -n "${ENV_SETUP:-}" ]]; then
  eval "${ENV_SETUP}"
else
  module load python
  mamba activate psamp
  module load cuda/12.4.1-fasrc01
  module load gcc/9.5.0-fasrc01
fi

export PYTHONPATH="${ROOT}:${PI3_ROOT}:${VIDEO_WORLD_MODEL_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

extra_args=()
if [[ -n "${SEQUENCES}" ]]; then
  extra_args+=(--sequences "${SEQUENCES}")
fi
if [[ -n "${NUM_FRAMES}" ]]; then
  extra_args+=(--num-frames "${NUM_FRAMES}")
fi
if [[ -n "${OVERRIDE_HEIGHT}" ]]; then
  extra_args+=(--override-height "${OVERRIDE_HEIGHT}")
fi
if [[ -n "${OVERRIDE_WIDTH}" ]]; then
  extra_args+=(--override-width "${OVERRIDE_WIDTH}")
fi

python scripts/run_ours_pose_distance_pi3.py \
  --dataset "${DATASET}" \
  --data-root "${DATA_ROOT}" \
  --pi3-root "${PI3_ROOT}" \
  --video-world-model-root "${VIDEO_WORLD_MODEL_ROOT}" \
  --ckpt-path "${OURS_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --sample-steps "${SAMPLE_STEPS}" \
  --pose-eval-stride "${POSE_EVAL_STRIDE}" \
  --start-index "${START_INDEX}" \
  --max-sequences "${MAX_SEQUENCES}" \
  --override-n-frames "${OVERRIDE_N_FRAMES}" \
  --image-preprocess-mode "${IMAGE_PREPROCESS_MODE}" \
  "${extra_args[@]}"
