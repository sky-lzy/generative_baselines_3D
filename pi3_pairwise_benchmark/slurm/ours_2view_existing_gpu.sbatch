#!/bin/bash
#SBATCH --job-name=ours-2view-smoke
#SBATCH --output=pi3_pairwise_benchmark/outputs/slurm/%x-%j.out
#SBATCH --error=pi3_pairwise_benchmark/outputs/slurm/%x-%j.err
#SBATCH --account=kempner_rcai_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VIDEO_WORLD_MODEL_ROOT="${VIDEO_WORLD_MODEL_ROOT:-/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model}"
OURS_CKPT="${OURS_CKPT:-/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt}"
DEVICE="${DEVICE:-cuda}"
SAMPLE_STEPS="${SAMPLE_STEPS:-4}"
MAX_SEQUENCES="${MAX_SEQUENCES:-1}"
MAX_PAIRS_PER_SEQUENCE="${MAX_PAIRS_PER_SEQUENCE:-}"
PAIR_POLICY="${PAIR_POLICY:-all}"
PAIR_SEED="${PAIR_SEED:-42}"
SHORT_GAP_MIN="${SHORT_GAP_MIN:-1}"
SHORT_GAP_MAX="${SHORT_GAP_MAX:-5}"
MEDIUM_GAP_MIN="${MEDIUM_GAP_MIN:-10}"
MEDIUM_GAP_MAX="${MEDIUM_GAP_MAX:-20}"
LONG_GAP_MIN="${LONG_GAP_MIN:-30}"
LONG_GAP_MAX="${LONG_GAP_MAX:-}"

if [[ -z "${MANIFEST:-}" ]]; then
  echo "MANIFEST must be set" >&2
  exit 2
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR must be set" >&2
  exit 2
fi

cd "${ROOT}"
mkdir -p "${ROOT}/outputs/slurm" "${OUTPUT_DIR}"

if [[ -n "${ENV_SETUP:-}" ]]; then
  eval "${ENV_SETUP}"
fi

export PYTHONPATH="${ROOT}:${ROOT}/external/Pi3:${VIDEO_WORLD_MODEL_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

args=(
  scripts/run_ours_2view_pairs.py
  --manifest "${MANIFEST}" \
  --video-world-model-root "${VIDEO_WORLD_MODEL_ROOT}" \
  --ckpt-path "${OURS_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --sample-steps "${SAMPLE_STEPS}" \
  --max-sequences "${MAX_SEQUENCES}" \
  --pair-policy "${PAIR_POLICY}" \
  --pair-seed "${PAIR_SEED}" \
  --short-gap-min "${SHORT_GAP_MIN}" \
  --short-gap-max "${SHORT_GAP_MAX}" \
  --medium-gap-min "${MEDIUM_GAP_MIN}" \
  --medium-gap-max "${MEDIUM_GAP_MAX}" \
  --long-gap-min "${LONG_GAP_MIN}"
)

if [[ -n "${MAX_PAIRS_PER_SEQUENCE}" ]]; then
  args+=(--max-pairs-per-sequence "${MAX_PAIRS_PER_SEQUENCE}")
fi
if [[ -n "${LONG_GAP_MAX}" ]]; then
  args+=(--long-gap-max "${LONG_GAP_MAX}")
fi

python "${args[@]}"
