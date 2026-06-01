#!/bin/bash
#SBATCH --job-name=vwm-import-diag
#SBATCH --output=pi3_pairwise_benchmark/outputs/slurm/%x-%j.out
#SBATCH --error=pi3_pairwise_benchmark/outputs/slurm/%x-%j.err
#SBATCH --account=kempner_rcai_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VIDEO_WORLD_MODEL_ROOT="${VIDEO_WORLD_MODEL_ROOT:-/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model}"

cd "${ROOT}"
mkdir -p "${ROOT}/outputs/slurm"

if [[ -n "${ENV_SETUP:-}" ]]; then
  eval "${ENV_SETUP}"
fi

export PYTHONPATH="${ROOT}:${ROOT}/external/Pi3:${VIDEO_WORLD_MODEL_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONFAULTHANDLER=1

python scripts/diagnose_vwm_import.py \
  --video-world-model-root "${VIDEO_WORLD_MODEL_ROOT}" \
  --dump-after "${DUMP_AFTER:-60}"
