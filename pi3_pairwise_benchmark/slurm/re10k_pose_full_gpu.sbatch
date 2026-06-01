#!/bin/bash
#SBATCH --job-name=pi3-re10k-full
#SBATCH --output=pi3_pairwise_benchmark/outputs/slurm/%x-%j.out
#SBATCH --error=pi3_pairwise_benchmark/outputs/slurm/%x-%j.err
#SBATCH --account=kempner_rcai_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${ROOT}/outputs/slurm"
cd "${ROOT}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -n "${ENV_SETUP:-}" ]]; then
  eval "${ENV_SETUP}"
fi

bash scripts/run_re10k_pose_full.sh
