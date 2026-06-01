#!/bin/bash
#SBATCH --job-name=pi3-50view-save
#SBATCH --output=pi3_pairwise_benchmark/outputs/slurm/%x-%j.out
#SBATCH --error=pi3_pairwise_benchmark/outputs/slurm/%x-%j.err
#SBATCH --account=kempner_rcai_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEVICE="${DEVICE:-cuda}"
PI3_CHECKPOINT="${PI3_CHECKPOINT:-${ROOT}/external/Pi3/checkpoints/Pi3}"
MAX_SEQUENCES="${MAX_SEQUENCES:-}"

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
else
  export PATH="/n/home01/rcai/.conda/envs/psamp/bin:${PATH}"
fi

export PYTHONPATH="${ROOT}:${ROOT}/external/Pi3:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

args=(
  scripts/run_pi3_50view_sequences.py
  --manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --pretrained-model-name-or-path "${PI3_CHECKPOINT}"
  --device "${DEVICE}"
)

if [[ -n "${MAX_SEQUENCES}" ]]; then
  args+=(--max-sequences "${MAX_SEQUENCES}")
fi

python "${args[@]}"
