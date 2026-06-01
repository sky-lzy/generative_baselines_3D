#!/bin/bash
#SBATCH --job-name=pi3-2view-existing
#SBATCH --output=pi3_pairwise_benchmark/outputs/slurm/%x-%j.out
#SBATCH --error=pi3_pairwise_benchmark/outputs/slurm/%x-%j.err
#SBATCH --account=kempner_rcai_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEVICE="${DEVICE:-cuda}"
PI3_CHECKPOINT="${PI3_CHECKPOINT:-${ROOT}/external/Pi3/checkpoints/Pi3}"
MAX_SEQUENCES="${MAX_SEQUENCES:-}"
PAIR_POLICY="${PAIR_POLICY:-long}"
PAIRS_PER_SEQUENCE="${PAIRS_PER_SEQUENCE:-20}"
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

export PYTHONPATH="${ROOT}:${ROOT}/external/Pi3:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

args=(
  scripts/run_pi3_2view_pairs.py
  --manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --pretrained-model-name-or-path "${PI3_CHECKPOINT}"
  --device "${DEVICE}"
  --pair-policy "${PAIR_POLICY}"
  --pairs-per-sequence "${PAIRS_PER_SEQUENCE}"
  --pair-seed "${PAIR_SEED}"
  --short-gap-min "${SHORT_GAP_MIN}"
  --short-gap-max "${SHORT_GAP_MAX}"
  --medium-gap-min "${MEDIUM_GAP_MIN}"
  --medium-gap-max "${MEDIUM_GAP_MAX}"
  --long-gap-min "${LONG_GAP_MIN}"
)

if [[ -n "${MAX_SEQUENCES}" ]]; then
  args+=(--max-sequences "${MAX_SEQUENCES}")
fi
if [[ -n "${LONG_GAP_MAX}" ]]; then
  args+=(--long-gap-max "${LONG_GAP_MAX}")
fi

python "${args[@]}"
