#!/bin/bash
#SBATCH --job-name=geo4d-2view-smoke
#SBATCH --output=pi3_pairwise_benchmark/outputs/slurm/%x-%j.out
#SBATCH --error=pi3_pairwise_benchmark/outputs/slurm/%x-%j.err
#SBATCH --account=kempner_rcai_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GEO4D_ROOT="${GEO4D_ROOT:-${ROOT}/external/Geo4D}"
GEO4D_CKPT="${GEO4D_CKPT:-${GEO4D_ROOT}/checkpoints/geo4d/model.ckpt}"
GEO4D_CONFIG="${GEO4D_CONFIG:-${GEO4D_ROOT}/configs/inference_geo4d.yaml}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
DDIM_STEPS="${DDIM_STEPS:-5}"
STRIDE="${STRIDE:-4}"
MAX_SEQUENCES="${MAX_SEQUENCES:-3}"
MAX_PAIRS_PER_SEQUENCE="${MAX_PAIRS_PER_SEQUENCE:-20}"
PAIR_POLICY="${PAIR_POLICY:-long}"
PAIR_SEED="${PAIR_SEED:-42}"
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
mkdir -p "${ROOT}/outputs/slurm" "${OUTPUT_DIR}" "/tmp/geo4d-mpl-${SLURM_JOB_ID:-manual}"

if [[ -n "${ENV_SETUP:-}" ]]; then
  eval "${ENV_SETUP}"
fi

export PYTHONPATH="${ROOT}:${GEO4D_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/geo4d-mpl-${SLURM_JOB_ID:-manual}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

args=(
  scripts/run_geo4d_2view_pairs.py
  --manifest "${MANIFEST}"
  --geo4d-root "${GEO4D_ROOT}"
  --ckpt-path "${GEO4D_CKPT}"
  --config-path "${GEO4D_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --ddim-steps "${DDIM_STEPS}"
  --stride "${STRIDE}"
  --max-sequences "${MAX_SEQUENCES}"
  --pair-policy "${PAIR_POLICY}"
  --pair-seed "${PAIR_SEED}"
  --long-gap-min "${LONG_GAP_MIN}"
)

if [[ -n "${MAX_PAIRS_PER_SEQUENCE}" ]]; then
  args+=(--max-pairs-per-sequence "${MAX_PAIRS_PER_SEQUENCE}")
fi
if [[ -n "${LONG_GAP_MAX}" ]]; then
  args+=(--long-gap-max "${LONG_GAP_MAX}")
fi

python "${args[@]}"
