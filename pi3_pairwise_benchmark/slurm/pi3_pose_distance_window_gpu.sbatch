#!/bin/bash
#SBATCH --job-name=pi3-tum-window
#SBATCH --output=pi3_pairwise_benchmark/outputs/slurm/%x-%j.out
#SBATCH --error=pi3_pairwise_benchmark/outputs/slurm/%x-%j.err
#SBATCH --account=kempner_rcai_lab
#SBATCH --partition=kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PI3_ROOT="${PI3_ROOT:-${ROOT}/external/Pi3}"
PI3_CHECKPOINT="${PI3_CHECKPOINT:-${PI3_ROOT}/checkpoints/Pi3}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "${TUM_ROOT:-}" ]]; then
  echo "TUM_ROOT must be set" >&2
  exit 2
fi
if [[ -z "${SEQUENCE:-}" ]]; then
  echo "SEQUENCE must be set" >&2
  exit 2
fi
if [[ -z "${START_INDEX:-}" ]]; then
  echo "START_INDEX must be set" >&2
  exit 2
fi
if [[ -z "${NUM_FRAMES:-}" ]]; then
  echo "NUM_FRAMES must be set" >&2
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

export PYTHONPATH="${ROOT}:${PI3_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

python scripts/run_pi3_pose_distance_window.py \
  --tum-root "${TUM_ROOT}" \
  --sequence "${SEQUENCE}" \
  --start-index "${START_INDEX}" \
  --num-frames "${NUM_FRAMES}" \
  --pi3-root "${PI3_ROOT}" \
  --checkpoint "${PI3_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"
