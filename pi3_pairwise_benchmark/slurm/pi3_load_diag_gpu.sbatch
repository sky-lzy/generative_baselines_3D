#!/bin/bash
#SBATCH --job-name=pi3-load-diag
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
cd "${ROOT}"
mkdir -p "${ROOT}/outputs/slurm"

if [[ -n "${ENV_SETUP:-}" ]]; then
  eval "${ENV_SETUP}"
fi

export PYTHONPATH="${ROOT}:${ROOT}/external/Pi3:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python scripts/diagnose_pi3_load.py \
  --pi3-root "${ROOT}/external/Pi3" \
  --checkpoint "${PI3_CHECKPOINT:-${ROOT}/external/Pi3/checkpoints/Pi3}" \
  --device "${DEVICE:-cuda}"
