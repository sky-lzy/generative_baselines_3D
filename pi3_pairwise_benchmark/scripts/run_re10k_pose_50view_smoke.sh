#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_RE10K="${SOURCE_RE10K:-/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/data/re10k}"
WORK_ROOT="${WORK_ROOT:-/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke_50view/${SLURM_JOB_ID:-local}}"
NUM_SEQUENCES="${NUM_SEQUENCES:-3}"
CANDIDATE_SEQUENCES="${CANDIDATE_SEQUENCES:-$((NUM_SEQUENCES * 4))}"
NUM_FRAMES="${NUM_FRAMES:-50}"
DEVICE="${DEVICE:-cuda}"
PI3_CHECKPOINT="${PI3_CHECKPOINT:-${ROOT}/external/Pi3/checkpoints/Pi3}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/external/Pi3:${PYTHONPATH:-}"
mkdir -p "${WORK_ROOT}/prepared_re10k" "${WORK_ROOT}/subsets" "${WORK_ROOT}/outputs"

run_in_env() {
  local setup="$1"
  shift
  if [[ -n "${setup}" ]]; then
    bash -lc "${setup}; cd '${ROOT}'; export PYTHONPATH='${ROOT}:${ROOT}/external/Pi3':\${PYTHONPATH:-}; $*"
  else
    "$@"
  fi
}

SEQ_MAP="${ROOT}/external/Pi3/datasets/seq-id-maps/Re10K_relpose_seq-id-map_seed42.json"
SEQ_FILE="${ROOT}/external/Pi3/datasets/sequences/re10k_test_1719.txt"
CANDIDATE_MAP="${WORK_ROOT}/subsets/Re10K_contiguous_${NUM_FRAMES}_candidates_n${CANDIDATE_SEQUENCES}.json"
CANDIDATE_FILE="${WORK_ROOT}/subsets/re10k_contiguous_${NUM_FRAMES}_candidates_${CANDIDATE_SEQUENCES}.txt"
SUBSET_MAP="${WORK_ROOT}/subsets/Re10K_contiguous_${NUM_FRAMES}_valid_n${NUM_SEQUENCES}.json"
SUBSET_FILE="${WORK_ROOT}/subsets/re10k_contiguous_${NUM_FRAMES}_valid_${NUM_SEQUENCES}.txt"

python scripts/make_re10k_contiguous_seq_map.py \
  --source-root "${SOURCE_RE10K}" \
  --seq-file "${SEQ_FILE}" \
  --seq-id-map "${SEQ_MAP}" \
  --num-frames "${NUM_FRAMES}" \
  --max-sequences "${CANDIDATE_SEQUENCES}" \
  --output-seq-id-map "${CANDIDATE_MAP}" \
  --output-seq-file "${CANDIDATE_FILE}"

run_in_env "${PREP_ENV_SETUP:-}" python scripts/prepare_re10k_pi3_format_from_vwm.py \
  --source-root "${SOURCE_RE10K}" \
  --output-root "${WORK_ROOT}/prepared_re10k" \
  --seq-id-map "${CANDIDATE_MAP}" \
  --seq-file "${CANDIDATE_FILE}" \
  --summary-json "${WORK_ROOT}/outputs/prepare_summary.json" \
  --allow-failures

python scripts/filter_prepared_re10k_subset.py \
  --prepared-root "${WORK_ROOT}/prepared_re10k" \
  --candidate-seq-id-map "${CANDIDATE_MAP}" \
  --candidate-seq-file "${CANDIDATE_FILE}" \
  --prepare-summary "${WORK_ROOT}/outputs/prepare_summary.json" \
  --num-sequences "${NUM_SEQUENCES}" \
  --output-seq-id-map "${SUBSET_MAP}" \
  --output-seq-file "${SUBSET_FILE}"

run_in_env "${PREP_ENV_SETUP:-}" python scripts/materialize_re10k_samples.py \
  --re10k-root "${WORK_ROOT}/prepared_re10k" \
  --seq-id-map "${SUBSET_MAP}" \
  --seq-file "${SUBSET_FILE}" \
  --output "${WORK_ROOT}/outputs/manifest_${NUM_FRAMES}view.jsonl" \
  --max-sequences "${NUM_SEQUENCES}"

run_in_env "${MODEL_ENV_SETUP:-}" python scripts/run_pi3_10view_repro.py \
  --re10k-dir "${WORK_ROOT}/prepared_re10k" \
  --output-dir "${WORK_ROOT}/outputs/pi3_${NUM_FRAMES}view" \
  --pretrained-model-name-or-path "${PI3_CHECKPOINT}" \
  --device "${DEVICE}" \
  "data.Re10K.cfg.seq_file=${SUBSET_FILE}" \
  "data.Re10K.seq_id_map=${SUBSET_MAP}" \
  "data.Re10K.cfg.cache_file=${WORK_ROOT}/outputs/re10k_cache_${NUM_FRAMES}view_n${NUM_SEQUENCES}.npy" \
  "eval_datasets=[Re10K]"

echo "50-view manifest: ${WORK_ROOT}/outputs/manifest_${NUM_FRAMES}view.jsonl"
echo "Pi3 ${NUM_FRAMES}-view metrics: ${WORK_ROOT}/outputs/pi3_${NUM_FRAMES}view/Re10K-metric.csv"
