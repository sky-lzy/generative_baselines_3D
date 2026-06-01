#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_RE10K="${SOURCE_RE10K:-/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/data/re10k}"
WORK_ROOT="${WORK_ROOT:-/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke}"
NUM_SEQUENCES="${NUM_SEQUENCES:-3}"
CANDIDATE_SEQUENCES="${CANDIDATE_SEQUENCES:-$((NUM_SEQUENCES * 4))}"
DEVICE="${DEVICE:-cuda}"
RUN_OURS="${RUN_OURS:-1}"
OURS_MAX_SEQUENCES="${OURS_MAX_SEQUENCES:-1}"
OURS_CKPT="${OURS_CKPT:-/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt}"
VIDEO_WORLD_MODEL_ROOT="${VIDEO_WORLD_MODEL_ROOT:-/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
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
CANDIDATE_MAP="${WORK_ROOT}/subsets/Re10K_relpose_seq-id-map_seed42_candidates_n${CANDIDATE_SEQUENCES}.json"
CANDIDATE_FILE="${WORK_ROOT}/subsets/re10k_test_candidates_${CANDIDATE_SEQUENCES}.txt"
SUBSET_MAP="${WORK_ROOT}/subsets/Re10K_relpose_seq-id-map_seed42_valid_n${NUM_SEQUENCES}.json"
SUBSET_FILE="${WORK_ROOT}/subsets/re10k_test_valid_${NUM_SEQUENCES}.txt"

python scripts/make_re10k_subset.py \
  --seq-id-map "${SEQ_MAP}" \
  --seq-file "${SEQ_FILE}" \
  --num-sequences "${CANDIDATE_SEQUENCES}" \
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
  --output "${WORK_ROOT}/outputs/manifest.jsonl" \
  --max-sequences "${NUM_SEQUENCES}"

run_in_env "${MODEL_ENV_SETUP:-}" python scripts/run_pi3_10view_repro.py \
  --re10k-dir "${WORK_ROOT}/prepared_re10k" \
  --output-dir "${WORK_ROOT}/outputs/pi3_10view" \
  --pretrained-model-name-or-path "${PI3_CHECKPOINT}" \
  --device "${DEVICE}" \
  "data.Re10K.cfg.seq_file=${SUBSET_FILE}" \
  "data.Re10K.seq_id_map=${SUBSET_MAP}" \
  "data.Re10K.cfg.cache_file=${WORK_ROOT}/outputs/re10k_cache_n${NUM_SEQUENCES}.npy" \
  "eval_datasets=[Re10K]"

run_in_env "${MODEL_ENV_SETUP:-}" python scripts/run_pi3_2view_pairs.py \
  --manifest "${WORK_ROOT}/outputs/manifest.jsonl" \
  --output-dir "${WORK_ROOT}/outputs/pi3_2view" \
  --pretrained-model-name-or-path "${PI3_CHECKPOINT}" \
  --device "${DEVICE}" \
  --max-sequences "${NUM_SEQUENCES}"

if [[ "${RUN_OURS}" == "1" ]]; then
  run_in_env "${MODEL_ENV_SETUP:-}" python scripts/run_ours_2view_pairs.py \
    --manifest "${WORK_ROOT}/outputs/manifest.jsonl" \
    --video-world-model-root "${VIDEO_WORLD_MODEL_ROOT}" \
    --ckpt-path "${OURS_CKPT}" \
    --output-dir "${WORK_ROOT}/outputs/ours_2view" \
    --device "${DEVICE}" \
    --sample-steps "${SAMPLE_STEPS}" \
    --max-sequences "${OURS_MAX_SEQUENCES}"
fi
