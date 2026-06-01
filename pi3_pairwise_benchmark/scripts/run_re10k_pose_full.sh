#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_RE10K="${SOURCE_RE10K:-/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/video_world_model/data/re10k}"
WORK_ROOT="${WORK_ROOT:-/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/full}"
DEVICE="${DEVICE:-cuda}"
RUN_OURS="${RUN_OURS:-1}"
OURS_CKPT="${OURS_CKPT:-/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model/outputs/2026-05-04/12-34-53/checkpoints/last.ckpt}"
VIDEO_WORLD_MODEL_ROOT="${VIDEO_WORLD_MODEL_ROOT:-/n/holylfs05/LABS/rcai_lab/Lab/video_model/test-time/video_world_model}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/external/Pi3:${PYTHONPATH:-}"
mkdir -p "${WORK_ROOT}/prepared_re10k" "${WORK_ROOT}/outputs"

SEQ_MAP="${ROOT}/external/Pi3/datasets/seq-id-maps/Re10K_relpose_seq-id-map_seed42.json"
SEQ_FILE="${ROOT}/external/Pi3/datasets/sequences/re10k_test_1719.txt"

python scripts/prepare_re10k_pi3_format_from_vwm.py \
  --source-root "${SOURCE_RE10K}" \
  --output-root "${WORK_ROOT}/prepared_re10k" \
  --seq-id-map "${SEQ_MAP}" \
  --seq-file "${SEQ_FILE}" \
  --summary-json "${WORK_ROOT}/outputs/prepare_summary.json"

python scripts/materialize_re10k_samples.py \
  --re10k-root "${WORK_ROOT}/prepared_re10k" \
  --seq-id-map "${SEQ_MAP}" \
  --output "${WORK_ROOT}/outputs/manifest.jsonl"

python scripts/run_pi3_10view_repro.py \
  --re10k-dir "${WORK_ROOT}/prepared_re10k" \
  --output-dir "${WORK_ROOT}/outputs/pi3_10view" \
  --device "${DEVICE}" \
  "data.Re10K.cfg.cache_file=${WORK_ROOT}/outputs/re10k_cache.npy" \
  "eval_datasets=[Re10K]"

python scripts/run_pi3_2view_pairs.py \
  --manifest "${WORK_ROOT}/outputs/manifest.jsonl" \
  --output-dir "${WORK_ROOT}/outputs/pi3_2view" \
  --device "${DEVICE}"

if [[ "${RUN_OURS}" == "1" ]]; then
  python scripts/run_ours_2view_pairs.py \
    --manifest "${WORK_ROOT}/outputs/manifest.jsonl" \
    --video-world-model-root "${VIDEO_WORLD_MODEL_ROOT}" \
    --ckpt-path "${OURS_CKPT}" \
    --output-dir "${WORK_ROOT}/outputs/ours_2view" \
    --device "${DEVICE}" \
    --sample-steps "${SAMPLE_STEPS}"
fi
