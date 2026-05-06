#!/usr/bin/env bash
#SBATCH --job-name=metadata_2f
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/metadata_2f_%A_%a.log
#SBATCH --error=logs/metadata_2f_%A_%a.err

# --- Slack Notification Setup ---
# This function will be called by the trap when the script exits
# It checks the exit code of the last command ($?)
finish() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        ~/bin/notify_slack.sh "SUCCESS" "Job '$SLURM_JOB_NAME' completed successfully."
    else
        ~/bin/notify_slack.sh "FAILED" "Job '$SLURM_JOB_NAME' failed with exit code $EXIT_CODE."
    fi
}

# Set the trap to call the 'finish' function on EXIT
trap finish EXIT
# --------------------------------

# Send a notification that the job has started
~/bin/notify_slack.sh "BEGIN" "Job '$SLURM_JOB_NAME' has started."

set -euo pipefail

ROOT=${ROOT:-/net/holy-isilon/ifs/rc_labs/ydu_lab/zhiyi24/workspace/generative_baselines_3D}
MAX_SAMPLES=${MAX_SAMPLES:-}

DATASETS=(
    aria
    dl3dv
    dl3dv_test
    re10k
    scannetpp
    scenenet_depth
    spatialvid_nvs
    tanksandtemples
    vkitti2
    7scenes
    tum
    bonn
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
cd "$ROOT"

if [ -n "$MAX_SAMPLES" ]; then
    bash "$ROOT/run_dataset_metadata_2f.sh" "$DATASET" 0 "$MAX_SAMPLES"
else
    bash "$ROOT/run_dataset_metadata_2f.sh" "$DATASET" 0
fi
