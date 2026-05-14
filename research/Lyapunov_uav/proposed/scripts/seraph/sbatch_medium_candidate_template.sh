#!/usr/bin/env bash
#SBATCH --job-name=hrl-medium-candidate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=04:00:00
#SBATCH --array=0-TODO_MAX_ARRAY_INDEX
#SBATCH --output=/data/%u/hrl_uav/logs/%x-%A_%a.out
#SBATCH --error=/data/%u/hrl_uav/logs/%x-%A_%a.err

set -euo pipefail

# Seraph template for medium-length candidate stability runs.
# Edit TODO_PARTITION, TODO_MAX_ARRAY_INDEX, CONDA_ENV, and PROJECT_DIR.
# This is not final long training and does not run baselines.

DATA_ROOT="${DATA_ROOT:-/data/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${DATA_ROOT}/research}"
CONDA_ENV="${CONDA_ENV:-uav}"
MANIFEST_JSON="${MANIFEST_JSON:?Set MANIFEST_JSON to the medium manifest JSON path}"
JOB_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

MAX_QUEUE="${MAX_QUEUE:-100.0}"
ENTROPY_MIN="${ENTROPY_MIN:-1e-8}"
REWARD_TOL="${REWARD_TOL:-1e-8}"
EPS="${EPS:-1e-8}"
DOMINANCE_THRESHOLD="${DOMINANCE_THRESHOLD:-0.8}"
SUSTAINED_FRACTION="${SUSTAINED_FRACTION:-0.8}"
NEAR_ZERO_FRACTION="${NEAR_ZERO_FRACTION:-0.8}"
REWARD_LOSS_MIN_RATIO="${REWARD_LOSS_MIN_RATIO:-1e-3}"
REWARD_LOSS_MAX_RATIO="${REWARD_LOSS_MAX_RATIO:-1e3}"
LOSS_SCALE_EPS="${LOSS_SCALE_EPS:-1e-6}"
ENTROPY_COLLAPSE_RATIO="${ENTROPY_COLLAPSE_RATIO:-0.1}"
APPROX_KL_MAX="${APPROX_KL_MAX:-0.05}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
else
    echo "conda command not found; continuing with current Python environment"
fi

cd "${PROJECT_DIR}"

JOB_LINE="$(
python3 - "${MANIFEST_JSON}" "${JOB_INDEX}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser()
job_index = int(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
jobs = manifest["jobs"]
if job_index < 0 or job_index >= len(jobs):
    raise SystemExit(f"job_index={job_index} is outside [0, {len(jobs) - 1}]")
job = jobs[job_index]
cfg = manifest["medium_config"]
fields = [
    job["preset_name"],
    str(job["seed"]),
    job["run_name"],
    job["log_dir"],
    job["checkpoint_dir"],
    job["output_dir"],
    str(cfg["max_steps"]),
    str(cfg["max_updates"]),
    str(cfg["rollout_steps"]),
    str(cfg["hidden_dim"]),
    str(cfg["device"]),
    str(cfg["checkpoint_interval"]),
]
print("\t".join(fields))
PY
)"

IFS=$'\t' read -r REWARD_PRESET SEED RUN_NAME LOG_DIR CHECKPOINT_DIR OUTPUT_DIR \
    MAX_STEPS MAX_UPDATES ROLLOUT_STEPS HIDDEN_DIM DEVICE CHECKPOINT_INTERVAL <<< "${JOB_LINE}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${OUTPUT_DIR}"

LOG_PATH="${LOG_DIR}/${RUN_NAME}.jsonl"
REPORT_JSON="${OUTPUT_DIR}/${RUN_NAME}_report.json"
REPORT_MD="${OUTPUT_DIR}/${RUN_NAME}_report.md"
SCALE_JSON="${OUTPUT_DIR}/${RUN_NAME}_scale_analysis.json"
SCALE_MD="${OUTPUT_DIR}/${RUN_NAME}_scale_analysis.md"
SANITY_TXT="${OUTPUT_DIR}/${RUN_NAME}_sanity.txt"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "MANIFEST_JSON=${MANIFEST_JSON}"
echo "JOB_INDEX=${JOB_INDEX}"
echo "RUN_NAME=${RUN_NAME}"
echo "REWARD_PRESET=${REWARD_PRESET}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "MAX_UPDATES=${MAX_UPDATES}"
echo "ROLLOUT_STEPS=${ROLLOUT_STEPS}"
echo "LOG_DIR=${LOG_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python3 -c "import torch; print('torch_cuda_available=', torch.cuda.is_available()); print('torch_cuda_device_count=', torch.cuda.device_count())"

python3 Lyapunov_uav/proposed/scripts/short_hrl_train.py \
    --max-steps "${MAX_STEPS}" \
    --max-updates "${MAX_UPDATES}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --seed "${SEED}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --device "${DEVICE}" \
    --reward-preset "${REWARD_PRESET}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    --run-name "${RUN_NAME}" \
    --log-dir "${LOG_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    | tee "${LOG_DIR}/${RUN_NAME}_console.log"

python3 Lyapunov_uav/proposed/scripts/check_short_train_sanity.py \
    --log-path "${LOG_PATH}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --max-queue "${MAX_QUEUE}" \
    --entropy-min "${ENTROPY_MIN}" \
    --reward-tol "${REWARD_TOL}" \
    | tee "${SANITY_TXT}"

python3 Lyapunov_uav/proposed/scripts/report_short_train.py \
    --log-path "${LOG_PATH}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-json "${REPORT_JSON}" \
    --output-md "${REPORT_MD}" \
    --max-queue "${MAX_QUEUE}" \
    --entropy-min "${ENTROPY_MIN}" \
    --reward-tol "${REWARD_TOL}"

python3 Lyapunov_uav/proposed/scripts/analyze_reward_scale.py \
    --log-path "${LOG_PATH}" \
    --output-json "${SCALE_JSON}" \
    --output-md "${SCALE_MD}" \
    --eps "${EPS}" \
    --dominance-threshold "${DOMINANCE_THRESHOLD}" \
    --sustained-fraction "${SUSTAINED_FRACTION}" \
    --near-zero-fraction "${NEAR_ZERO_FRACTION}" \
    --reward-loss-min-ratio "${REWARD_LOSS_MIN_RATIO}" \
    --reward-loss-max-ratio "${REWARD_LOSS_MAX_RATIO}" \
    --loss-scale-eps "${LOSS_SCALE_EPS}" \
    --entropy-collapse-ratio "${ENTROPY_COLLAPSE_RATIO}" \
    --entropy-min "${ENTROPY_MIN}" \
    --approx-kl-max "${APPROX_KL_MAX}"

echo "medium_candidate_job_done run_name=${RUN_NAME}"
