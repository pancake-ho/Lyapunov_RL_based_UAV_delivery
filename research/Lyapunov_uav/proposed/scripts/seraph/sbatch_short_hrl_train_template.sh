#!/usr/bin/env bash
#SBATCH --job-name=hrl-short-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=00:30:00
#SBATCH --output=/data/%u/hrl_uav/logs/%x-%j.out
#SBATCH --error=/data/%u/hrl_uav/logs/%x-%j.err

set -euo pipefail

# Seraph template for a short HRL PPO train run before long training.
# Edit TODO_PARTITION, CONDA_ENV, and PROJECT_DIR for your account.

DATA_ROOT="${DATA_ROOT:-/data/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${DATA_ROOT}/research}"
CONDA_ENV="${CONDA_ENV:-uav}"

RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/hrl_uav}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/outputs}"

RUN_NAME="${RUN_NAME:-short_hrl_train_${SLURM_JOB_ID:-local}}"
MAX_STEPS="${MAX_STEPS:-20}"
MAX_UPDATES="${MAX_UPDATES:-2}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-4}"
SEED="${SEED:-2026}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEVICE="${DEVICE:-auto}"
REWARD_PRESET="${REWARD_PRESET:-balanced}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${OUTPUT_DIR}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
else
    echo "conda command not found; continuing with current Python environment"
fi

cd "${PROJECT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-local}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "DEVICE=${DEVICE}"
echo "REWARD_PRESET=${REWARD_PRESET}"

python3 -c "import torch; print('torch_cuda_available=', torch.cuda.is_available()); print('torch_cuda_device_count=', torch.cuda.device_count())"

python3 Lyapunov_uav/proposed/scripts/short_hrl_train.py \
    --max-steps "${MAX_STEPS}" \
    --max-updates "${MAX_UPDATES}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --seed "${SEED}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --device "${DEVICE}" \
    --reward-preset "${REWARD_PRESET}" \
    --run-name "${RUN_NAME}" \
    --log-dir "${LOG_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    | tee "${LOG_DIR}/${RUN_NAME}_console.log"
