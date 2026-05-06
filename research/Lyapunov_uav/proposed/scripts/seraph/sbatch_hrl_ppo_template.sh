#!/usr/bin/env bash
#SBATCH --job-name=hrl-ppo-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=TODO_PARTITION
#SBATCH --time=00:20:00
#SBATCH --output=/data/%u/hrl_uav/logs/%x-%j.out
#SBATCH --error=/data/%u/hrl_uav/logs/%x-%j.err

set -euo pipefail

# Seraph sbatch template for short HRL PPO smoke/train runs.
# Edit TODO_PARTITION, CONDA_ENV, PROJECT_DIR, and TRAIN_COMMAND for your account.
# Keep runtime artifacts under /data/$USER; leave /home for shell/conda config only.

DATA_ROOT="${DATA_ROOT:-/data/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${DATA_ROOT}/research}"
CONDA_ENV="${CONDA_ENV:-uav}"

RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/hrl_uav}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/outputs}"

SLOW_STEPS="${SLOW_STEPS:-4}"
FAST_STEPS="${FAST_STEPS:-4}"
SEED="${SEED:-2026}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEVICE="${DEVICE:-auto}"

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
echo "DATA_ROOT=${DATA_ROOT}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DEVICE=${DEVICE}"

python3 -c "import torch; print('torch_cuda_available=', torch.cuda.is_available()); print('torch_cuda_device_count=', torch.cuda.device_count())"

TRAIN_COMMAND="${TRAIN_COMMAND:-python3 Lyapunov_uav/proposed/scripts/ppo_update_smoke_test.py --slow-steps ${SLOW_STEPS} --fast-steps ${FAST_STEPS} --seed ${SEED} --hidden-dim ${HIDDEN_DIM} --device ${DEVICE}}"

echo "TRAIN_COMMAND=${TRAIN_COMMAND}"
eval "${TRAIN_COMMAND}" | tee "${LOG_DIR}/hrl_ppo_${SLURM_JOB_ID:-local}.jsonl"
