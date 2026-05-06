#!/usr/bin/env bash
set -euo pipefail

# Short interactive Seraph smoke runner.
# Keep code, logs, checkpoints, and outputs under /data/$USER.

DATA_ROOT="${DATA_ROOT:-/data/${USER}}"
PROJECT_DIR="${PROJECT_DIR:-${DATA_ROOT}/research}"
CONDA_ENV="${CONDA_ENV:-uav}"
SLOW_STEPS="${SLOW_STEPS:-4}"
FAST_STEPS="${FAST_STEPS:-4}"
SEED="${SEED:-2026}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
DEVICE="${DEVICE:-auto}"

RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/hrl_uav}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/outputs}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${OUTPUT_DIR}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

cd "${PROJECT_DIR}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DEVICE=${DEVICE}"

python3 -c "import torch; print('torch_cuda_available=', torch.cuda.is_available()); print('torch_cuda_device_count=', torch.cuda.device_count())"

python3 Lyapunov_uav/proposed/scripts/ppo_update_smoke_test.py \
    --slow-steps "${SLOW_STEPS}" \
    --fast-steps "${FAST_STEPS}" \
    --seed "${SEED}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --device "${DEVICE}" \
    | tee "${LOG_DIR}/ppo_update_smoke_seed${SEED}.jsonl"
