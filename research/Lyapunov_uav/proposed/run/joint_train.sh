#!/usr/bin/bash

#SBATCH -J slow-dpp-joint
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

set -euo pipefail

PROJECT_ROOT="/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed"
CONDA_ROOT="/data/surt321/anaconda3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/lab"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
DEFAULT_FAST_CHECKPOINT="${PROJECT_ROOT}/fast/fast_mixed_seed2026_continuous_mobility_slot1s_noklstop/checkpoints/fast_ppo_final.pt"
FAST_CHECKPOINT="${JOINT_FAST_CHECKPOINT:-${DEFAULT_FAST_CHECKPOINT}}"
MPL_CACHE_ROOT="${PROJECT_ROOT}/.runtime-cache/matplotlib"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"
hash -r

cd "${PROJECT_ROOT}"
mkdir -p "${MPL_CACHE_ROOT}"
export MPLCONFIGDIR="${MPL_CACHE_ROOT}"
export JOINT_FAST_CHECKPOINT="${FAST_CHECKPOINT}"

echo "=================================================="
echo "[JOINT ENVIRONMENT CHECK]"
echo "=================================================="
echo "Hostname      : $(hostname)"
echo "Working dir   : $(pwd)"
echo "Git branch    : $(git branch --show-current)"
echo "Git commit    : $(git rev-parse --short HEAD)"
echo "CONDA_PREFIX  : ${CONDA_PREFIX:-NOT_SET}"
echo "Fast source   : ${JOINT_FAST_CHECKPOINT}"
echo "Resume source : ${JOINT_RESUME_CHECKPOINT:-NONE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable does not exist: ${PYTHON_BIN}"
    exit 127
fi
if [[ ! -f "${JOINT_FAST_CHECKPOINT}" ]]; then
    echo "[ERROR] Fast checkpoint does not exist: ${JOINT_FAST_CHECKPOINT}"
    exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the joint run.")
print("GPU:", torch.cuda.get_device_name(0))
PY

"${PYTHON_BIN}" -m compileall -q \
    agent/PPO/joint \
    agent/PPO/slow \
    agent/PPO/fast \
    env \
    config.py

"${PYTHON_BIN}" -u -m agent.PPO.slow.test_fast_slow
"${PYTHON_BIN}" -u -m agent.PPO.joint.test_joint_checkpoint

echo "=================================================="
echo "[SLOW DPP + FAST PPO JOINT TRAIN]"
echo "Start time: $(date)"
echo "=================================================="

"${PYTHON_BIN}" -u -m agent.PPO.joint.joint_train

echo "=================================================="
echo "End time: $(date)"
echo "=================================================="
