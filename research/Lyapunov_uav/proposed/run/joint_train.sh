#!/usr/bin/bash

#SBATCH -J slow-dpp-joint
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

set -euo pipefail

PROJECT_ROOT="/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed"
CONDA_ROOT="/data/surt321/anaconda3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/lab"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
RUNTIME_CACHE_ROOT="${PROJECT_ROOT}/.runtime-cache"
MPL_CACHE_ROOT="${RUNTIME_CACHE_ROOT}/matplotlib"
TMP_CACHE_PARENT="${RUNTIME_CACHE_ROOT}/tmp"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"
hash -r

cd "${PROJECT_ROOT}"

mkdir -p \
    "${MPL_CACHE_ROOT}" \
    "${TMP_CACHE_PARENT}" \
    logs

RUNTIME_TMPDIR="$(
    mktemp -d \
        "${TMP_CACHE_PARENT}/joint-${SLURM_JOB_ID:-manual}-XXXXXX"
)"
trap 'rm -rf -- "${RUNTIME_TMPDIR}"' EXIT

export MPLCONFIGDIR="${MPL_CACHE_ROOT}"
export TMPDIR="${RUNTIME_TMPDIR}"

echo "=================================================="
echo "[JOINT ENVIRONMENT CHECK]"
echo "=================================================="
echo "Hostname      : $(hostname)"
echo "Working dir   : $(pwd)"
echo "Git branch    : $(git branch --show-current)"
echo "Git commit    : $(git rev-parse --short HEAD)"
echo "CONDA_PREFIX  : ${CONDA_PREFIX:-NOT_SET}"
echo "Fast init     : random_from_scratch"
echo "Fast source   : NONE"
echo "Resume source : ${JOINT_RESUME_CHECKPOINT:-NONE}"
echo "TMPDIR        : ${TMPDIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable does not exist: ${PYTHON_BIN}"
    exit 127
fi
if [[ -n "${JOINT_FAST_CHECKPOINT:-}" ]]; then
    echo "[ERROR] JOINT_FAST_CHECKPOINT is not supported."
    echo "        Fresh joint training must not load Fast-only weights."
    exit 2
fi
if [[ -n "${JOINT_RESUME_CHECKPOINT:-}" ]] \
    && [[ ! -f "${JOINT_RESUME_CHECKPOINT}" ]]; then
    echo "[ERROR] Joint resume checkpoint does not exist:"
    echo "        ${JOINT_RESUME_CHECKPOINT}"
    exit 2
fi

"${PYTHON_BIN}" -c '
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the joint run.")
print("GPU:", torch.cuda.get_device_name(0))
'

"${PYTHON_BIN}" -m compileall -q \
    agent/PPO/joint \
    agent/PPO/slow \
    agent/PPO/fast \
    env \
    config.py

"${PYTHON_BIN}" -u -m agent.PPO.slow.test_fast_slow
"${PYTHON_BIN}" -u -m agent.PPO.joint.test_joint_checkpoint

echo "=================================================="
echo "[SLOW DPP + FAST PPO FROM-SCRATCH JOINT TRAIN]"
echo "Start time: $(date)"
echo "=================================================="

"${PYTHON_BIN}" -u -m agent.PPO.joint.joint_train

echo "=================================================="
echo "End time: $(date)"
echo "=================================================="