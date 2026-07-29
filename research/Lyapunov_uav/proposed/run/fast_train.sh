#!/usr/bin/bash

#SBATCH -J fast-ppo
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

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"
hash -r

cd "${PROJECT_ROOT}"

echo "=================================================="
echo "[ENVIRONMENT CHECK]"
echo "=================================================="
echo "Hostname      : $(hostname)"
echo "Working dir   : $(pwd)"
echo "CONDA_PREFIX  : ${CONDA_PREFIX:-NOT_SET}"
echo "Python target : ${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable does not exist:"
    echo "        ${PYTHON_BIN}"
    echo
    echo "[INFO] Files in the environment bin directory:"
    ls -al "${CONDA_ENV_PATH}/bin" 2>/dev/null || true
    exit 127
fi

"${PYTHON_BIN}" --version

echo "=================================================="
echo "[FAST PPO TRAIN]"
echo "Start time: $(date)"
echo "=================================================="

"${PYTHON_BIN}" -u -m agent.PPO.fast.fast_train

echo "=================================================="
echo "End time: $(date)"
echo "=================================================="