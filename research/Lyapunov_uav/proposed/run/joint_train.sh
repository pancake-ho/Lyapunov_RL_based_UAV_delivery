#!/usr/bin/bash

#SBATCH -J slow-dpp-joint
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=29G
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
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
RESOURCE_LOG="${PROJECT_ROOT}/logs/resources-${SLURM_JOB_ID:-manual}.csv"
GPU_MONITOR_PID=""

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

cleanup() {
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    if [[ -d "${RUNTIME_TMPDIR}" ]]; then
        rm -rf -- "${RUNTIME_TMPDIR}"
    fi
}
trap cleanup EXIT

export MPLCONFIGDIR="${MPL_CACHE_ROOT}"
export TMPDIR="${RUNTIME_TMPDIR}"
export PYTHONUNBUFFERED=1

# Each forecast environment is already parallelized explicitly. Prevent
# NumPy/BLAS/PyTorch from creating another thread team inside every worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

export JOINT_FORECAST_ENV_WORKERS="${JOINT_FORECAST_ENV_WORKERS:-${ALLOCATED_CPUS}}"
export JOINT_FORECAST_CANDIDATE_BATCH_SIZE="${JOINT_FORECAST_CANDIDATE_BATCH_SIZE:-64}"

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
echo "Slurm CPUs    : ${ALLOCATED_CPUS}"
echo "Forecast CPUs : ${JOINT_FORECAST_ENV_WORKERS}"
echo "Forecast batch: ${JOINT_FORECAST_CANDIDATE_BATCH_SIZE}"
echo "Resource log  : ${RESOURCE_LOG}"

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
import os
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the joint run.")
print("GPU:", torch.cuda.get_device_name(0))

available_cpus = sorted(os.sched_getaffinity(0))
forecast_workers = int(os.environ["JOINT_FORECAST_ENV_WORKERS"])
print("CPU affinity:", available_cpus)
if forecast_workers > len(available_cpus):
    raise SystemExit(
        "JOINT_FORECAST_ENV_WORKERS exceeds Slurm CPU affinity: "
        f"workers={forecast_workers}, available={len(available_cpus)}"
    )
'

nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv \
    -l 5 \
    > "${RESOURCE_LOG}" 2>&1 &
GPU_MONITOR_PID="$!"

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