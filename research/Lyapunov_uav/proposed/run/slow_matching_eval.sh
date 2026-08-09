#!/usr/bin/bash

#SBATCH -J slow-mwm-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-slow-mwm-eval-%A.out

set -euo pipefail
umask 027

PROJECT_ROOT="${FAST_PROJECT_ROOT:-/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
CONDA_ROOT="${FAST_CONDA_ROOT:-/data/surt321/anaconda3}"
CONDA_ENV_PATH="${FAST_CONDA_ENV_PATH:-${CONDA_ROOT}/envs/lab}"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
EXPECTED_BRANCH="feat/no-hrl"

FAST_PPO_CHECKPOINT="${FAST_PPO_CHECKPOINT:-}"
FAST_PPO_SEED="${FAST_PPO_SEED:-2026}"
FAST_PPO_EVAL_EPISODES="${FAST_PPO_EVAL_EPISODES:-1}"
FAST_PPO_EVAL_ROUNDS_PER_EPISODE="${FAST_PPO_EVAL_ROUNDS_PER_EPISODE:-1}"
FAST_PPO_DPP_FORECAST_WORKERS="${FAST_PPO_DPP_FORECAST_WORKERS:-18}"
FAST_PPO_RUN_NAME="${FAST_PPO_RUN_NAME:-eval_joint_mwm_seed${FAST_PPO_SEED}_smoke}"
FAST_PPO_OUTPUT_ROOT="${FAST_PPO_OUTPUT_ROOT:-eval}"

JOB_ID="${SLURM_JOB_ID:-manual}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
RUNTIME_CACHE_ROOT="${PROJECT_ROOT}/.runtime-cache"
MPL_CACHE_ROOT="${RUNTIME_CACHE_ROOT}/matplotlib"
TMP_CACHE_PARENT="${RUNTIME_CACHE_ROOT}/tmp"
JOB_LOG_ROOT="${PROJECT_ROOT}/logs/jobs/${JOB_ID}"
GPU_RESOURCE_LOG="${JOB_LOG_ROOT}/gpu-resources.csv"
CPU_RESOURCE_LOG="${JOB_LOG_ROOT}/cpu-resources.txt"
RUN_CONSOLE_LOG="${JOB_LOG_ROOT}/slow-mwm-eval-console.log"
ENVIRONMENT_LOG="${JOB_LOG_ROOT}/environment.txt"

GPU_MONITOR_PID=""
CPU_MONITOR_PID=""
RUNTIME_TMPDIR=""

die() {
    echo "[ERROR] $*" >&2
    exit 2
}

cleanup() {
    local exit_code=$?
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    if [[ -n "${CPU_MONITOR_PID}" ]]; then
        kill "${CPU_MONITOR_PID}" 2>/dev/null || true
        wait "${CPU_MONITOR_PID}" 2>/dev/null || true
    fi
    if [[ -n "${RUNTIME_TMPDIR}" && -d "${RUNTIME_TMPDIR}" ]]; then
        rm -rf -- "${RUNTIME_TMPDIR}"
    fi
    printf '[CLEANUP] exit_code=%s time=%s\n' \
        "${exit_code}" "$(date --iso-8601=seconds)" || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

[[ -d "${PROJECT_ROOT}" ]] \
    || die "PROJECT_ROOT does not exist: ${PROJECT_ROOT}"
[[ -x "${PYTHON_BIN}" ]] \
    || die "Python executable does not exist: ${PYTHON_BIN}"
[[ -n "${FAST_PPO_CHECKPOINT}" ]] \
    || die "FAST_PPO_CHECKPOINT is required."

cd "${PROJECT_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"
[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] \
    || die "Wrong branch: expected=${EXPECTED_BRANCH}, actual=${CURRENT_BRANCH}"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
    || die "Working tree is not clean. Commit or stash source changes first."

mkdir -p \
    "${MPL_CACHE_ROOT}" \
    "${TMP_CACHE_PARENT}" \
    "${JOB_LOG_ROOT}"

if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
    RUNTIME_TMPDIR="${SLURM_TMPDIR}/slow-mwm-eval-${JOB_ID}"
    mkdir -p "${RUNTIME_TMPDIR}"
else
    RUNTIME_TMPDIR="$(
        mktemp -d \
            "${TMP_CACHE_PARENT}/slow-mwm-eval-${JOB_ID}-XXXXXX"
    )"
fi

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"
hash -r
[[ "$(command -v python)" == "${PYTHON_BIN}" ]] \
    || die "Activated Python differs from ${PYTHON_BIN}."

"${PYTHON_BIN}" - <<'PY'
import scipy
from scipy.optimize import linear_sum_assignment
print("[DEPENDENCY] scipy =", scipy.__version__)
print("[DEPENDENCY] linear_sum_assignment = OK")
PY

export MPLCONFIGDIR="${MPL_CACHE_ROOT}"
export TMPDIR="${RUNTIME_TMPDIR}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${FAST_PPO_SEED}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

export FAST_PPO_PHASE=eval_joint
export FAST_PPO_CHECKPOINT
export FAST_PPO_SEED
export FAST_PPO_EVAL_EPISODES
export FAST_PPO_EVAL_ROUNDS_PER_EPISODE
export FAST_PPO_DPP_FORECAST_WORKERS
export FAST_PPO_RUN_NAME
export FAST_PPO_OUTPUT_ROOT
export FAST_PPO_DETERMINISTIC_TORCH=1
export FAST_PPO_DEVICE="${FAST_PPO_DEVICE:-cuda}"
export FAST_PPO_FAIL_IF_CUDA_UNAVAILABLE=1

unset FAST_PRETRAIN_RESUME_CHECKPOINT
unset JOINT_FAST_CHECKPOINT
unset JOINT_RESUME_CHECKPOINT
unset FAST_PPO_RESUME

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "job_id=${JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-NOT_SET}"
    echo "working_directory=$(pwd)"
    echo "git_branch=${CURRENT_BRANCH}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_describe=$(git describe --always --dirty --tags 2>/dev/null || true)"
    echo "conda_prefix=${CONDA_PREFIX:-NOT_SET}"
    echo "python=${PYTHON_BIN}"
    echo "allocated_cpus=${ALLOCATED_CPUS}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-NOT_SET}"
    echo "checkpoint=${FAST_PPO_CHECKPOINT}"
    echo "seed=${FAST_PPO_SEED}"
    echo "eval_episodes=${FAST_PPO_EVAL_EPISODES}"
    echo "eval_rounds=${FAST_PPO_EVAL_ROUNDS_PER_EPISODE}"
    echo "dpp_workers=${FAST_PPO_DPP_FORECAST_WORKERS}"
    echo "run_name=${FAST_PPO_RUN_NAME}"
} | tee "${ENVIRONMENT_LOG}"

"${PYTHON_BIN}" -m py_compile \
    env/env.py \
    agent/PPO/slow/slow_matching.py \
    agent/PPO/slow/slow_train.py \
    agent/PPO/fast/fast_train.py

"${PYTHON_BIN}" -m unittest -v \
    agent.PPO.slow.test_fast_slow \
    agent.PPO.fast.test_fast_pretrain_contract

if command -v nvidia-smi >/dev/null 2>&1; then
    (
        while true; do
            nvidia-smi \
                --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
                --format=csv,noheader,nounits \
                >> "${GPU_RESOURCE_LOG}" 2>/dev/null || true
            sleep 30
        done
    ) &
    GPU_MONITOR_PID=$!
fi

(
    while true; do
        {
            date --iso-8601=seconds
            ps -o pid,ppid,pcpu,pmem,rss,vsz,etime,cmd -u "${USER}" \
                | head -n 80
            echo
        } >> "${CPU_RESOURCE_LOG}" 2>/dev/null || true
        sleep 60
    done
) &
CPU_MONITOR_PID=$!

echo "[SLOW MWM EVAL START]"
echo "checkpoint=${FAST_PPO_CHECKPOINT}"
echo "run_name=${FAST_PPO_RUN_NAME}"

set +e
"${PYTHON_BIN}" -u -m agent.PPO.fast.fast_train \
    2>&1 | tee "${RUN_CONSOLE_LOG}"
status=${PIPESTATUS[0]}
set -e

if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] Slow-MWM evaluation failed with status=${status}." >&2
    exit "${status}"
fi

echo "[SLOW MWM EVAL DONE]"
echo "output=${FAST_PPO_OUTPUT_ROOT}/${FAST_PPO_RUN_NAME}"
