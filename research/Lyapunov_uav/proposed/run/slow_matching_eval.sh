#!/usr/bin/bash

#SBATCH -J slow-mwm-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH --exclude=moana-y5
#SBATCH -o logs/slurm-slow-mwm-eval-%A.out

set -euo pipefail
umask 027


# =====================================================================
# Fixed project/runtime paths
# =====================================================================

PROJECT_ROOT="/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed"

CONDA_ROOT="/data/surt321/anaconda3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/lab"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"

EXPECTED_BRANCH="feat/no-hrl"

FAST_PPO_CHECKPOINT="${PROJECT_ROOT}/fast/fast_pretrain_h2_seed2026_ep400_v1/checkpoints/fast_ppo_pretrain_ep300.pt"

JOB_ID="${SLURM_JOB_ID:-manual}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-16}"

if (( ALLOCATED_CPUS < 2 )); then
    echo "[ERROR] Slow DPP requires at least 2 allocated CPUs." >&2
    exit 2
fi

# Keep one allocated CPU for the main process.
DPP_WORKERS=$((ALLOCATED_CPUS - 1))


# =====================================================================
# Evaluation mode
# =====================================================================

MODE="${1:-smoke}"

case "${MODE}" in

    smoke)
        EVAL_SEEDS=(
            2026
        )

        EVAL_EPISODES=1
        EVAL_ROUNDS_PER_EPISODE=1

        # Strong mutation checks are cheap enough for the one-round smoke.
        AUDIT_RUNTIME_INVARIANTS=1
        ;;

    pilot)
        EVAL_SEEDS=(
            2026
            2027
            2028
        )

        EVAL_EPISODES=2
        EVAL_ROUNDS_PER_EPISODE=5

        AUDIT_RUNTIME_INVARIANTS=0
        ;;

    *)
        echo "[ERROR] Unknown mode: ${MODE}" >&2
        echo "Usage:" >&2
        echo "  sbatch run/slow_matching_eval.sh smoke" >&2
        echo "  sbatch run/slow_matching_eval.sh pilot" >&2
        exit 2
        ;;
esac


OUTPUT_ROOT="${PROJECT_ROOT}/eval/slow_mwm_ep300_${MODE}_job${JOB_ID}"


# =====================================================================
# Runtime cache / logs
# =====================================================================

RUNTIME_CACHE_ROOT="${PROJECT_ROOT}/.runtime-cache"

MPL_CACHE_ROOT="${RUNTIME_CACHE_ROOT}/matplotlib"

TMP_CACHE_PARENT="${RUNTIME_CACHE_ROOT}/tmp"

JOB_LOG_ROOT="${PROJECT_ROOT}/logs/jobs/${JOB_ID}"

GPU_RESOURCE_LOG="${JOB_LOG_ROOT}/gpu-resources.csv"

ENVIRONMENT_LOG="${JOB_LOG_ROOT}/environment.txt"

GPU_MONITOR_PID=""
RUNTIME_TMPDIR=""


# =====================================================================
# Helpers
# =====================================================================

die() {
    echo "[ERROR] $*" >&2
    exit 2
}


cleanup() {
    local exit_code=$?

    set +e

    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" \
            2>/dev/null || true

        wait "${GPU_MONITOR_PID}" \
            2>/dev/null || true
    fi

    if [[ -n "${RUNTIME_TMPDIR}" \
        && -d "${RUNTIME_TMPDIR}" ]]; then

        rm -rf -- \
            "${RUNTIME_TMPDIR}"
    fi

    printf \
        '[CLEANUP] exit_code=%s time=%s\n' \
        "${exit_code}" \
        "$(date --iso-8601=seconds)" \
        || true
}


trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP


# =====================================================================
# Remove stale experiment variables
# =====================================================================

unset FAST_PPO_PHASE || true
unset FAST_PPO_CHECKPOINT || true
unset FAST_PPO_SEED || true

unset FAST_PPO_EVAL_EPISODES || true
unset FAST_PPO_EVAL_ROUNDS_PER_EPISODE || true

unset FAST_PPO_RUN_NAME || true
unset FAST_PPO_OUTPUT_ROOT || true

unset FAST_PPO_DPP_FORECAST_WORKERS || true
unset FAST_PPO_AUDIT_RUNTIME_INVARIANTS || true

unset FAST_PPO_RESUME || true
unset FAST_PRETRAIN_RESUME_CHECKPOINT || true
unset JOINT_FAST_CHECKPOINT || true
unset JOINT_RESUME_CHECKPOINT || true

unset ALLOW_EXISTING_RUN_DIR || true


# =====================================================================
# Static checks
# =====================================================================

[[ -d "${PROJECT_ROOT}" ]] \
    || die \
    "PROJECT_ROOT does not exist: ${PROJECT_ROOT}"

[[ -x "${PYTHON_BIN}" ]] \
    || die \
    "Python executable does not exist: ${PYTHON_BIN}"

[[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]] \
    || die \
    "Conda initialization script does not exist."

[[ -f "${FAST_PPO_CHECKPOINT}" ]] \
    || die \
    "Fast checkpoint not found: ${FAST_PPO_CHECKPOINT}"


cd "${PROJECT_ROOT}"


# =====================================================================
# Git contract
# =====================================================================

CURRENT_BRANCH="$(
    git branch --show-current
)"

[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] \
    || die \
    "Wrong branch: expected=${EXPECTED_BRANCH}, actual=${CURRENT_BRANCH}"

[[ -z "$(
    git status \
        --porcelain \
        --untracked-files=normal
)" ]] \
    || die \
    "Working tree is not clean. Commit or stash source changes first."


# =====================================================================
# Runtime directories
# =====================================================================

mkdir -p \
    "${MPL_CACHE_ROOT}" \
    "${TMP_CACHE_PARENT}" \
    "${JOB_LOG_ROOT}" \
    "${OUTPUT_ROOT}"


if [[ -n "${SLURM_TMPDIR:-}" \
    && -d "${SLURM_TMPDIR}" ]]; then

    RUNTIME_TMPDIR="${SLURM_TMPDIR}/slow-mwm-${JOB_ID}"

    mkdir -p \
        "${RUNTIME_TMPDIR}"

else

    RUNTIME_TMPDIR="$(
        mktemp -d \
        "${TMP_CACHE_PARENT}/slow-mwm-${JOB_ID}-XXXXXX"
    )"

fi


# =====================================================================
# Conda
# =====================================================================

# shellcheck source=/dev/null
source \
    "${CONDA_ROOT}/etc/profile.d/conda.sh"

conda activate \
    "${CONDA_ENV_PATH}"

hash -r


[[ "$(command -v python)" == "${PYTHON_BIN}" ]] \
    || die \
    "Activated Python differs from ${PYTHON_BIN}: $(command -v python)"


# =====================================================================
# Common runtime environment
# =====================================================================

export MPLCONFIGDIR="${MPL_CACHE_ROOT}"
export TMPDIR="${RUNTIME_TMPDIR}"

export PYTHONUNBUFFERED=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2


export FAST_PPO_PHASE="eval_joint"

export FAST_PPO_CHECKPOINT="${FAST_PPO_CHECKPOINT}"

export FAST_PPO_DPP_FORECAST_WORKERS="${DPP_WORKERS}"

export FAST_PPO_DETERMINISTIC_TORCH=1

export FAST_PPO_DEVICE="cuda"

export FAST_PPO_FAIL_IF_CUDA_UNAVAILABLE=1

export FAST_PPO_AUDIT_RUNTIME_INVARIANTS="${AUDIT_RUNTIME_INVARIANTS}"

export ALLOW_EXISTING_RUN_DIR=0


# =====================================================================
# Environment log
# =====================================================================

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"

    echo "job_id=${JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-NOT_SET}"

    echo "working_directory=$(pwd)"

    echo "git_branch=${CURRENT_BRANCH}"
    echo "git_commit=$(git rev-parse HEAD)"

    echo "git_describe=$(
        git describe \
            --always \
            --dirty \
            --tags \
            2>/dev/null \
            || true
    )"

    echo "conda_prefix=${CONDA_PREFIX:-NOT_SET}"
    echo "python=${PYTHON_BIN}"

    echo "allocated_cpus=${ALLOCATED_CPUS}"
    echo "dpp_workers=${DPP_WORKERS}"

    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-NOT_SET}"

    echo "mode=${MODE}"

    echo "checkpoint=${FAST_PPO_CHECKPOINT}"

    echo "eval_seeds=${EVAL_SEEDS[*]}"

    echo "eval_episodes=${EVAL_EPISODES}"

    echo "eval_rounds_per_episode=${EVAL_ROUNDS_PER_EPISODE}"

    echo "output_root=${OUTPUT_ROOT}"

} | tee \
    "${ENVIRONMENT_LOG}"


# =====================================================================
# Dependency check
# =====================================================================

"${PYTHON_BIN}" - <<'PY'

import scipy

from scipy.optimize import (
    linear_sum_assignment,
)


print(
    "[DEPENDENCY] scipy =",
    scipy.__version__,
)

print(
    "[DEPENDENCY] linear_sum_assignment = OK"
)

PY


# =====================================================================
# Compile / unit tests
# =====================================================================

"${PYTHON_BIN}" -m py_compile \
    env/env.py \
    agent/PPO/config.py \
    agent/PPO/slow/slow_matching.py \
    agent/PPO/slow/slow_train.py \
    agent/PPO/slow/test_fast_slow.py \
    agent/PPO/fast/fast_train.py


"${PYTHON_BIN}" -m unittest -v \
    agent.PPO.slow.test_fast_slow \
    agent.PPO.fast.test_fast_pretrain_contract


# =====================================================================
# CUDA preflight
# =====================================================================

echo "============================================================"
echo "[CUDA PREFLIGHT]"
echo "============================================================"


srun \
    --ntasks=1 \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" - <<'PY'

import os
import socket

import torch


print(
    "hostname             =",
    socket.gethostname(),
)

print(
    "torch                =",
    torch.__version__,
)

print(
    "torch.version.cuda   =",
    torch.version.cuda,
)

print(
    "CUDA_VISIBLE_DEVICES =",
    os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ),
)

print(
    "cuda available       =",
    torch.cuda.is_available(),
)

print(
    "cuda device count    =",
    torch.cuda.device_count(),
)


if torch.version.cuda is None:
    raise SystemExit(
        "[ERROR] PyTorch is a CPU-only build."
    )


if not torch.cuda.is_available():
    raise SystemExit(
        "[ERROR] CUDA is unavailable."
    )


if torch.cuda.device_count() < 1:
    raise SystemExit(
        "[ERROR] No CUDA GPU is visible."
    )


print(
    "device 0             =",
    torch.cuda.get_device_name(0),
)

print(
    "[CUDA PREFLIGHT] PASS"
)

PY


# =====================================================================
# Best-effort GPU monitor
# =====================================================================

if command -v nvidia-smi \
    >/dev/null 2>&1; then

    (
        while true; do

            nvidia-smi \
                --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
                --format=csv,noheader,nounits \
                >> "${GPU_RESOURCE_LOG}" \
                2>/dev/null \
                || true

            sleep 30

        done
    ) &

    GPU_MONITOR_PID=$!

fi


# =====================================================================
# Sequential multi-seed evaluation
# =====================================================================

echo "============================================================"
echo "[SLOW MWM ${MODE^^} START]"
echo "checkpoint=${FAST_PPO_CHECKPOINT}"
echo "output_root=${OUTPUT_ROOT}"
echo "============================================================"


for SEED in "${EVAL_SEEDS[@]}"; do

    export PYTHONHASHSEED="${SEED}"

    export FAST_PPO_SEED="${SEED}"

    export FAST_PPO_EVAL_EPISODES="${EVAL_EPISODES}"

    export FAST_PPO_EVAL_ROUNDS_PER_EPISODE="${EVAL_ROUNDS_PER_EPISODE}"

    export FAST_PPO_OUTPUT_ROOT="${OUTPUT_ROOT}"

    export FAST_PPO_RUN_NAME="seed${SEED}"


    SEED_CONSOLE_LOG="${JOB_LOG_ROOT}/slow-mwm-${MODE}-seed${SEED}.log"


    echo "------------------------------------------------------------"
    echo "[SEED START]"
    echo "seed=${SEED}"
    echo "run=${OUTPUT_ROOT}/seed${SEED}"
    echo "console=${SEED_CONSOLE_LOG}"
    echo "------------------------------------------------------------"


    srun \
        --ntasks=1 \
        --cpus-per-task="${ALLOCATED_CPUS}" \
        --cpu-bind=cores \
        --kill-on-bad-exit=1 \
        "${PYTHON_BIN}" \
        -u \
        -m \
        agent.PPO.fast.fast_train \
        2>&1 \
        | tee \
        "${SEED_CONSOLE_LOG}"


    ROUND_CSV="${OUTPUT_ROOT}/seed${SEED}/logs/eval_rounds.csv"

    EPISODE_CSV="${OUTPUT_ROOT}/seed${SEED}/logs/eval_episodes.csv"


    [[ -f "${ROUND_CSV}" ]] \
        || die \
        "Missing eval_rounds.csv for seed=${SEED}"

    [[ -f "${EPISODE_CSV}" ]] \
        || die \
        "Missing eval_episodes.csv for seed=${SEED}"


    echo "[SEED DONE] seed=${SEED}"

done


# =====================================================================
# Complete
# =====================================================================

echo "============================================================"
echo "[SLOW MWM ${MODE^^} DONE]"
echo "time=$(date --iso-8601=seconds)"
echo "output_root=${OUTPUT_ROOT}"
echo "============================================================"