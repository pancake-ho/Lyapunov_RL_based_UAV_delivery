#!/usr/bin/bash

#SBATCH -J fast-h2-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH --exclude=moana-y5
#SBATCH -o logs/slurm-fast-sweep-%A.out

set -euo pipefail
umask 027

# =====================================================================
# Do NOT inherit old experiment-specific exports
# =====================================================================
#
# This launcher is self-contained.
# Old login-shell experiment variables must not affect this sweep.
#
# IMPORTANT:
# CUDA_VISIBLE_DEVICES must NOT be unset because Slurm owns it.
# =====================================================================

while IFS='=' read -r name _; do
    case "${name}" in
        FAST_PPO_*|\
        FAST_CHECKPOINT_SWEEP_*|\
        FAST_PRETRAIN_*|\
        JOINT_*|\
        FAST_H1_RUN_ROOT)
            unset "${name}"
            ;;
    esac
done < <(env)


# =====================================================================
# Fixed experiment paths
# =====================================================================

PROJECT_ROOT="/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed"

CONDA_ROOT="/data/surt321/anaconda3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/lab"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"

EXPECTED_BRANCH="feat/no-hrl"

FAST_RUN_ROOT="${PROJECT_ROOT}/fast/fast_pretrain_h2_seed2026_ep400_v1"

JOB_ID="${SLURM_JOB_ID:-manual}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-16}"


# =====================================================================
# Sweep mode
# =====================================================================
#
# Usage:
#
#   sbatch run/fast_checkpoint_sweep.sh smoke
#
# or
#
#   sbatch run/fast_checkpoint_sweep.sh full
#
# No manual export is required.
# =====================================================================

MODE="${1:-smoke}"

case "${MODE}" in

    smoke)
        CHECKPOINT_EPISODES=(
            305
        )

        EVAL_SEEDS=(
            2026
            2027
        )

        EVAL_EPISODES=1
        EVAL_ROUNDS=1
        ;;

    full)
        CHECKPOINT_EPISODES=(
            100
            150
            200
            225
            250
            275
            300
            305
        )

        EVAL_SEEDS=(
            2026
            2027
            2028
        )

        EVAL_EPISODES=5
        EVAL_ROUNDS=5
        ;;

    *)
        echo "[ERROR] Unknown mode: ${MODE}" >&2
        echo "Use: smoke or full" >&2
        exit 2
        ;;
esac


# =====================================================================
# Fixed evaluation criteria
# =====================================================================

WORKLOAD_TOLERANCE="0.02"
MINIMUM_ALLOWED_SOC="19.95"

# Every sbatch job gets an independent directory.
# This prevents a failed previous evaluation from contaminating a rerun.
SWEEP_OUTPUT="${FAST_RUN_ROOT}/checkpoint_sweep_h2_${MODE}_job${JOB_ID}"


# =====================================================================
# Helpers
# =====================================================================

die() {
    echo "[ERROR] $*" >&2
    exit 2
}


# =====================================================================
# Static checks
# =====================================================================

[[ -d "${PROJECT_ROOT}" ]] \
    || die "Project root not found: ${PROJECT_ROOT}"

[[ -d "${FAST_RUN_ROOT}" ]] \
    || die "Fast run root not found: ${FAST_RUN_ROOT}"

[[ -d "${FAST_RUN_ROOT}/checkpoints" ]] \
    || die "Checkpoint directory not found: ${FAST_RUN_ROOT}/checkpoints"

[[ -x "${PYTHON_BIN}" ]] \
    || die "Python not found: ${PYTHON_BIN}"

[[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]] \
    || die "Conda initialization script not found."


cd "${PROJECT_ROOT}"


# =====================================================================
# Git contract
# =====================================================================

CURRENT_BRANCH="$(git branch --show-current)"

[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] \
    || die \
    "Wrong branch: expected=${EXPECTED_BRANCH}, actual=${CURRENT_BRANCH}"

[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
    || die \
    "Working tree is not clean. Commit or stash source changes first."


# =====================================================================
# Conda
# =====================================================================

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

conda activate "${CONDA_ENV_PATH}"

hash -r

[[ "$(command -v python)" == "${PYTHON_BIN}" ]] \
    || die \
    "Activated Python differs from ${PYTHON_BIN}: $(command -v python)"


# =====================================================================
# Runtime environment
# =====================================================================

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2026

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

# Only the runtime properties required by Fast evaluation are set here.
# checkpoint / seed / run name / phase are supplied by
# fast_checkpoint_sweep.py independently for each evaluation subprocess.
export FAST_PPO_DEVICE="cuda"
export FAST_PPO_DETERMINISTIC_TORCH=1


# =====================================================================
# Allocation information
# =====================================================================

echo "============================================================"
echo "[SLURM GPU ALLOCATION]"
echo "hostname=${HOSTNAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NOT_SET}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-NOT_SET}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-NOT_SET}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NOT_SET}"
echo "============================================================"


# =====================================================================
# CUDA preflight
# =====================================================================
#
# Do not use nvidia-smi as the acceptance criterion.
# The actual experiment uses PyTorch, so PyTorch CUDA visibility is the
# authoritative preflight.
#
# Run the check inside a Slurm job step, matching the real sweep execution.
# =====================================================================

echo "[CUDA PREFLIGHT]"

srun \
    --ntasks=1 \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" - <<'PY'

import os
import socket
import sys

import torch


print("hostname             =", socket.gethostname())
print("torch                =", torch.__version__)
print("torch.version.cuda   =", torch.version.cuda)

print(
    "CUDA_VISIBLE_DEVICES =",
    os.environ.get("CUDA_VISIBLE_DEVICES"),
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
        "[ERROR] Installed PyTorch is a CPU-only build."
    )


if not torch.cuda.is_available():
    raise SystemExit(
        "[ERROR] CUDA is unavailable inside the Slurm GPU job step."
    )


if torch.cuda.device_count() < 1:
    raise SystemExit(
        "[ERROR] No CUDA GPU is visible inside the Slurm GPU job step."
    )


print(
    "device 0             =",
    torch.cuda.get_device_name(0),
)

print("[CUDA PREFLIGHT] PASS")

PY


# =====================================================================
# Compile / unit tests
# =====================================================================

"${PYTHON_BIN}" -m py_compile \
    agent/PPO/fast/fast_train.py \
    agent/PPO/fast/fast_checkpoint_sweep.py


"${PYTHON_BIN}" -m unittest -v \
    agent.PPO.fast.test_fast_checkpoint_sweep \
    agent.PPO.fast.test_fast_pretrain_contract


# =====================================================================
# Verify requested checkpoints
# =====================================================================

for episode in "${CHECKPOINT_EPISODES[@]}"; do

    checkpoint="${FAST_RUN_ROOT}/checkpoints/fast_ppo_pretrain_ep${episode}.pt"

    [[ -f "${checkpoint}" ]] \
        || die "Checkpoint not found: ${checkpoint}"

done


# =====================================================================
# Build sweep command
# =====================================================================

COMMAND=(
    "${PYTHON_BIN}"
    -u
    -m
    agent.PPO.fast.fast_checkpoint_sweep

    --checkpoint-dir
    "${FAST_RUN_ROOT}/checkpoints"

    --output-dir
    "${SWEEP_OUTPUT}"

    --episodes
    "${CHECKPOINT_EPISODES[@]}"

    --eval-episodes
    "${EVAL_EPISODES}"

    --eval-rounds-per-episode
    "${EVAL_ROUNDS}"

    --minimum-allowed-soc
    "${MINIMUM_ALLOWED_SOC}"

    --workload-relative-tolerance
    "${WORKLOAD_TOLERANCE}"

    --seeds
    "${EVAL_SEEDS[@]}"
)


# =====================================================================
# Run information
# =====================================================================

echo "============================================================"
echo "[SWEEP START]"
echo "mode                = ${MODE}"
echo "hostname            = $(hostname)"
echo "job_id              = ${JOB_ID}"
echo "fast_run_root       = ${FAST_RUN_ROOT}"
echo "output              = ${SWEEP_OUTPUT}"
echo "checkpoints         = ${CHECKPOINT_EPISODES[*]}"
echo "eval_seeds          = ${EVAL_SEEDS[*]}"
echo "eval_episodes       = ${EVAL_EPISODES}"
echo "eval_rounds         = ${EVAL_ROUNDS}"
echo "git_branch          = ${CURRENT_BRANCH}"
echo "git_commit          = $(git rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES:-NOT_SET}"
echo "============================================================"


# =====================================================================
# Actual checkpoint sweep
# =====================================================================
#
# Match the execution style already proven by run/fast_train.sh:
# batch allocation -> srun job step -> Python.
# =====================================================================

srun \
    --ntasks=1 \
    --cpus-per-task="${ALLOCATED_CPUS}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${COMMAND[@]}"


# =====================================================================
# Complete
# =====================================================================

echo "============================================================"
echo "[SWEEP DONE]"
echo "time      = $(date --iso-8601=seconds)"
echo "output    = ${SWEEP_OUTPUT}"
echo "decision  = ${SWEEP_OUTPUT}/selection.json"
echo "============================================================"