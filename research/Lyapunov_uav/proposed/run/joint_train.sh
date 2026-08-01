#!/usr/bin/bash

#SBATCH -J joint-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

set -euo pipefail
umask 027

PROJECT_ROOT="/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed"
CONDA_ROOT="/data/surt321/anaconda3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/lab"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
EXPECTED_BRANCH="feat/no-hrl"

JOB_ID="${SLURM_JOB_ID:-manual}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

RUNTIME_CACHE_ROOT="${PROJECT_ROOT}/.runtime-cache"
MPL_CACHE_ROOT="${RUNTIME_CACHE_ROOT}/matplotlib"
TMP_CACHE_PARENT="${RUNTIME_CACHE_ROOT}/tmp"
JOB_LOG_ROOT="${PROJECT_ROOT}/logs/jobs/${JOB_ID}"

GPU_RESOURCE_LOG="${JOB_LOG_ROOT}/gpu-resources.csv"
CPU_RESOURCE_LOG="${JOB_LOG_ROOT}/cpu-resources.txt"
TRAIN_CONSOLE_LOG="${JOB_LOG_ROOT}/train-console.log"
ENVIRONMENT_LOG="${JOB_LOG_ROOT}/environment.txt"
RESOLVED_CONFIG_LOG="${JOB_LOG_ROOT}/resolved-config.json"
GIT_STATUS_LOG="${JOB_LOG_ROOT}/git-status.txt"
GIT_DIFF_LOG="${JOB_LOG_ROOT}/working-tree.patch"

GPU_MONITOR_PID=""
CPU_MONITOR_PID=""
RUNTIME_TMPDIR=""

# Set ALLOW_EXISTING_RUN_DIR=1 only when intentionally writing into an
# existing run directory. The default prevents accidental CSV/checkpoint
# mixing with a previous run that used the same config.run_name.
ALLOW_EXISTING_RUN_DIR="${ALLOW_EXISTING_RUN_DIR:-0}"

# ======================================================================
# Cleanup and signal handling
# ======================================================================
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

# ======================================================================
# Basic path and branch checks
# ======================================================================
if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "[ERROR] PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
    exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable does not exist: ${PYTHON_BIN}" >&2
    exit 127
fi

cd "${PROJECT_ROOT}"

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
    echo "[ERROR] Wrong Git branch." >&2
    echo "        expected=${EXPECTED_BRANCH}" >&2
    echo "        actual=${CURRENT_BRANCH}" >&2
    exit 2
fi

mkdir -p \
    "${MPL_CACHE_ROOT}" \
    "${TMP_CACHE_PARENT}" \
    "${JOB_LOG_ROOT}"

# Prefer a Slurm-provided node-local temporary directory when available.
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
    RUNTIME_TMPDIR="${SLURM_TMPDIR}/dpp-fastppo-${JOB_ID}"
    mkdir -p "${RUNTIME_TMPDIR}"
else
    RUNTIME_TMPDIR="$(
        mktemp -d \
            "${TMP_CACHE_PARENT}/dpp-fastppo-${JOB_ID}-XXXXXX"
    )"
fi

# ======================================================================
# Conda and process-level runtime settings
# ======================================================================
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"
hash -r

if [[ "$(command -v python)" != "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Activated Python is not the expected interpreter." >&2
    echo "        expected=${PYTHON_BIN}" >&2
    echo "        actual=$(command -v python)" >&2
    exit 2
fi

export MPLCONFIGDIR="${MPL_CACHE_ROOT}"
export TMPDIR="${RUNTIME_TMPDIR}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2026
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# deterministic_torch=True can require a deterministic cuBLAS workspace.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# The code already parallelizes forecast Env.step() with ThreadPoolExecutor.
# Prevent nested BLAS/OpenMP thread teams inside each forecast worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

# ======================================================================
# Reproducibility metadata
# ======================================================================
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
    echo "tmpdir=${TMPDIR}"
    echo "mplconfigdir=${MPLCONFIGDIR}"
    echo "ulimit_n=$(ulimit -n)"
    echo
    echo "[SLURM JOB]"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        scontrol show job "${SLURM_JOB_ID}" || true
    fi
    echo
    echo "[PARTITION]"
    if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
        scontrol show partition "${SLURM_JOB_PARTITION}" || true
    fi
    echo
    echo "[CPU AFFINITY]"
    taskset -pc $$ || true
    echo
    echo "[MEMORY]"
    free -h || true
    echo
    echo "[FILESYSTEM]"
    df -h "${PROJECT_ROOT}" "${TMPDIR}" || true
} | tee "${ENVIRONMENT_LOG}"

git status --short > "${GIT_STATUS_LOG}"
git diff --binary > "${GIT_DIFF_LOG}"

# ======================================================================
# Static compile check: only the final existing-file execution path
# ======================================================================
"${PYTHON_BIN}" -m compileall -q \
    config.py \
    env \
    agent/PPO/config.py \
    agent/PPO/common \
    agent/PPO/fast

# ======================================================================
# Resolve and validate the final joint-training configuration
# ======================================================================
RUN_DIR="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from agent.PPO.config import get_fast_ppo_config

cfg = get_fast_ppo_config()
root = Path.cwd()
output_root = Path(cfg.output_root)
if not output_root.is_absolute():
    output_root = root / output_root
run_name = str(cfg.run_name) if cfg.run_name is not None else "joint_dpp_fastppo"
print((output_root / run_name).resolve())
PY
)"

if [[ "${ALLOW_EXISTING_RUN_DIR}" != "1" ]] \
    && [[ -d "${RUN_DIR}" ]] \
    && find "${RUN_DIR}" -mindepth 1 -maxdepth 2 -type f -print -quit \
        | grep -q .; then
    echo "[ERROR] Resolved run directory already contains files:" >&2
    echo "        ${RUN_DIR}" >&2
    echo "        Use a new run_name in agent/PPO/config.py." >&2
    echo "        ALLOW_EXISTING_RUN_DIR=1 is only for deliberate reuse." >&2
    exit 2
fi

export ALLOCATED_CPUS
export RESOLVED_CONFIG_LOG

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from agent.PPO.config import get_fast_ppo_config
from agent.PPO.fast import fast_train
from env.env import Env

cfg = get_fast_ppo_config()
env_cfg = fast_train.build_env_config()
fast_train._assert_joint_config(cfg, env_cfg)

allocated_cpus = int(os.environ["ALLOCATED_CPUS"])
workers = int(cfg.dpp_forecast_workers)

if cfg.mode != "train":
    raise SystemExit(f"mode must be train, got {cfg.mode!r}")
if cfg.slow_decision_mode != "dpp":
    raise SystemExit(
        "slow_decision_mode must be 'dpp' for the final joint run, "
        f"got {cfg.slow_decision_mode!r}"
    )
checkpoint = None
checkpoint_strict_load = False
initialization = "from_scratch"

if cfg.checkpoint is not None:
    checkpoint = fast_train._resolve_checkpoint(cfg.checkpoint)

    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit(
            f"Checkpoint does not exist: {checkpoint}"
        )

    checkpoint_strict_load = True
    initialization = "checkpoint"
if workers < 1:
    raise SystemExit("dpp_forecast_workers must be at least 1.")
if workers > max(1, allocated_cpus - 2):
    raise SystemExit(
        "dpp_forecast_workers leaves insufficient CPU capacity for the "
        "main Python/GPU thread and monitoring process: "
        f"workers={workers}, allocated_cpus={allocated_cpus}"
    )
if int(cfg.dpp_candidate_batch_size) < workers:
    raise SystemExit(
        "dpp_candidate_batch_size should not be smaller than "
        f"dpp_forecast_workers: batch={cfg.dpp_candidate_batch_size}, "
        f"workers={workers}"
    )



# Verify that the patched fast_train.py, not the obsolete joint/ entrypoint,
# contains the finalized Slow-DPP selector.
if not hasattr(fast_train, "select_slow_action_dpp"):
    raise SystemExit(
        "Patched fast_train.py is missing select_slow_action_dpp()."
    )

# CUDA and strict checkpoint compatibility check.
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required; CPU fallback is not allowed.")
if torch.cuda.device_count() < 1:
    raise SystemExit("No visible CUDA device was allocated.")

probe_env = Env(env_cfg)
probe_obs, _ = probe_env.reset()
ppo_cfg = fast_train.build_agent_ppo_config(cfg)
probe_agent = fast_train._initialize_agent(
    probe_env,
    probe_obs,
    cfg,
    ppo_cfg,
)

parameter_count = sum(
    parameter.numel() for parameter in probe_agent.model.parameters()
)

resolved = {
    "fast_train_file": str(Path(fast_train.__file__).resolve()),
    "mode": cfg.mode,
    "slow_decision_mode": cfg.slow_decision_mode,
    "run_name": cfg.run_name,
    "initialization": initialization,
    "checkpoint": (
        None if checkpoint is None else str(checkpoint)
    ),
    "checkpoint_strict_load": checkpoint_strict_load,
    "device": str(probe_agent.device),
    "gpu_name": torch.cuda.get_device_name(0),
    "parameter_count": int(parameter_count),
    "num_episodes": int(cfg.num_episodes),
    "rounds_per_episode": int(cfg.rounds_per_episode),
    "slow_T": int(env_cfg.slow_T),
    "rollout_slots": int(cfg.rollout_slots),
    "batch_size": int(cfg.batch_size),
    "update_epochs": int(cfg.update_epochs),
    "dpp_forecast_horizon": int(cfg.dpp_forecast_horizon),
    "dpp_forecast_scenarios": int(cfg.dpp_forecast_scenarios),
    "dpp_candidate_batch_size": int(cfg.dpp_candidate_batch_size),
    "dpp_forecast_workers": workers,
    "dpp_coordinate_sweeps": int(cfg.dpp_coordinate_sweeps),
    "mobility_curriculum": list(cfg.mobility_curriculum),
    "allocated_cpus": allocated_cpus,
}

with Path(os.environ["RESOLVED_CONFIG_LOG"]).open(
    "w", encoding="utf-8"
) as handle:
    json.dump(resolved, handle, indent=2, ensure_ascii=False)

print(json.dumps(resolved, indent=2, ensure_ascii=False))

# Release the preflight CUDA allocation before the actual training process.
del probe_agent, probe_env
torch.cuda.empty_cache()
PY

# ======================================================================
# Runtime resource monitors
# ======================================================================
nvidia-smi \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    --loop=5 \
    > "${GPU_RESOURCE_LOG}" 2>&1 &
GPU_MONITOR_PID=$!

(
    echo "timestamp job_step ave_cpu ave_rss max_rss max_vmsize"
    while true; do
        printf '%s ' "$(date --iso-8601=seconds)"
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            sstat \
                --jobs="${SLURM_JOB_ID}" \
                --noheader \
                --parsable2 \
                --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize \
                2>/dev/null \
                | tr '\n' ';' \
                || true
        else
            printf 'manual-run'
        fi
        printf '\n'
        sleep 30
    done
) > "${CPU_RESOURCE_LOG}" 2>&1 &
CPU_MONITOR_PID=$!

# ======================================================================
# Final entrypoint: Slow-DPP + Fast-PPO alternating training
# ======================================================================
echo "============================================================"
echo "[SLOW-DPP + FAST-PPO FULL TRAIN]"
echo "start_time : $(date --iso-8601=seconds)"
echo "project    : ${PROJECT_ROOT}"
echo "run_dir    : ${RUN_DIR}"
echo "python     : ${PYTHON_BIN}"
echo "cpus       : ${ALLOCATED_CPUS}"
echo "gpu_log    : ${GPU_RESOURCE_LOG}"
echo "cpu_log    : ${CPU_RESOURCE_LOG}"
echo "============================================================"

# srun enforces the Slurm CPU/GPU allocation and core affinity for the
# actual training process. The final code path is fast_train.py; obsolete
# agent/PPO/joint and agent/PPO/slow test/entrypoint modules are not used.
srun \
    --ntasks=1 \
    --cpus-per-task="${ALLOCATED_CPUS}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" -u -m agent.PPO.fast.fast_train \
    2>&1 | tee -a "${TRAIN_CONSOLE_LOG}"

echo "============================================================"
echo "[TRAIN COMPLETED]"
echo "end_time   : $(date --iso-8601=seconds)"
echo "run_dir    : ${RUN_DIR}"
echo "============================================================"