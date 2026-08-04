#!/usr/bin/bash

#SBATCH -J fast-ppo
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=29G
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
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-8}"
REQUIRE_CLEAN_TREE="${REQUIRE_CLEAN_TREE:-1}"
ALLOW_EXISTING_RUN_DIR="${ALLOW_EXISTING_RUN_DIR:-0}"

JOB_LOG_ROOT="${PROJECT_ROOT}/logs/jobs/${JOB_ID}"
RUNTIME_CACHE_PARENT="${PROJECT_ROOT}/.runtime-cache/tmp"
TRAIN_CONSOLE_LOG="${JOB_LOG_ROOT}/train-console.log"
ENVIRONMENT_LOG="${JOB_LOG_ROOT}/environment.txt"
RESOLVED_CONFIG_LOG="${JOB_LOG_ROOT}/resolved-config.json"
GPU_RESOURCE_LOG="${JOB_LOG_ROOT}/gpu-resources.csv"

GPU_MONITOR_PID=""
RUNTIME_TMPDIR=""
die() {
    echo "[ERROR] $*" >&2
    exit 2
}

cleanup() {
    local exit_code="${1:-0}"
    set +e

    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi

    if [[ -n "${RUNTIME_TMPDIR}" && -d "${RUNTIME_TMPDIR}" ]]; then
        case "${RUNTIME_TMPDIR}" in
            "${RUNTIME_CACHE_PARENT}"/fast-ppo-"${JOB_ID}"-*)
                rm -rf -- "${RUNTIME_TMPDIR}"
                ;;
            *)
                echo "[WARN] Refusing to remove unexpected TMPDIR: ${RUNTIME_TMPDIR}" >&2
                ;;
        esac
    fi

    printf '[CLEANUP] exit_code=%s time=%s\n' \
        "${exit_code}" "$(date --iso-8601=seconds)"
}

trap 'exit_code=$?; trap - EXIT; cleanup "${exit_code}"; exit "${exit_code}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

[[ -d "${PROJECT_ROOT}" ]] || die "PROJECT_ROOT does not exist: ${PROJECT_ROOT}"
[[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]] \
    || die "Conda initialization script does not exist."
[[ -x "${PYTHON_BIN}" ]] || die "Python executable does not exist: ${PYTHON_BIN}"

cd "${PROJECT_ROOT}"

CURRENT_BRANCH="$(git branch --show-current)"
[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] \
    || die "Wrong branch: expected=${EXPECTED_BRANCH}, actual=${CURRENT_BRANCH}"

if [[ "${REQUIRE_CLEAN_TREE}" == "1" ]]; then
    git diff --quiet || die "Tracked working-tree changes exist. Commit or stash them first."
    git diff --cached --quiet || die "Staged but uncommitted changes exist."
fi

REQUESTED_PHASE="${FAST_PPO_PHASE:-pretrain}"
[[ "${REQUESTED_PHASE}" == "pretrain" ]] \
    || die "run/fast_train.sh only supports FAST_PPO_PHASE=pretrain."
export FAST_PPO_PHASE=pretrain

mkdir -p "${JOB_LOG_ROOT}" "${RUNTIME_CACHE_PARENT}"
RUNTIME_TMPDIR="$(mktemp -d "${RUNTIME_CACHE_PARENT}/fast-ppo-${JOB_ID}-XXXXXX")"
mkdir -p "${RUNTIME_TMPDIR}/matplotlib" "${RUNTIME_TMPDIR}/pycache"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"
hash -r

[[ "$(command -v python)" == "${PYTHON_BIN}" ]] \
    || die "Activated Python differs from ${PYTHON_BIN}: $(command -v python)"

export TMPDIR="${RUNTIME_TMPDIR}"
export MPLCONFIGDIR="${RUNTIME_TMPDIR}/matplotlib"
export PYTHONPYCACHEPREFIX="${RUNTIME_TMPDIR}/pycache"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${FAST_PPO_SEED:-2026}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# The environment transition loop is single-process in Fast pretraining.
# Avoid nested BLAS/OpenMP teams competing with it.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

"${PYTHON_BIN}" -m compileall -q \
    config.py \
    env \
    agent/PPO/config.py \
    agent/PPO/common \
    agent/PPO/fast

RUN_DIR="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from agent.PPO.config import get_fast_ppo_config

cfg = get_fast_ppo_config()
if cfg.phase != "pretrain" or cfg.mode != "train":
    raise SystemExit(f"Invalid Fast phase/mode: {cfg.phase}/{cfg.mode}")
if cfg.slow_decision_mode != "random":
    raise SystemExit(
        f"Fast pretraining requires random slow actions, got {cfg.slow_decision_mode!r}"
    )

root = Path.cwd()
output_root = Path(cfg.output_root)
if not output_root.is_absolute():
    output_root = root / output_root
print((output_root / str(cfg.run_name)).resolve())
PY
)"

if [[ "${ALLOW_EXISTING_RUN_DIR}" != "1" ]] \
    && [[ -d "${RUN_DIR}" ]] \
    && find "${RUN_DIR}" -mindepth 1 -maxdepth 2 -type f -print -quit | grep -q .; then
    die "Run directory already contains files: ${RUN_DIR}. Use a new FAST_PPO_RUN_NAME."
fi

export RESOLVED_CONFIG_LOG
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import torch

from agent.PPO.config import get_fast_ppo_config
from agent.PPO.fast.fast_train import build_env_config

cfg = get_fast_ppo_config()
env_cfg = build_env_config(cfg)

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA GPU is required; CPU fallback is not allowed.")
if int(cfg.rollout_slots) != int(env_cfg.slow_T):
    raise SystemExit(
        f"rollout_slots ({cfg.rollout_slots}) != slow_T ({env_cfg.slow_T})"
    )

resolved = {
    "phase": cfg.phase,
    "run_name": cfg.run_name,
    "output_root": cfg.output_root,
    "checkpoint": cfg.checkpoint,
    "resume": cfg.resume,
    "seed": cfg.seed,
    "num_episodes": cfg.num_episodes,
    "rounds_per_episode": cfg.rounds_per_episode,
    "rollout_slots": cfg.rollout_slots,
    "batch_size": cfg.batch_size,
    "update_epochs": cfg.update_epochs,
    "slow_decision_mode": cfg.slow_decision_mode,
    "device": cfg.device,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
}

path = Path(os.environ["RESOLVED_CONFIG_LOG"])
path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(resolved, indent=2, ensure_ascii=False))
PY

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
    echo "run_dir=${RUN_DIR}"
    echo "tmpdir=${TMPDIR}"
    echo
    git status --short
    echo
    scontrol show job "${SLURM_JOB_ID}" 2>/dev/null || true
} | tee "${ENVIRONMENT_LOG}"

nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    --loop=5 \
    > "${GPU_RESOURCE_LOG}" 2>&1 &
GPU_MONITOR_PID=$!

echo "============================================================"
echo "[FAST-PPO PRETRAIN START]"
echo "time      : $(date --iso-8601=seconds)"
echo "run_dir   : ${RUN_DIR}"
echo "console   : ${TRAIN_CONSOLE_LOG}"
echo "============================================================"

srun \
    --ntasks=1 \
    --cpus-per-task="${ALLOCATED_CPUS}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" -u -m agent.PPO.fast.fast_train \
    2>&1 | tee -a "${TRAIN_CONSOLE_LOG}"

echo "============================================================"
echo "[FAST-PPO PRETRAIN COMPLETED]"
echo "time      : $(date --iso-8601=seconds)"
echo "run_dir   : ${RUN_DIR}"
echo "============================================================"