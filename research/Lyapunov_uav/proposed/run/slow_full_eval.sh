#!/usr/bin/bash

#SBATCH -J slow-full-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH --exclude=moana-y5
#SBATCH -o logs/slurm-slow-full-%A.out

set -euo pipefail
umask 027


# =====================================================================
# Fixed paths
# =====================================================================

PROJECT_ROOT="/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed"

CONDA_ROOT="/data/surt321/anaconda3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/lab"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"

EXPECTED_BRANCH="feat/no-hrl"

SELECTED_FAST_CHECKPOINT="${PROJECT_ROOT}/fast/fast_pretrain_h2_seed2026_ep400_v1/checkpoints/fast_ppo_pretrain_ep300.pt"

JOB_ID="${SLURM_JOB_ID:-manual}"
ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-16}"

SLOW_T=3600
SLOT_LOG_STRIDE=1


die() {
    echo "[ERROR] $*" >&2
    exit 2
}


if (( ALLOCATED_CPUS < 2 )); then
    die "At least 2 CPUs are required."
fi

DPP_WORKERS=$((ALLOCATED_CPUS - 1))


# =====================================================================
# Arguments
#
# $1 METHOD:
#       random | rsu_only | full_dpp
#
# $2 TRACE_MODE:
#       round | slot
#
# $3 SNR_OFFSET_DB:
#       e.g. -10, -5, 0, 5, 10
#
# $4 PROFILE:
#       full   -> 5 seeds
#       single -> seed 2026 only
# =====================================================================

METHOD="${1:-}"
TRACE_MODE="${2:-round}"
SNR_OFFSET_DB="${3:-0}"
PROFILE="${4:-full}"


case "${METHOD}" in
    random)
        FAST_PHASE="eval_pretrain"
        DPP_ENABLE_UAV="1"
        ;;

    rsu_only)
        FAST_PHASE="eval_joint"
        DPP_ENABLE_UAV="0"
        ;;

    full_dpp)
        FAST_PHASE="eval_joint"
        DPP_ENABLE_UAV="1"
        ;;

    *)
        echo "Usage:" >&2
        echo "  sbatch run/slow_full_eval.sh METHOD [round|slot] [SNR_OFFSET_DB] [full|single]" >&2
        echo >&2
        echo "Examples:" >&2
        echo "  sbatch run/slow_full_eval.sh full_dpp round 0 full" >&2
        echo "  sbatch run/slow_full_eval.sh full_dpp slot 0 single" >&2
        echo "  sbatch run/slow_full_eval.sh full_dpp round -10 single" >&2
        exit 2
        ;;
esac


case "${TRACE_MODE}" in
    round)
        SLOT_LOGGING=0
        ;;

    slot)
        SLOT_LOGGING=1
        ;;

    *)
        die "TRACE_MODE must be round or slot."
        ;;
esac


case "${PROFILE}" in
    full)
        EVAL_SEEDS=(
            2026
            2027
            2028
            2029
            2030
        )
        ;;

    single)
        EVAL_SEEDS=(
            2026
        )
        ;;

    *)
        die "PROFILE must be full or single."
        ;;
esac


EVAL_EPISODES=5
EVAL_ROUNDS_PER_EPISODE=10


# =====================================================================
# SNR tag
# =====================================================================

if [[ "${SNR_OFFSET_DB}" == -* ]]; then
    SNR_TAG="m${SNR_OFFSET_DB#-}"
else
    SNR_TAG="p${SNR_OFFSET_DB#+}"
fi

SNR_TAG="${SNR_TAG//./p}"


# =====================================================================
# Paths
# =====================================================================

OUTPUT_ROOT="${PROJECT_ROOT}/eval/full_ep300_${METHOD}_${PROFILE}_${TRACE_MODE}_snr${SNR_TAG}db_job${JOB_ID}"

RUNTIME_CACHE_ROOT="${PROJECT_ROOT}/.runtime-cache"
MPL_CACHE_ROOT="${RUNTIME_CACHE_ROOT}/matplotlib"
TMP_CACHE_PARENT="${RUNTIME_CACHE_ROOT}/tmp"

JOB_LOG_ROOT="${PROJECT_ROOT}/logs/jobs/${JOB_ID}/${METHOD}_${PROFILE}_${TRACE_MODE}_snr${SNR_TAG}db"

ENVIRONMENT_LOG="${JOB_LOG_ROOT}/environment.txt"
GPU_RESOURCE_LOG="${JOB_LOG_ROOT}/gpu-resources.csv"

RUNTIME_TMPDIR=""
GPU_MONITOR_PID=""


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
# Clear stale experiment variables
# =====================================================================

while IFS='=' read -r name _; do
    case "${name}" in
        FAST_PPO_*|\
        FAST_PRETRAIN_*|\
        JOINT_*)
            unset "${name}"
            ;;
    esac
done < <(env)

unset ALLOW_EXISTING_RUN_DIR || true


# =====================================================================
# Static checks
# =====================================================================

[[ -d "${PROJECT_ROOT}" ]] \
    || die "Project root not found."

[[ -x "${PYTHON_BIN}" ]] \
    || die "Python not found: ${PYTHON_BIN}"

[[ -f "${SELECTED_FAST_CHECKPOINT}" ]] \
    || die \
    "Fast checkpoint not found: ${SELECTED_FAST_CHECKPOINT}"


cd "${PROJECT_ROOT}"


CURRENT_BRANCH="$(git branch --show-current)"

[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] \
    || die \
    "Wrong branch: expected=${EXPECTED_BRANCH}, actual=${CURRENT_BRANCH}"

[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
    || die \
    "Working tree must be clean."


# =====================================================================
# Conda
# =====================================================================

mkdir -p \
    "${MPL_CACHE_ROOT}" \
    "${TMP_CACHE_PARENT}" \
    "${JOB_LOG_ROOT}" \
    "${OUTPUT_ROOT}"

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

conda activate "${CONDA_ENV_PATH}"

hash -r

[[ "$(command -v python)" == "${PYTHON_BIN}" ]] \
    || die \
    "Unexpected Python: $(command -v python)"


# Validate SNR argument numerically.

"${PYTHON_BIN}" - "${SNR_OFFSET_DB}" <<'PY'
import math
import sys

value = float(sys.argv[1])

if not math.isfinite(value):
    raise SystemExit(
        "[ERROR] SNR offset must be finite."
    )

print(
    f"[CONFIG] common SNR offset = {value:g} dB"
)
PY


# =====================================================================
# Runtime dirs
# =====================================================================

if [[ -n "${SLURM_TMPDIR:-}" \
    && -d "${SLURM_TMPDIR}" ]]; then

    RUNTIME_TMPDIR="${SLURM_TMPDIR}/slow-full-${METHOD}-${JOB_ID}"

    mkdir -p \
        "${RUNTIME_TMPDIR}"

else

    RUNTIME_TMPDIR="$(
        mktemp -d \
        "${TMP_CACHE_PARENT}/slow-full-${METHOD}-${JOB_ID}-XXXXXX"
    )"

fi


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


# =====================================================================
# Evaluation contract
# =====================================================================

export FAST_PPO_PHASE="${FAST_PHASE}"

export FAST_PPO_CHECKPOINT="${SELECTED_FAST_CHECKPOINT}"

export FAST_PPO_DPP_ENABLE_UAV="${DPP_ENABLE_UAV}"

export FAST_PPO_DPP_FORECAST_WORKERS="${DPP_WORKERS}"

export FAST_PPO_DETERMINISTIC_TORCH=1
export FAST_PPO_DEVICE="cuda"
export FAST_PPO_FAIL_IF_CUDA_UNAVAILABLE=1
export FAST_PPO_AUDIT_RUNTIME_INVARIANTS=0

# -------------------------------------------------------------
# Plot / diagnostic options
# -------------------------------------------------------------

export FAST_PPO_EVAL_SLOT_LOGGING="${SLOT_LOGGING}"

export FAST_PPO_EVAL_SLOT_LOG_STRIDE="${SLOT_LOG_STRIDE}"

export FAST_PPO_CHANNEL_SNR_OFFSET_DB="${SNR_OFFSET_DB}"

export ALLOW_EXISTING_RUN_DIR=0


# =====================================================================
# Provenance
# =====================================================================

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"

    echo "job_id=${JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-NOT_SET}"

    echo "git_branch=${CURRENT_BRANCH}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_describe=$(git describe --always --dirty --tags 2>/dev/null || true)"

    echo "method=${METHOD}"
    echo "profile=${PROFILE}"

    echo "fast_phase=${FAST_PHASE}"
    echo "dpp_enable_uav=${DPP_ENABLE_UAV}"

    echo "trace_mode=${TRACE_MODE}"
    echo "slot_logging=${SLOT_LOGGING}"
    echo "slot_log_stride=${SLOT_LOG_STRIDE}"

    echo "snr_offset_db=${SNR_OFFSET_DB}"

    echo "checkpoint=${SELECTED_FAST_CHECKPOINT}"

    echo "eval_seeds=${EVAL_SEEDS[*]}"
    echo "eval_episodes=${EVAL_EPISODES}"
    echo "eval_rounds_per_episode=${EVAL_ROUNDS_PER_EPISODE}"

    echo "slow_T=${SLOW_T}"

    echo "dpp_workers=${DPP_WORKERS}"

    echo "output_root=${OUTPUT_ROOT}"

} | tee "${ENVIRONMENT_LOG}"


# =====================================================================
# Compile / unit tests
# =====================================================================

"${PYTHON_BIN}" -m py_compile \
    env/env.py \
    env/delivery/rsu_delivery.py \
    env/delivery/uav_delivery.py \
    agent/PPO/config.py \
    agent/PPO/slow/slow_matching.py \
    agent/PPO/slow/test_fast_slow.py \
    agent/PPO/fast/fast_train.py \
    agent/PPO/fast/plot_eval_diagnostics.py \
    agent/PPO/fast/plot_slot_diagnostics.py


"${PYTHON_BIN}" -m unittest -v \
    env.delivery.test_rsu_delivery_crn \
    agent.PPO.slow.test_fast_slow \
    agent.PPO.fast.test_fast_pretrain_contract


# =====================================================================
# Config preflight
# =====================================================================

"${PYTHON_BIN}" - <<'PY'
from agent.PPO.config import get_fast_ppo_config
from agent.PPO.fast.fast_train import build_env_config

cfg = get_fast_ppo_config()
env_cfg = build_env_config(cfg)

print("[EVAL CONFIG]")
print("phase =", cfg.phase)
print("mode =", cfg.mode)
print("slot logging =", cfg.eval_slot_logging)
print("slot stride =", cfg.eval_slot_log_stride)
print("snr offset dB =", cfg.channel_snr_offset_db)

print(
    "RSU gamma_db =",
    env_cfg.rsu_channel.gamma_db,
)

print(
    "UAV beta_zero =",
    env_cfg.uav_channel.beta_zero,
)
PY


# =====================================================================
# CUDA preflight
# =====================================================================

srun \
    --ntasks=1 \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${PYTHON_BIN}" - <<'PY'

import os
import socket
import torch

print("hostname =", socket.gethostname())
print("torch =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print(
    "CUDA_VISIBLE_DEVICES =",
    os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ),
)

print(
    "cuda available =",
    torch.cuda.is_available(),
)

print(
    "cuda device count =",
    torch.cuda.device_count(),
)

if torch.version.cuda is None:
    raise SystemExit(
        "[ERROR] PyTorch is CPU-only."
    )

if not torch.cuda.is_available():
    raise SystemExit(
        "[ERROR] CUDA unavailable."
    )

if torch.cuda.device_count() < 1:
    raise SystemExit(
        "[ERROR] No CUDA device."
    )

print(
    "device 0 =",
    torch.cuda.get_device_name(0),
)

print(
    "[CUDA PREFLIGHT] PASS"
)

PY


# =====================================================================
# GPU monitoring
# =====================================================================

if command -v nvidia-smi >/dev/null 2>&1; then

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
# Evaluation
# =====================================================================

echo "============================================================"
echo "[SLOW EVALUATION START]"
echo "method=${METHOD}"
echo "profile=${PROFILE}"
echo "trace=${TRACE_MODE}"
echo "snr_offset_db=${SNR_OFFSET_DB}"
echo "commit=$(git rev-parse HEAD)"
echo "checkpoint=${SELECTED_FAST_CHECKPOINT}"
echo "============================================================"

EXPECTED_ROUND_ROWS=$((EVAL_EPISODES * EVAL_ROUNDS_PER_EPISODE))

EXPECTED_SLOT_ROWS_PER_ROUND=$(((SLOW_T + SLOT_LOG_STRIDE - 1) / SLOT_LOG_STRIDE))

EXPECTED_SLOT_ROWS=$((EXPECTED_ROUND_ROWS * EXPECTED_SLOT_ROWS_PER_ROUND))

echo "============================================================"
echo "[ROW-COUNT PREFLIGHT]"
echo "eval_episodes=${EVAL_EPISODES}"
echo "eval_rounds_per_episode=${EVAL_ROUNDS_PER_EPISODE}"
echo "slow_T=${SLOW_T}"
echo "slot_log_stride=${SLOT_LOG_STRIDE}"
echo "expected_round_rows=${EXPECTED_ROUND_ROWS}"
echo "expected_slot_rows_per_round=${EXPECTED_SLOT_ROWS_PER_ROUND}"
echo "expected_slot_rows=${EXPECTED_SLOT_ROWS}"
echo "============================================================"


[[ "${EXPECTED_ROUND_ROWS}" -gt 0 ]] \
    || die \
    "EXPECTED_ROUND_ROWS must be positive."


[[ "${EXPECTED_SLOT_ROWS_PER_ROUND}" -gt 0 ]] \
    || die \
    "EXPECTED_SLOT_ROWS_PER_ROUND must be positive."


[[ "${EXPECTED_SLOT_ROWS}" -gt 0 ]] \
    || die \
    "EXPECTED_SLOT_ROWS must be positive."


for SEED in "${EVAL_SEEDS[@]}"; do

    export PYTHONHASHSEED="${SEED}"

    export FAST_PPO_SEED="${SEED}"

    export FAST_PPO_EVAL_EPISODES="${EVAL_EPISODES}"

    export FAST_PPO_EVAL_ROUNDS_PER_EPISODE="${EVAL_ROUNDS_PER_EPISODE}"

    export FAST_PPO_OUTPUT_ROOT="${OUTPUT_ROOT}"

    export FAST_PPO_RUN_NAME="seed${SEED}"

    SEED_LOG="${JOB_LOG_ROOT}/full-${METHOD}-seed${SEED}.log"

    echo "------------------------------------------------------------"
    echo "[SEED START] ${SEED}"
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
        | tee "${SEED_LOG}"


    ROUND_CSV="${OUTPUT_ROOT}/seed${SEED}/logs/eval_rounds.csv"

    EPISODE_CSV="${OUTPUT_ROOT}/seed${SEED}/logs/eval_episodes.csv"


    [[ -f "${ROUND_CSV}" ]] \
        || die \
        "Missing round CSV for seed=${SEED}"

    [[ -f "${EPISODE_CSV}" ]] \
        || die \
        "Missing episode CSV for seed=${SEED}"


    ROUND_ROW_COUNT="$(
        tail -n +2 \
        "${ROUND_CSV}" \
        | wc -l
    )"


    [[ "${ROUND_ROW_COUNT}" -eq "${EXPECTED_ROUND_ROWS}" ]] \
        || die \
        "Expected ${EXPECTED_ROUND_ROWS} rounds for seed=${SEED}, got=${ROUND_ROW_COUNT}"


    if [[ "${SLOT_LOGGING}" -eq 1 ]]; then

        SLOT_CSV="${OUTPUT_ROOT}/seed${SEED}/logs/eval_slots.csv"

        [[ -f "${SLOT_CSV}" ]] \
            || die \
            "Missing slot CSV for seed=${SEED}"

        SLOT_ROW_COUNT="$(
            tail -n +2 \
            "${SLOT_CSV}" \
            | wc -l
        )"

        [[ "${SLOT_ROW_COUNT}" -eq "${EXPECTED_SLOT_ROWS}" ]] \
            || die \
            "Expected ${EXPECTED_SLOT_ROWS} slot rows for seed=${SEED}, got=${SLOT_ROW_COUNT}"

    fi


    echo "[SEED DONE] ${SEED}"

done


echo "============================================================"
echo "[SLOW EVALUATION DONE]"
echo "method=${METHOD}"
echo "profile=${PROFILE}"
echo "trace=${TRACE_MODE}"
echo "snr_offset_db=${SNR_OFFSET_DB}"
echo "time=$(date --iso-8601=seconds)"
echo "output=${OUTPUT_ROOT}"
echo "============================================================"