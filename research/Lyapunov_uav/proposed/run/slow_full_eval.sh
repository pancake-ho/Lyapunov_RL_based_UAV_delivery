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

if (( ALLOCATED_CPUS < 2 )); then
    echo "[ERROR] At least 2 CPUs are required." >&2
    exit 2
fi

DPP_WORKERS=$((ALLOCATED_CPUS - 1))


# =====================================================================
# Method
# =====================================================================

METHOD="${1:-}"
TRACE_MODE="${2:-round}"
SNR_OFFSET_DB="${3:-0}"

case "${TRACE_MODE}" in
    round)
        SLOT_LOGGING=0
        ;;
    slot)
        SLOT_LOGGING=1
        ;;
    *)
        echo "[ERROR] Unknown trace mode: ${TRACE_MODE}" >&2
        echo "Use round or slot." >&2
        exit 2
        ;;
esac
SNR_TAG="$(
    "${PYTHON_BIN}" - "${SNR_OFFSET_DB}" <<'PY'
import sys

value = float(sys.argv[1])

sign = (
    "p"
    if value >= 0.0
    else "m"
)

text = (
    f"{abs(value):g}"
    .replace(".", "p")
)

print(
    f"{sign}{text}"
)
PY
)"

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
        echo "[ERROR] Unknown method: ${METHOD}" >&2
        echo "Usage:" >&2
        echo "  sbatch run/slow_full_eval.sh random" >&2
        echo "  sbatch run/slow_full_eval.sh rsu_only" >&2
        echo "  sbatch run/slow_full_eval.sh full_dpp" >&2
        exit 2
        ;;
esac


# =====================================================================
# Final frozen-Fast evaluation contract
# =====================================================================

EVAL_SEEDS=(
    2026
    2027
    2028
    2029
    2030
)

EVAL_EPISODES=5
EVAL_ROUNDS_PER_EPISODE=10

OUTPUT_ROOT="${PROJECT_ROOT}/eval/full_ep300_${METHOD}_job${JOB_ID}"

RUNTIME_CACHE_ROOT="${PROJECT_ROOT}/.runtime-cache"
MPL_CACHE_ROOT="${RUNTIME_CACHE_ROOT}/matplotlib"
TMP_CACHE_PARENT="${RUNTIME_CACHE_ROOT}/tmp"

JOB_LOG_ROOT="${PROJECT_ROOT}/logs/jobs/${JOB_ID}"
ENVIRONMENT_LOG="${JOB_LOG_ROOT}/environment.txt"
GPU_RESOURCE_LOG="${JOB_LOG_ROOT}/gpu-resources.csv"

RUNTIME_TMPDIR=""
GPU_MONITOR_PID=""


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
# Clear stale experiment exports
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
    || die "Python not found."

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
# Conda / runtime dirs
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


if [[ -n "${SLURM_TMPDIR:-}" \
    && -d "${SLURM_TMPDIR}" ]]; then

    RUNTIME_TMPDIR="${SLURM_TMPDIR}/slow-full-${METHOD}-${JOB_ID}"

    mkdir -p "${RUNTIME_TMPDIR}"

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
# Evaluation environment
# =====================================================================

export FAST_PPO_PHASE="${FAST_PHASE}"

export FAST_PPO_CHECKPOINT="${SELECTED_FAST_CHECKPOINT}"

export FAST_PPO_DPP_ENABLE_UAV="${DPP_ENABLE_UAV}"

export FAST_PPO_DPP_FORECAST_WORKERS="${DPP_WORKERS}"

export FAST_PPO_DETERMINISTIC_TORCH=1
export FAST_PPO_DEVICE="cuda"
export FAST_PPO_FAIL_IF_CUDA_UNAVAILABLE=1
export FAST_PPO_AUDIT_RUNTIME_INVARIANTS=0

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
    echo "fast_phase=${FAST_PHASE}"

    echo "dpp_enable_uav=${DPP_ENABLE_UAV}"

    echo "checkpoint=${SELECTED_FAST_CHECKPOINT}"

    echo "eval_seeds=${EVAL_SEEDS[*]}"
    echo "eval_episodes=${EVAL_EPISODES}"
    echo "eval_rounds_per_episode=${EVAL_ROUNDS_PER_EPISODE}"

    echo "dpp_workers=${DPP_WORKERS}"

    echo "output_root=${OUTPUT_ROOT}"

} | tee "${ENVIRONMENT_LOG}"


# =====================================================================
# Compile / unit tests
# =====================================================================

"${PYTHON_BIN}" -m py_compile \
    env/env.py \
    env/delivery/rsu_delivery.py \
    agent/PPO/config.py \
    agent/PPO/slow/slow_matching.py \
    agent/PPO/slow/test_fast_slow.py \
    agent/PPO/fast/fast_train.py


"${PYTHON_BIN}" -m unittest -v \
    env.delivery.test_rsu_delivery_crn \
    agent.PPO.slow.test_fast_slow \
    agent.PPO.fast.test_fast_pretrain_contract


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
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda available =", torch.cuda.is_available())
print("cuda device count =", torch.cuda.device_count())

if torch.version.cuda is None:
    raise SystemExit("[ERROR] PyTorch is CPU-only.")

if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA unavailable.")

if torch.cuda.device_count() < 1:
    raise SystemExit("[ERROR] No CUDA device.")

print("device 0 =", torch.cuda.get_device_name(0))
print("[CUDA PREFLIGHT] PASS")

PY


# =====================================================================
# Best-effort GPU monitoring
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
# 5-seed full evaluation
# =====================================================================

echo "============================================================"
echo "[FULL SLOW EVALUATION START]"
echo "method=${METHOD}"
echo "commit=$(git rev-parse HEAD)"
echo "checkpoint=${SELECTED_FAST_CHECKPOINT}"
echo "============================================================"


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

    ROW_COUNT="$(
        tail -n +2 "${ROUND_CSV}" | wc -l
    )"

    [[ "${ROW_COUNT}" -eq 50 ]] \
        || die \
        "Expected 50 rounds for seed=${SEED}, got=${ROW_COUNT}"

    echo "[SEED DONE] ${SEED}"

done


echo "============================================================"
echo "[FULL SLOW EVALUATION DONE]"
echo "method=${METHOD}"
echo "time=$(date --iso-8601=seconds)"
echo "output=${OUTPUT_ROOT}"
echo "============================================================"