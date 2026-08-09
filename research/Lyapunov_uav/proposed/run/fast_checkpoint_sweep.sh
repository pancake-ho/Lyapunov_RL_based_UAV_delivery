#!/usr/bin/bash

#SBATCH -J fast-h2-sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-fast-sweep-%A.out

set -euo pipefail
umask 027

PROJECT_ROOT="${FAST_PROJECT_ROOT:-/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
CONDA_ROOT="${FAST_CONDA_ROOT:-/data/surt321/anaconda3}"
CONDA_ENV_PATH="${FAST_CONDA_ENV_PATH:-${CONDA_ROOT}/envs/lab}"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
EXPECTED_BRANCH="feat/no-hrl"

FAST_RUN_ROOT="${FAST_RUN_ROOT:-${FAST_H1_RUN_ROOT:-}}"
SWEEP_OUTPUT="${FAST_CHECKPOINT_SWEEP_OUTPUT:-}"

CHECKPOINT_EPISODES_TEXT="${FAST_CHECKPOINT_SWEEP_CHECKPOINT_EPISODES:-100 150 200 225 250 275 300 305}"
EVAL_SEEDS_TEXT="${FAST_CHECKPOINT_SWEEP_SEEDS:-2026 2027 2028}"
EVAL_EPISODES="${FAST_CHECKPOINT_SWEEP_EPISODES:-5}"
EVAL_ROUNDS="${FAST_CHECKPOINT_SWEEP_ROUNDS:-5}"
WORKLOAD_TOLERANCE="${FAST_CHECKPOINT_SWEEP_WORKLOAD_TOLERANCE:-0.02}"
MINIMUM_ALLOWED_SOC="${FAST_CHECKPOINT_SWEEP_MIN_SOC:-19.95}"
REUSE_COMPLETED="${FAST_CHECKPOINT_SWEEP_REUSE:-0}"

die() {
    echo "[ERROR] $*" >&2
    exit 2
}

[[ -n "${FAST_RUN_ROOT}" ]] \
    || die "Set FAST_RUN_ROOT to the H2 Fast-pretraining run directory."
[[ -d "${PROJECT_ROOT}" ]] \
    || die "Project root not found: ${PROJECT_ROOT}"
[[ -d "${FAST_RUN_ROOT}/checkpoints" ]] \
    || die "Checkpoint directory not found: ${FAST_RUN_ROOT}/checkpoints"
[[ -x "${PYTHON_BIN}" ]] \
    || die "Python not found: ${PYTHON_BIN}"
[[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]] \
    || die "Conda initialization script not found."

if [[ -z "${SWEEP_OUTPUT}" ]]; then
    SWEEP_OUTPUT="${FAST_RUN_ROOT}/checkpoint_sweep_h2_v1"
fi

cd "${PROJECT_ROOT}"
CURRENT_BRANCH="$(git branch --show-current)"
[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] \
    || die "Wrong branch: expected=${EXPECTED_BRANCH}, actual=${CURRENT_BRANCH}"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
    || die "Working tree is not clean. Commit or stash source changes first."

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"
hash -r
[[ "$(command -v python)" == "${PYTHON_BIN}" ]] \
    || die "Activated Python differs from ${PYTHON_BIN}."
echo "============================================================"
echo "[GPU PREFLIGHT]"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NOT_SET}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-NOT_SET}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-NOT_SET}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NOT_SET}"
echo "============================================================"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2026
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FAST_PPO_DEVICE="${FAST_PPO_DEVICE:-cuda}"
export FAST_PPO_DETERMINISTIC_TORCH=1

unset FAST_PRETRAIN_RESUME_CHECKPOINT
unset JOINT_RESUME_CHECKPOINT
unset JOINT_FAST_CHECKPOINT
unset FAST_PPO_RESUME

"${PYTHON_BIN}" -m py_compile \
    agent/PPO/fast/fast_train.py \
    agent/PPO/fast/fast_checkpoint_sweep.py

"${PYTHON_BIN}" -m unittest -v \
    agent.PPO.fast.test_fast_checkpoint_sweep

read -r -a CHECKPOINT_EPISODES <<< "${CHECKPOINT_EPISODES_TEXT}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS_TEXT}"

[[ "${#CHECKPOINT_EPISODES[@]}" -ge 1 ]] \
    || die "No checkpoint episodes were supplied."
[[ "${#EVAL_SEEDS[@]}" -ge 2 ]] \
    || die "Use at least two evaluation seeds for a confidence interval."

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

if [[ "${REUSE_COMPLETED}" == "1" ]]; then
    COMMAND+=(--reuse-completed)
fi

echo "[SWEEP START]"
echo "  fast_run_root       = ${FAST_RUN_ROOT}"
echo "  output              = ${SWEEP_OUTPUT}"
echo "  checkpoints         = ${CHECKPOINT_EPISODES[*]}"
echo "  eval_seeds          = ${EVAL_SEEDS[*]}"
echo "  eval_episodes       = ${EVAL_EPISODES}"
echo "  eval_rounds         = ${EVAL_ROUNDS}"
echo "  git_commit          = $(git rev-parse HEAD)"

srun \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${COMMAND[@]}"

echo "[SWEEP DONE]"
echo "  decision=${SWEEP_OUTPUT}/selection.json"
