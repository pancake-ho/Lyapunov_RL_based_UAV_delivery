#!/usr/bin/bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/$USER/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
cd "$PROJECT_DIR"

export P3_RUN_DIR="${P3_RUN_DIR:-$PWD/outputs/p3_ppo_seed2026_f400_e100_139723}"
export EVAL_ROOT="${EVAL_ROOT:-$PWD/outputs/p3_eval_stage2_reduced_s30_f400}"
export EVAL_FRAMES="${EVAL_FRAMES:-400}"
export EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-4}"
export EVAL_SELECTION_WORKERS="${EVAL_SELECTION_WORKERS:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-2}"

# Internal simulator key remains 'dpp'. Paper-facing report relabels it to Proposed.
export EVAL_BASELINE_POLICIES="${EVAL_BASELINE_POLICIES:-dpp:always_hire:rsu_only}"

# Unless explicitly overridden, p3_submit_eval_professor.sh supplies the 30
# held-out seeds 120026..120055 and audits overlap with train/val/acceptance.
unset EVAL_SEEDS

echo "[FINAL-EVAL] root=$EVAL_ROOT"
echo "[FINAL-EVAL] baselines=$EVAL_BASELINE_POLICIES + ppo_best + ppo_latest"
echo "[FINAL-EVAL] frames=$EVAL_FRAMES rollouts=$EVAL_ROLLOUTS selection_workers=$EVAL_SELECTION_WORKERS parallel=$MAX_PARALLEL"

exec bash run/p3_submit_eval_professor.sh
