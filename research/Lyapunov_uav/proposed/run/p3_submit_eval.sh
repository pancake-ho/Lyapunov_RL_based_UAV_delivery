#!/usr/bin/bash

set -Eeuo pipefail

readonly EXPECTED_BRANCH="feat/new-form-p3"
readonly PROJECT_DIR="${PROJECT_DIR:-$PWD}"
readonly P3_RUN_DIR="${P3_RUN_DIR:-$PROJECT_DIR/outputs/p3_formfix_seed2026_20260904_024036}"
readonly BEST_CHECKPOINT="${BEST_CHECKPOINT:-$P3_RUN_DIR/best.pt}"
readonly LATEST_CHECKPOINT="${LATEST_CHECKPOINT:-$P3_RUN_DIR/latest.pt}"
readonly EVAL_ROOT="${EVAL_ROOT:-$PROJECT_DIR/outputs/p3_eval_formfix_seed2026}"
readonly EVAL_SEEDS="${EVAL_SEEDS:-3026:3027:3028:3029:3030:3031:3032:3033:3034:3035}"
readonly EVAL_BASELINE_POLICIES="${EVAL_BASELINE_POLICIES:-dpp:load_threshold:always_hire:fixed_rsu:nearest_hotspot:rsu_only}"
readonly EVAL_FRAMES="${EVAL_FRAMES:-400}"
readonly EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-4}"
readonly EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-10}"
readonly EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
readonly EVAL_SELECTION_WORKERS="${EVAL_SELECTION_WORKERS:-4}"
readonly MAX_PARALLEL="${MAX_PARALLEL:-2}"

cd "$PROJECT_DIR"
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "Expected branch $EXPECTED_BRANCH, got $(git branch --show-current)" >&2
    exit 2
fi
if [[ ! -f "$BEST_CHECKPOINT" || ! -f "$LATEST_CHECKPOINT" ]]; then
    echo "Missing checkpoint(s): best=$BEST_CHECKPOINT latest=$LATEST_CHECKPOINT" >&2
    exit 2
fi
if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || (( MAX_PARALLEL > 2 )); then
    echo "MAX_PARALLEL must be 1 or 2 for the current Seraph QoS" >&2
    exit 2
fi
if ! [[ "$EVAL_SELECTION_WORKERS" =~ ^[1-9][0-9]*$ ]] || (( EVAL_SELECTION_WORKERS > 4 )); then
    echo "EVAL_SELECTION_WORKERS must be between 1 and 4" >&2
    exit 2
fi

seed_count="$(awk -F: '{print NF}' <<< "$EVAL_SEEDS")"
policy_count="$(awk -F: '{print NF}' <<< "$EVAL_BASELINE_POLICIES")"
task_count="$((seed_count * (policy_count + 2)))"
last_task="$((task_count - 1))"
mkdir -p logs "$EVAL_ROOT"

export PROJECT_DIR BEST_CHECKPOINT LATEST_CHECKPOINT EVAL_ROOT
export EVAL_SEEDS EVAL_BASELINE_POLICIES EVAL_FRAMES EVAL_ROLLOUTS
export EVAL_PROGRESS_INTERVAL EVAL_DEVICE EVAL_SELECTION_WORKERS

preflight_submission="$(sbatch --parsable --export=ALL run/submit_p3_eval_preflight.sbatch)"
preflight_job="${preflight_submission%%;*}"
array_submission="$(
    sbatch --parsable \
        --dependency="afterok:$preflight_job" \
        --array="0-${last_task}%${MAX_PARALLEL}" \
        --export=ALL \
        run/submit_p3_eval.sbatch
)"
array_job="${array_submission%%;*}"
aggregate_submission="$(
    sbatch --parsable \
        --dependency="afterok:$array_job" \
        --export=ALL \
        run/submit_p3_eval_aggregate.sbatch
)"
aggregate_job="${aggregate_submission%%;*}"

{
    echo "submitted_at=$(date --iso-8601=seconds)"
    echo "project=$PROJECT_DIR"
    echo "run_dir=$P3_RUN_DIR"
    echo "eval_root=$EVAL_ROOT"
    echo "seeds=$EVAL_SEEDS"
    echo "baseline_policies=$EVAL_BASELINE_POLICIES"
    echo "frames=$EVAL_FRAMES"
    echo "rollouts=$EVAL_ROLLOUTS"
    echo "selection_workers=$EVAL_SELECTION_WORKERS"
    echo "tasks=$task_count"
    echo "max_parallel=$MAX_PARALLEL"
    echo "preflight_job=$preflight_job"
    echo "array_job=$array_job"
    echo "aggregate_job=$aggregate_job"
} | tee "$EVAL_ROOT/submission.txt"

echo "Monitor with: EVAL_ROOT=$EVAL_ROOT bash run/p3_watch_eval.sh"
