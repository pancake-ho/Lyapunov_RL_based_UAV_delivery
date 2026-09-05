#!/usr/bin/bash

set -Eeuo pipefail

readonly EXPECTED_BRANCH="feat/new-form-p3"
readonly PROJECT_DIR="${PROJECT_DIR:-$PWD}"
readonly P3_RUN_DIR="${P3_RUN_DIR:?export P3_RUN_DIR to the accepted training output}"
readonly BEST_CHECKPOINT="${BEST_CHECKPOINT:-$P3_RUN_DIR/best.pt}"
readonly LATEST_CHECKPOINT="${LATEST_CHECKPOINT:-$P3_RUN_DIR/latest.pt}"
readonly ACCEPTANCE_JSON="${ACCEPTANCE_JSON:-$P3_RUN_DIR/acceptance.json}"
readonly EVAL_ROOT="${EVAL_ROOT:-$PROJECT_DIR/outputs/p3_eval_formfix_seed2026}"
readonly EVAL_SEEDS="${EVAL_SEEDS:-3026:3027:3028:3029:3030:3031:3032:3033:3034:3035}"
readonly EVAL_BASELINE_POLICIES="${EVAL_BASELINE_POLICIES:-dpp:load_threshold:always_hire:fixed_rsu:nearest_hotspot:rsu_only}"
readonly EVAL_FRAMES="${EVAL_FRAMES:-400}"
readonly EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-4}"
readonly EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-10}"
readonly EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
readonly EVAL_SELECTION_WORKERS="${EVAL_SELECTION_WORKERS:-4}"
readonly MAX_PARALLEL="${MAX_PARALLEL:-2}"
readonly MAX_SUBMITTED_JOBS="${MAX_SUBMITTED_JOBS:-10}"

cd "$PROJECT_DIR"
if ! command -v sbatch >/dev/null ||
    ! command -v squeue >/dev/null ||
    ! command -v scancel >/dev/null; then
    echo "Slurm commands sbatch, squeue, and scancel are required" >&2
    exit 2
fi
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "Expected branch $EXPECTED_BRANCH, got $(git branch --show-current)" >&2
    exit 2
fi
if [[ ! -f "$BEST_CHECKPOINT" || ! -f "$LATEST_CHECKPOINT" ]]; then
    echo "Missing checkpoint(s): best=$BEST_CHECKPOINT latest=$LATEST_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$ACCEPTANCE_JSON" ]] || ! python - "$ACCEPTANCE_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
if payload.get("event") != "passed":
    raise SystemExit(1)
PY
then
    echo "Training acceptance gate did not pass: $ACCEPTANCE_JSON" >&2
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

# Seraph counts every array element toward MaxSubmitPU.  This workflow submits
# one preflight job plus MAX_PARALLEL long-lived array workers, never 80 array
# elements.  Check the accounting limit before creating a partial chain.
existing_job_count="$(
    squeue -h -r -u "$USER" -o '%i' \
        | sed '/^[[:space:]]*$/d' \
        | sort -u \
        | wc -l
)"
required_submit_slots="$((1 + MAX_PARALLEL))"
if (( existing_job_count + required_submit_slots > MAX_SUBMITTED_JOBS )); then
    echo "Insufficient MaxSubmitPU slots: existing=$existing_job_count required=$required_submit_slots limit=$MAX_SUBMITTED_JOBS" >&2
    squeue -u "$USER" -o '%.18i %.12P %.24j %.2t %.10M %.10l %R' >&2
    exit 2
fi

seed_count="$(awk -F: '{print NF}' <<< "$EVAL_SEEDS")"
policy_count="$(awk -F: '{print NF}' <<< "$EVAL_BASELINE_POLICIES")"
task_count="$((seed_count * (policy_count + 2)))"
last_worker="$((MAX_PARALLEL - 1))"
mkdir -p logs "$EVAL_ROOT"

export PROJECT_DIR BEST_CHECKPOINT LATEST_CHECKPOINT EVAL_ROOT
export EVAL_SEEDS EVAL_BASELINE_POLICIES EVAL_FRAMES EVAL_ROLLOUTS
export EVAL_PROGRESS_INTERVAL EVAL_DEVICE EVAL_SELECTION_WORKERS
export EVAL_JOB_WORKERS="$MAX_PARALLEL"

preflight_submission="$(sbatch --parsable --export=ALL run/submit_p3_eval_preflight.sbatch)"
preflight_job="${preflight_submission%%;*}"
if ! array_submission="$(
    sbatch --parsable \
        --dependency="afterok:$preflight_job" \
        --array="0-${last_worker}%${MAX_PARALLEL}" \
        --export=ALL \
        run/submit_p3_eval.sbatch
)"; then
    echo "Worker array submission failed; cancelling orphan preflight $preflight_job" >&2
    scancel "$preflight_job"
    exit 1
fi
array_job="${array_submission%%;*}"

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
    echo "job_workers=$MAX_PARALLEL"
    echo "tasks_per_worker=$(((task_count + MAX_PARALLEL - 1) / MAX_PARALLEL))"
    echo "submitted_slots=$required_submit_slots"
    echo "preflight_job=$preflight_job"
    echo "array_job=$array_job"
    echo "aggregate_mode=final_completed_array_task"
} | tee "$EVAL_ROOT/submission.txt"

echo "Monitor with: EVAL_ROOT=$EVAL_ROOT bash run/p3_watch_eval.sh"
