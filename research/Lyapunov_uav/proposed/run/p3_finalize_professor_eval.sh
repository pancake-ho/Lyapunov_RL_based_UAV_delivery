#!/usr/bin/bash

set -Eeuo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-$PWD}"
readonly EVAL_ROOT="${EVAL_ROOT:?export EVAL_ROOT to the completed evaluation root}"
readonly CONDA_SH="${CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
readonly CONDA_ENV="${CONDA_ENV:-lab}"

cd "$PROJECT_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ ! -f "$EVAL_ROOT/aggregate/experiment.json" ]]; then
    echo "Missing aggregate/experiment.json; evaluation workers may still be running." >&2
    echo "Monitor with: EVAL_ROOT=$EVAL_ROOT bash run/p3_watch_eval.sh" >&2
    exit 2
fi

python -u -m run.p3_professor_eval_report \
    --input "$EVAL_ROOT" \
    --output "$EVAL_ROOT/aggregate/professor"

printf '%s\n' \
    "[REPORT] $EVAL_ROOT/aggregate/professor/professor_aggregate_metrics.csv" \
    "[REPORT] $EVAL_ROOT/aggregate/professor/p3_bitrate_stall_quality.png" \
    "[REPORT] $EVAL_ROOT/aggregate/professor/p3_quality_distribution.png" \
    "[REPORT] $EVAL_ROOT/aggregate/professor/p3_charging_safety.png" \
    "[REPORT] $EVAL_ROOT/aggregate/professor/p3_distance_effects_professor.png"
