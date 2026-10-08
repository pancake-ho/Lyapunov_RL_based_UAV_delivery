#!/usr/bin/bash

set -Eeuo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    echo "usage: bash run/p3_watch.sh JOB_ID [OUTPUT_DIR]" >&2
    exit 2
fi

readonly JOB_ID="$1"
readonly PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
readonly SLURM_OUT="$PROJECT_DIR/logs/p3-ppo-$JOB_ID.out"
readonly SLURM_ERR="$PROJECT_DIR/logs/p3-ppo-$JOB_ID.err"
readonly OUTPUT_DIR="${2:-}"

echo "[WATCH] job status"
squeue -j "$JOB_ID" -o "%.18i %.12P %.24j %.2t %.10M %.10l %.6D %R" || true
echo
echo "[WATCH] Ctrl-C stops only this viewer; it does not cancel the job."

while [[ ! -e "$SLURM_OUT" && ! -e "$SLURM_ERR" ]]; do
    if [[ -z "$(squeue -h -j "$JOB_ID")" ]]; then
        echo "Job $JOB_ID is no longer queued and no Slurm log was found." >&2
        exit 1
    fi
    sleep 1
done

if [[ -n "$OUTPUT_DIR" ]]; then
    echo "[WATCH] live status file: $OUTPUT_DIR/status.json"
    echo "[WATCH] structured events: $OUTPUT_DIR/training_events.jsonl"
fi

touch "$SLURM_OUT" "$SLURM_ERR"
tail -n 60 -F "$SLURM_OUT" "$SLURM_ERR"
