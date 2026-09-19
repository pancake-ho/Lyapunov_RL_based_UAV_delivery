#!/usr/bin/env bash
# Source run is read-only; every allocation writes a new segment directory.
set -Eeuo pipefail
if [[ $# != 1 ]]; then
    echo 'Usage: bash run/hrl_resume_submit.sh /absolute/path/to/source-run' >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export HPPO_PROJECT_ROOT="$(cd -- "$script_dir/.." && pwd -P)"
export HPPO_SOURCE_RUN="$(cd -- "$1" && pwd -P)"
export HPPO_PARTITION="${HPPO_PARTITION:-batch_eebme_ugrad}"
for tool in sinfo squeue scontrol sbatch; do command -v "$tool" >/dev/null; done
sinfo -p "$HPPO_PARTITION" -o '%P %a %l %D %G'
scontrol show partition "$HPPO_PARTITION"
squeue -u "${USER:-$(id -un)}"
mkdir -p "$HPPO_PROJECT_ROOT/slurm_logs"
# 24h allocation, advance notice 30 minutes before its end.
# exec in the batch script makes Python the batch-shell PID receiving B:USR1.
sbatch --job-name=hrl_resume --partition="$HPPO_PARTITION" \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-gpu="${HPPO_CPUS:-16}" --mem-per-gpu="${HPPO_MEM:-29G}" \
    --time=1-00:00:00 --signal=B:USR1@1800 --no-requeue \
    --chdir="$HPPO_PROJECT_ROOT" \
    --output="$HPPO_PROJECT_ROOT/slurm_logs/%x-%j.out" \
    --error="$HPPO_PROJECT_ROOT/slurm_logs/%x-%j.err" \
    --export=ALL "$script_dir/hrl_resume.sbatch"
