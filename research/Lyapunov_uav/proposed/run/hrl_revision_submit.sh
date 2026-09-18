#!/usr/bin/env bash
set -Eeuo pipefail
mode="${1:-train}"
if [[ $# -gt 1 || ( "$mode" != train && "$mode" != smoke ) ]]; then
    echo 'Usage: bash run/hrl_revision_submit.sh [train|smoke]' >&2; exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
: "${HPPO_PARTITION:?Set HPPO_PARTITION after checking sinfo}"
for tool in sinfo squeue scontrol sbatch; do
    command -v "$tool" >/dev/null
done
sinfo -p "$HPPO_PARTITION" -o '%P %a %l %D %G'
squeue -u "${USER:-$(id -un)}"
scontrol show partition "$HPPO_PARTITION"
mkdir -p "$project_root/slurm_logs"
export HPPO_PROJECT_ROOT="$project_root" HPPO_MODE="$mode"
limit=1-00:00:00
if [[ "$mode" == smoke ]]; then limit=01:00:00; fi
sbatch --job-name="hrl_revision_$mode" --partition="$HPPO_PARTITION" \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-gpu="${HPPO_CPUS:-16}" --mem-per-gpu="${HPPO_MEM:-29G}" \
    --time="${HPPO_TIME:-$limit}" --no-requeue --chdir="$project_root" \
    --output="$project_root/slurm_logs/%x-%j.out" \
    --error="$project_root/slurm_logs/%x-%j.err" \
    --export=ALL "$script_dir/hrl_revision200.sbatch"
