#!/usr/bin/env bash
# Usage:
#   bash run/hrl_export.sh RUN_DIR [export options]
#
# Execute on a local PC or inside an allocated compute-node environment.
# This script does not submit a Slurm job or start training.

set -Eeuo pipefail

if (( $# < 1 )); then
    echo "Usage: bash run/hrl_export.sh RUN_DIR [options]" >&2
    exit 2
fi

host_name="$(hostname -s)"
if [[ "$host_name" == "moana-master" ]]; then
    echo "Do not run Python on moana-master." >&2
    echo "Use your local PC or an allocated compute-node environment." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
run_dir="$(cd -- "$1" && pwd -P)"
shift

cd -- "$project_root"

python_bin="${HPPO_EXPORT_PYTHON:-python}"

if ! command -v "$python_bin" >/dev/null; then
    echo "Python not found. Set HPPO_EXPORT_PYTHON=python3 if needed." >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec "$python_bin" -m hppo.export_files "$run_dir" "$@"