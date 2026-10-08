#!/usr/bin/env bash
# Run inside an allocated compute node, or on a local PC with Python >= 3.10.
# Usage: bash run/hrl_explain.sh RUN_DIR [--episode 0] [--region 0] [--watch 10]
set -Eeuo pipefail
if (( $# < 1 )); then
    echo 'Usage: bash run/hrl_explain.sh RUN_DIR [options]' >&2
    exit 2
fi
if [[ "$(hostname -s)" == moana-master ]]; then
    echo '계산 노드 할당 안에서 실행하세요: bash run/hrl_explain_submit.sh RUN_DIR [options]' >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
run_dir="$(cd -- "$1" && pwd -P)"
shift
cd -- "$project_root"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "${HPPO_EXPLAIN_PYTHON:-python}" hppo/explain_trace.py "$run_dir" "$@"
