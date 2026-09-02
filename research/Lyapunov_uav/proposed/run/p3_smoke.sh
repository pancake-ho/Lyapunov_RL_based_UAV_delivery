#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

P3_MPL_CACHE="${TMPDIR:-/tmp}/p3-matplotlib-cache"
mkdir -p "$P3_MPL_CACHE"
export MPLCONFIGDIR="$P3_MPL_CACHE"

python -m compileall -q .
python -m unittest discover -s tests -p 'test_p3_*.py' -v
python -m run.p3_compare \
  --policies dpp,rsu_only,always_hire,fixed_rsu,nearest_hotspot \
  --seeds 2026 \
  --frames 2 \
  --rollouts 1 \
  --output outputs/p3_smoke

echo "P3 smoke outputs: $PROJECT_DIR/outputs/p3_smoke"
