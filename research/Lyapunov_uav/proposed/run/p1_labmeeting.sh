#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

P1_MPL_CACHE="${TMPDIR:-/tmp}/p1-matplotlib-cache"
mkdir -p "$P1_MPL_CACHE"
export MPLCONFIGDIR="$P1_MPL_CACHE"

python -m unittest discover -s tests -p 'test_p1_*.py' -v

# Quick reproducible lab-meeting smoke. Increase --frames/--rollouts only after
# replacing the uncalibrated defaults in config_p1.py with documented values.
python -m run.p1_labmeeting \
  --policies dpp,rsu_only,always_hire,fixed_rsu \
  --seeds 2026,2027,2028 \
  --frames 30 \
  --rollouts 2 \
  --output outputs/p1_modular

echo "P1 outputs: $PROJECT_DIR/outputs/p1_modular"
