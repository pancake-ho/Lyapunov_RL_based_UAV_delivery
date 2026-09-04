#!/usr/bin/bash

set -Eeuo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-$PWD}"
readonly EVAL_ROOT="${EVAL_ROOT:-$PROJECT_DIR/outputs/p3_eval_formfix_seed2026}"
readonly INTERVAL_SECONDS="${INTERVAL_SECONDS:-30}"

cd "$PROJECT_DIR"
while true; do
    clear || true
    echo "[WATCH] $(date --iso-8601=seconds) root=$EVAL_ROOT"
    squeue -u "$USER" -o '%.18i %.10P %.24j %.2t %.10M %.10l %R' || true
    python - "$EVAL_ROOT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
statuses = []
for path in root.glob("*/status/status_*.json"):
    try:
        statuses.append(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        pass
counts = Counter(item.get("event", "unknown") for item in statuses)
print("status:", " ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none")
active = [item for item in statuses if item.get("event") in {"started", "progress"}]
active.sort(key=lambda item: item.get("timestamp_utc", ""), reverse=True)
for item in active[:4]:
    fp = item.get("fingerprint", {})
    done = int(item.get("processed_frames", 0))
    total = int(item.get("total_frames", fp.get("frames", 0)))
    eta = float(item.get("eta_seconds", 0.0)) / 60.0
    print(
        f"active group={fp.get('group')} policy={fp.get('policy')} "
        f"seed={fp.get('seed')} frame={done}/{total} eta={eta:.1f}m"
    )
submission = root / "submission.txt"
if submission.is_file():
    print("\nsubmission:")
    print(submission.read_text(encoding="utf-8").strip())
aggregate = root / "aggregate" / "aggregate_summary.csv"
print(f"\naggregate_ready={int(aggregate.is_file())}")
PY
    sleep "$INTERVAL_SECONDS"
done
