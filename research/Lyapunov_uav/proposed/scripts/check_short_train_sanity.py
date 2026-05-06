from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


DPP_KEYS = (
    "video_delivery_pressure",
    "quality_reward",
    "battery_service_pressure",
    "charging_effect",
    "total_dpp_reward",
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _summary_bounds_ok(summary: Dict[str, Any], *, low: float, high: float) -> bool:
    for key in ("min", "mean", "max", "sum"):
        if key not in summary or not _is_finite_number(summary[key]):
            return False
    return float(summary["min"]) >= low and float(summary["max"]) <= high


def _summary_nonnegative(summary: Dict[str, Any]) -> bool:
    for key in ("min", "mean", "max", "sum"):
        if key not in summary or not _is_finite_number(summary[key]):
            return False
    return float(summary["min"]) >= 0.0


def _add_result(results: List[Dict[str, str]], name: str, ok: bool, detail: str) -> None:
    results.append(
        {
            "status": "PASS" if ok else "FAIL",
            "check": name,
            "detail": detail,
        }
    )


def _checkpoint_candidates(rows: Iterable[Dict[str, Any]], checkpoint_dir: Path | None, run_name: str | None):
    seen_paths = []
    for row in rows:
        if row.get("event") == "checkpoint_saved" and row.get("checkpoint_path"):
            seen_paths.append(Path(str(row["checkpoint_path"])).expanduser())
    if checkpoint_dir is not None and run_name:
        seen_paths.extend(checkpoint_dir.expanduser().glob(f"{run_name}_update*.pt"))
    return list(dict.fromkeys(seen_paths))


def run_checks(
    *,
    log_path: Path,
    checkpoint_dir: Path | None,
    max_queue: float,
    entropy_min: float,
    reward_tol: float,
) -> List[Dict[str, str]]:
    rows = _load_jsonl(log_path)
    updates = [row for row in rows if row.get("event") == "short_train_update"]
    checkpoints = [row for row in rows if row.get("event") == "checkpoint_saved"]
    start = next((row for row in rows if row.get("event") == "short_train_start"), {})
    run_name = str(start.get("run_name", "")) or None

    results: List[Dict[str, str]] = []

    _add_result(
        results,
        "log_has_updates",
        len(updates) > 0,
        f"found {len(updates)} short_train_update records",
    )

    finite_metrics = ("actor_loss", "critic_loss", "total_reward")
    bad_finite = []
    for idx, row in enumerate(updates, start=1):
        for metric in finite_metrics:
            if not _is_finite_number(row.get(metric)):
                bad_finite.append(f"update {idx} {metric}={row.get(metric)!r}")
    _add_result(
        results,
        "finite_loss_reward",
        not bad_finite,
        "; ".join(bad_finite) if bad_finite else "actor_loss, critic_loss, total_reward are finite",
    )

    bad_entropy = []
    for idx, row in enumerate(updates, start=1):
        entropy = row.get("entropy")
        if not _is_finite_number(entropy) or float(entropy) <= entropy_min:
            bad_entropy.append(f"update {idx} entropy={entropy!r}")
    _add_result(
        results,
        "entropy_not_collapsed",
        not bad_entropy,
        "; ".join(bad_entropy) if bad_entropy else f"all entropy values > {entropy_min}",
    )

    rewards = [float(row["total_reward"]) for row in updates if _is_finite_number(row.get("total_reward"))]
    if len(rewards) <= 1:
        reward_ok = True
        reward_detail = "only one update; reward variation check skipped"
    else:
        reward_ok = (max(rewards) - min(rewards)) > reward_tol
        reward_detail = f"total_reward range={max(rewards) - min(rewards):.6g}, tol={reward_tol}"
    _add_result(results, "total_reward_not_constant", reward_ok, reward_detail)

    missing_dpp = []
    for idx, row in enumerate(updates, start=1):
        terms = row.get("dpp_terms")
        if not isinstance(terms, dict):
            missing_dpp.append(f"update {idx} dpp_terms missing")
            continue
        for key in DPP_KEYS:
            if key not in terms or not _is_finite_number(terms[key]):
                missing_dpp.append(f"update {idx} dpp_terms.{key}={terms.get(key)!r}")
    _add_result(
        results,
        "dpp_terms_present",
        not missing_dpp,
        "; ".join(missing_dpp) if missing_dpp else "all required DPP terms are present and finite",
    )

    bad_queue = []
    for idx, row in enumerate(updates, start=1):
        for metric in ("queue", "virtual_queue"):
            summary = row.get(metric)
            if not isinstance(summary, dict) or not _summary_bounds_ok(summary, low=0.0, high=max_queue):
                bad_queue.append(f"update {idx} {metric}={summary!r}")
    _add_result(
        results,
        "queue_bounds",
        not bad_queue,
        "; ".join(bad_queue) if bad_queue else f"queue summaries are within [0, {max_queue}]",
    )

    bad_soc = []
    for idx, row in enumerate(updates, start=1):
        summary = row.get("uav_soc")
        if not isinstance(summary, dict) or not _summary_bounds_ok(summary, low=0.0, high=100.0):
            bad_soc.append(f"update {idx} uav_soc={summary!r}")
    _add_result(
        results,
        "uav_soc_bounds",
        not bad_soc,
        "; ".join(bad_soc) if bad_soc else "UAV SoC summaries are within [0, 100]",
    )

    bad_battery_virtual = []
    for idx, row in enumerate(updates, start=1):
        summary = row.get("uav_virtual_queue")
        if not isinstance(summary, dict) or not _summary_nonnegative(summary):
            bad_battery_virtual.append(f"update {idx} uav_virtual_queue={summary!r}")
    _add_result(
        results,
        "battery_virtual_queue_nonnegative",
        not bad_battery_virtual,
        "; ".join(bad_battery_virtual)
        if bad_battery_virtual
        else "battery virtual queue summaries are nonnegative",
    )

    bad_done = []
    for idx, row in enumerate(updates, start=1):
        if not isinstance(row.get("terminated"), bool):
            bad_done.append(f"update {idx} terminated={row.get('terminated')!r}")
        if not isinstance(row.get("truncated"), bool):
            bad_done.append(f"update {idx} truncated={row.get('truncated')!r}")
    _add_result(
        results,
        "terminated_truncated_recorded",
        not bad_done,
        "; ".join(bad_done) if bad_done else "terminated/truncated are recorded as booleans",
    )

    checkpoint_paths = _checkpoint_candidates(rows, checkpoint_dir, run_name)
    existing_checkpoints = [path for path in checkpoint_paths if path.exists()]
    _add_result(
        results,
        "checkpoint_exists",
        len(existing_checkpoints) > 0,
        f"found {len(existing_checkpoints)} checkpoint files"
        if existing_checkpoints
        else f"no checkpoint files found; checkpoint_saved records={len(checkpoints)}",
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check short HRL PPO train JSONL logs.")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-queue", type=float, default=100.0)
    parser.add_argument("--entropy-min", type=float, default=1e-8)
    parser.add_argument("--reward-tol", type=float, default=1e-8)
    args = parser.parse_args()

    results = run_checks(
        log_path=Path(args.log_path).expanduser(),
        checkpoint_dir=Path(args.checkpoint_dir).expanduser() if args.checkpoint_dir else None,
        max_queue=float(args.max_queue),
        entropy_min=float(args.entropy_min),
        reward_tol=float(args.reward_tol),
    )
    failed = [result for result in results if result["status"] != "PASS"]
    for result in results:
        print(f"{result['status']} {result['check']}: {result['detail']}")
    print("PASS short_train_sanity" if not failed else "FAIL short_train_sanity")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
