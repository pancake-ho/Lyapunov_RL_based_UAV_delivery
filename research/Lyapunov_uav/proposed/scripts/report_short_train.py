from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from check_short_train_sanity import DPP_KEYS, _checkpoint_candidates, _is_finite_number, _load_jsonl, run_checks


METRIC_KEYS = ("total_reward", "slow_reward", "fast_reward", "actor_loss", "critic_loss", "entropy")


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _finite_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if _is_finite_number(value):
            values.append(float(value))
    return values


def _metric_stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, float | None]:
    values = _finite_values(rows, key)
    final_value = rows[-1].get(key) if rows else None
    return {
        "mean": _mean(values),
        "final": float(final_value) if _is_finite_number(final_value) else None,
    }


def _dpp_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float | None]]:
    stats: Dict[str, Dict[str, float | None]] = {}
    for key in DPP_KEYS:
        values = []
        for row in rows:
            terms = row.get("dpp_terms")
            if isinstance(terms, dict) and _is_finite_number(terms.get(key)):
                values.append(float(terms[key]))
        final_terms = rows[-1].get("dpp_terms") if rows else None
        final_value = final_terms.get(key) if isinstance(final_terms, dict) else None
        stats[key] = {
            "mean": _mean(values),
            "final": float(final_value) if _is_finite_number(final_value) else None,
        }
    return stats


def _summary_stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, float | None]:
    mins = []
    means = []
    maxes = []
    for row in rows:
        summary = row.get(key)
        if not isinstance(summary, dict):
            continue
        if _is_finite_number(summary.get("min")):
            mins.append(float(summary["min"]))
        if _is_finite_number(summary.get("mean")):
            means.append(float(summary["mean"]))
        if _is_finite_number(summary.get("max")):
            maxes.append(float(summary["max"]))
    return {
        "min": min(mins) if mins else None,
        "mean": _mean(means),
        "max": max(maxes) if maxes else None,
    }


def _latest_log_path(log_dir: Path) -> Path:
    candidates = sorted(
        (path for path in log_dir.expanduser().glob("*.jsonl") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .jsonl short train logs found in {log_dir}")
    return candidates[0]


def _resolve_log_path(log_path: Path | None, log_dir: Path | None, run_name: str | None) -> Path:
    if log_path is not None:
        return log_path.expanduser()
    if log_dir is None:
        raise ValueError("Provide either --log-path or --log-dir.")
    if run_name:
        return log_dir.expanduser() / f"{run_name}.jsonl"
    return _latest_log_path(log_dir)


def _result_lookup(results: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {result["check"]: result for result in results}


def _warning_flags(sanity_results: List[Dict[str, str]]) -> Dict[str, bool]:
    by_check = _result_lookup(sanity_results)

    def failed(check: str) -> bool:
        return by_check.get(check, {}).get("status") != "PASS"

    return {
        "loss_nan_inf": failed("finite_loss_reward"),
        "reward_constant_collapse": failed("total_reward_not_constant"),
        "entropy_collapse": failed("entropy_not_collapsed"),
        "dpp_terms_missing": failed("dpp_terms_present"),
        "soc_out_of_range": failed("uav_soc_bounds"),
        "virtual_queue_out_of_range": failed("queue_bounds") or failed("battery_virtual_queue_nonnegative"),
        "checkpoint_missing": failed("checkpoint_exists"),
    }


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_report(
    *,
    log_path: Path,
    checkpoint_dir: Path | None,
    max_queue: float,
    entropy_min: float,
    reward_tol: float,
) -> Dict[str, Any]:
    rows = _load_jsonl(log_path)
    start = next((row for row in rows if row.get("event") == "short_train_start"), {})
    done = next((row for row in reversed(rows) if row.get("event") == "short_train_done"), {})
    updates = [row for row in rows if row.get("event") == "short_train_update"]

    run_name = str(start.get("run_name") or (updates[-1].get("run_name") if updates else log_path.stem))
    resolved_checkpoint_dir = checkpoint_dir
    if resolved_checkpoint_dir is None and start.get("checkpoint_dir"):
        resolved_checkpoint_dir = Path(str(start["checkpoint_dir"])).expanduser()

    sanity_results = run_checks(
        log_path=log_path,
        checkpoint_dir=resolved_checkpoint_dir,
        max_queue=max_queue,
        entropy_min=entropy_min,
        reward_tol=reward_tol,
    )
    sanity_failed = [result for result in sanity_results if result["status"] != "PASS"]

    checkpoint_paths = _checkpoint_candidates(rows, resolved_checkpoint_dir, run_name)
    existing_checkpoints = [path for path in checkpoint_paths if path.exists()]
    latest_checkpoint = existing_checkpoints[-1] if existing_checkpoints else None

    metrics = {key: _metric_stats(updates, key) for key in METRIC_KEYS}
    final_update = updates[-1] if updates else {}
    total_steps = int(done.get("step") or final_update.get("step") or 0)
    total_updates = int(done.get("updates") or final_update.get("update") or len(updates))

    report = {
        "run_name": run_name,
        "log_path": str(log_path),
        "device": start.get("device"),
        "seed": start.get("seed"),
        "total_steps": total_steps,
        "total_updates": total_updates,
        "metrics": metrics,
        "dpp_terms": _dpp_stats(updates),
        "queue": _summary_stats(updates, "queue"),
        "virtual_queue": _summary_stats(updates, "virtual_queue"),
        "uav_soc": _summary_stats(updates, "uav_soc"),
        "uav_virtual_queue": _summary_stats(updates, "uav_virtual_queue"),
        "terminated_count": sum(1 for row in updates if row.get("terminated") is True),
        "truncated_count": sum(1 for row in updates if row.get("truncated") is True),
        "checkpoint_path": str(latest_checkpoint) if latest_checkpoint else None,
        "checkpoint_count": len(existing_checkpoints),
        "sanity": {
            "status": "PASS" if not sanity_failed else "FAIL",
            "passed": len(sanity_results) - len(sanity_failed),
            "failed": len(sanity_failed),
            "results": sanity_results,
            "warnings": _warning_flags(sanity_results),
        },
    }
    return _sanitize_for_json(report)


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_markdown(report: Dict[str, Any]) -> str:
    metrics = report["metrics"]
    dpp_terms = report["dpp_terms"]
    sanity = report["sanity"]

    lines = [
        "# Short HRL Train Report",
        "",
        "## Run",
        "",
        f"- run_name: `{report['run_name']}`",
        f"- device: `{_format_value(report.get('device'))}`",
        f"- seed: `{_format_value(report.get('seed'))}`",
        f"- total_steps: `{report['total_steps']}`",
        f"- total_updates: `{report['total_updates']}`",
        f"- log_path: `{report['log_path']}`",
        f"- checkpoint_path: `{_format_value(report.get('checkpoint_path'))}`",
        f"- checkpoint_count: `{report['checkpoint_count']}`",
        "",
        "## Rewards And Losses",
        "",
        "| metric | mean | final |",
        "| --- | ---: | ---: |",
    ]
    for key in METRIC_KEYS:
        stat = metrics[key]
        lines.append(f"| {key} | {_format_value(stat['mean'])} | {_format_value(stat['final'])} |")

    lines.extend(
        [
            "",
            "## DPP Terms",
            "",
            "| term | mean | final |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in DPP_KEYS:
        stat = dpp_terms[key]
        lines.append(f"| {key} | {_format_value(stat['mean'])} | {_format_value(stat['final'])} |")

    lines.extend(
        [
            "",
            "## Queues And Battery",
            "",
            "| summary | min | mean | max |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in ("queue", "virtual_queue", "uav_soc", "uav_virtual_queue"):
        stat = report[key]
        lines.append(
            f"| {key} | {_format_value(stat['min'])} | {_format_value(stat['mean'])} | {_format_value(stat['max'])} |"
        )

    lines.extend(
        [
            "",
            "## Done Counts",
            "",
            f"- terminated_count: `{report['terminated_count']}`",
            f"- truncated_count: `{report['truncated_count']}`",
            "",
            "## Sanity",
            "",
            f"- status: `{sanity['status']}`",
            f"- passed: `{sanity['passed']}`",
            f"- failed: `{sanity['failed']}`",
            "",
            "| status | check | detail |",
            "| --- | --- | --- |",
        ]
    )
    for result in sanity["results"]:
        detail = str(result["detail"]).replace("\n", " ")
        lines.append(f"| {result['status']} | {result['check']} | {detail} |")

    lines.extend(["", "## Warning Flags", ""])
    for key, enabled in sanity["warnings"].items():
        status = "WARN" if enabled else "OK"
        lines.append(f"- {status} {key}")

    return "\n".join(lines) + "\n"


def print_console_summary(report: Dict[str, Any]) -> None:
    print(format_markdown(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a short HRL PPO train JSONL log.")
    parser.add_argument("--log-path", default=None, help="Exact JSONL log path.")
    parser.add_argument("--log-dir", default=None, help="Directory containing short train JSONL logs.")
    parser.add_argument("--run-name", default=None, help="Run name without the .jsonl suffix.")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--max-queue", type=float, default=100.0)
    parser.add_argument("--entropy-min", type=float, default=1e-8)
    parser.add_argument("--reward-tol", type=float, default=1e-8)
    args = parser.parse_args()

    log_path = _resolve_log_path(
        Path(args.log_path) if args.log_path else None,
        Path(args.log_dir) if args.log_dir else None,
        args.run_name,
    )
    report = build_report(
        log_path=log_path,
        checkpoint_dir=Path(args.checkpoint_dir).expanduser() if args.checkpoint_dir else None,
        max_queue=float(args.max_queue),
        entropy_min=float(args.entropy_min),
        reward_tol=float(args.reward_tol),
    )

    print_console_summary(report)

    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md).expanduser()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(format_markdown(report), encoding="utf-8")

    raise SystemExit(0 if report["sanity"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
