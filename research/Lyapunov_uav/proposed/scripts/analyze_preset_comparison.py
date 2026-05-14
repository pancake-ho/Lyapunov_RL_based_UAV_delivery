from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List


REJECT_CHECKS = {
    "finite_loss_reward": "NaN/Inf or non-finite loss/reward was observed.",
    "dpp_terms_present": "required dpp_terms are missing or non-finite.",
    "uav_soc_bounds": "UAV SoC left the [0, 100] range.",
    "checkpoint_exists": "checkpoint was not created.",
}

CAUTION_CHECKS = {
    "queue_bounds": "user queue or video virtual queue left the configured range.",
    "battery_virtual_queue_nonnegative": "battery virtual queue became negative.",
    "entropy_not_collapsed": "policy entropy collapsed or became non-finite.",
    "total_reward_not_constant": "total_reward appears constant across updates.",
}

CAUTION_SCALE_WARNINGS = {
    "quality_reward_dominates_total_dpp_reward": "quality_reward dominates total_dpp_reward too often.",
    "battery_service_pressure_nearly_always_zero": "battery_service_pressure is nearly always zero.",
    "video_delivery_pressure_nearly_always_zero": "video_delivery_pressure is nearly always zero.",
    "battery_service_pressure_dominates_total_dpp_reward": "battery_service_pressure dominates total_dpp_reward too often.",
    "charging_effect_large_without_charging_event": "charging_effect is large without observed charging events.",
    "total_reward_scale_vs_critic_loss": "total_reward and critic_loss scales are imbalanced.",
    "total_reward_scale_vs_actor_loss": "total_reward and actor_loss scales are imbalanced.",
    "entropy_fast_collapse": "fast policy entropy collapsed in scale analysis.",
    "approx_kl_too_large": "approx_kl is above the configured threshold.",
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.6g}"
    return str(value)


def _check_lookup(report: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    results = report.get("sanity", {}).get("results", [])
    if not isinstance(results, list):
        return {}
    return {
        str(item.get("check")): item
        for item in results
        if isinstance(item, dict) and item.get("check") is not None
    }


def _check_passed(checks: Dict[str, Dict[str, str]], name: str) -> bool | None:
    item = checks.get(name)
    if item is None:
        return None
    return item.get("status") == "PASS"


def _scale_warning_lookup(scale_analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    warnings = scale_analysis.get("warnings", [])
    if not isinstance(warnings, list):
        return {}
    return {
        str(item.get("name")): item
        for item in warnings
        if isinstance(item, dict) and item.get("name") is not None
    }


def _metric_mean(report: Dict[str, Any], key: str) -> float | None:
    metric = report.get("metrics", {}).get(key)
    if not isinstance(metric, dict):
        return None
    value = metric.get("mean")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _resolve_artifact_path(
    row: Dict[str, Any],
    *,
    summary: Dict[str, Any],
    explicit_key: str,
    suffix: str,
) -> Path:
    if row.get(explicit_key):
        return Path(str(row[explicit_key])).expanduser()
    output_dir = Path(str(summary["output_dir"])).expanduser()
    preset = _safe_name(str(row["preset_name"]))
    run_name = str(row["run_name"])
    return output_dir / preset / f"{run_name}_{suffix}.json"


def _slow_fast_scale_check(
    report: Dict[str, Any],
    *,
    eps: float,
    max_ratio: float,
) -> Dict[str, Any]:
    slow = _metric_mean(report, "slow_reward")
    fast = _metric_mean(report, "fast_reward")
    if slow is None or fast is None:
        return {
            "status": "UNKNOWN",
            "ratio": None,
            "detail": "mean_slow_reward or mean_fast_reward is missing.",
        }

    slow_abs = abs(slow)
    fast_abs = abs(fast)
    if slow_abs <= eps and fast_abs <= eps:
        return {"status": "PASS", "ratio": 1.0, "detail": "both slow and fast reward scales are near zero."}
    if slow_abs <= eps:
        return {
            "status": "WARN",
            "ratio": None,
            "detail": f"abs(mean_slow_reward)<={eps} while abs(mean_fast_reward)={fast_abs:.6g}.",
        }

    ratio = max(slow_abs, fast_abs) / max(min(slow_abs, fast_abs), eps)
    return {
        "status": "PASS" if ratio <= max_ratio else "WARN",
        "ratio": ratio,
        "detail": f"scale_ratio={ratio:.6g}, max_ratio={max_ratio}.",
    }


def _classify_preset(
    row: Dict[str, Any],
    *,
    summary: Dict[str, Any],
    eps: float,
    slow_fast_max_ratio: float,
) -> Dict[str, Any]:
    preset = str(row["preset_name"])
    run_name = str(row.get("run_name") or preset)
    report_path = _resolve_artifact_path(row, summary=summary, explicit_key="report_json_path", suffix="report")
    scale_path = _resolve_artifact_path(
        row,
        summary=summary,
        explicit_key="scale_analysis_json_path",
        suffix="scale_analysis",
    )

    reject_reasons: List[str] = []
    caution_reasons: List[str] = []
    checks: Dict[str, Any] = {}
    report: Dict[str, Any] = {}
    scale_analysis: Dict[str, Any] = {}

    try:
        report = _load_json(report_path)
    except Exception as exc:  # noqa: BLE001 - artifact diagnosis should be reported per preset
        reject_reasons.append(f"report_missing_or_unreadable: {exc!r}")

    try:
        scale_analysis = _load_json(scale_path)
    except Exception as exc:  # noqa: BLE001 - keep diagnosing what is available
        caution_reasons.append(f"scale_analysis_missing_or_unreadable: {exc!r}")

    if report:
        sanity_status = str(report.get("sanity", {}).get("status") or row.get("sanity_status") or "UNKNOWN")
        checks["sanity_status"] = sanity_status
        if sanity_status != "PASS":
            reject_reasons.append(f"sanity_status={sanity_status}")

        sanity_checks = _check_lookup(report)
        for name, detail in REJECT_CHECKS.items():
            passed = _check_passed(sanity_checks, name)
            checks[name] = "UNKNOWN" if passed is None else ("PASS" if passed else "FAIL")
            if passed is None:
                reject_reasons.append(f"{name}: required sanity check is missing.")
            if passed is False:
                reject_reasons.append(f"{name}: {detail}")

        for name, detail in CAUTION_CHECKS.items():
            passed = _check_passed(sanity_checks, name)
            checks[name] = "UNKNOWN" if passed is None else ("PASS" if passed else "FAIL")
            if passed is False:
                caution_reasons.append(f"{name}: {detail}")

        slow_fast = _slow_fast_scale_check(report, eps=eps, max_ratio=slow_fast_max_ratio)
        checks["slow_fast_reward_scale"] = slow_fast
        if slow_fast["status"] == "WARN":
            caution_reasons.append(f"slow_fast_reward_scale: {slow_fast['detail']}")
    else:
        checks["sanity_status"] = "UNKNOWN"

    if scale_analysis:
        scale_warnings = _scale_warning_lookup(scale_analysis)
        for name, detail in CAUTION_SCALE_WARNINGS.items():
            item = scale_warnings.get(name)
            status = str(item.get("status")) if item else "UNKNOWN"
            checks[name] = status
            if status == "WARN":
                caution_reasons.append(f"{name}: {detail}")

    category = "reject" if reject_reasons else ("caution" if caution_reasons else "viable")
    return {
        "preset_name": preset,
        "run_name": run_name,
        "category": category,
        "report_json_path": str(report_path),
        "scale_analysis_json_path": str(scale_path),
        "reject_reasons": reject_reasons,
        "caution_reasons": caution_reasons,
        "checks": checks,
        "metrics": {
            "mean_slow_reward": _metric_mean(report, "slow_reward") if report else None,
            "mean_fast_reward": _metric_mean(report, "fast_reward") if report else None,
            "mean_total_reward": _metric_mean(report, "total_reward") if report else None,
            "scale_status": scale_analysis.get("status") if scale_analysis else None,
            "scale_warning_count": scale_analysis.get("warning_count") if scale_analysis else None,
        },
    }


def build_preset_analysis(
    *,
    summary_path: Path,
    eps: float,
    slow_fast_max_ratio: float,
) -> Dict[str, Any]:
    summary = _load_json(summary_path)
    rows = summary.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("comparison summary must contain a list-valued 'results' field.")

    results = [
        _classify_preset(
            row,
            summary=summary,
            eps=eps,
            slow_fast_max_ratio=slow_fast_max_ratio,
        )
        for row in rows
        if isinstance(row, dict)
    ]

    for failure in summary.get("failures", []):
        if not isinstance(failure, dict):
            continue
        results.append(
            {
                "preset_name": str(failure.get("preset_name", "unknown")),
                "run_name": str(failure.get("preset_name", "unknown")),
                "category": "reject",
                "report_json_path": None,
                "scale_analysis_json_path": None,
                "reject_reasons": [f"comparison_failure: {failure.get('error')!r}"],
                "caution_reasons": [],
                "checks": {"comparison_failure": "FAIL"},
                "metrics": {},
            }
        )

    counts = {
        "viable": sum(1 for item in results if item["category"] == "viable"),
        "caution": sum(1 for item in results if item["category"] == "caution"),
        "reject": sum(1 for item in results if item["category"] == "reject"),
    }
    return {
        "event": "reward_preset_viability_analysis",
        "comparison_summary_path": str(summary_path.expanduser()),
        "run_config": {
            "presets": summary.get("presets"),
            "seed": summary.get("seed"),
            "max_steps": summary.get("max_steps"),
            "max_updates": summary.get("max_updates"),
            "rollout_steps": summary.get("rollout_steps"),
            "device": summary.get("device"),
        },
        "thresholds": {
            "eps": eps,
            "slow_fast_max_ratio": slow_fast_max_ratio,
        },
        "counts": counts,
        "results": results,
        "status": "PASS" if counts["reject"] == 0 else "FAIL",
        "interpretation_note": (
            "This analysis only filters unsuitable candidate presets after short train comparison. "
            "It does not select final coefficients, run long training, or compare baselines."
        ),
    }


def format_markdown(analysis: Dict[str, Any]) -> str:
    lines = [
        "# Reward Preset Viability Analysis",
        "",
        "This report filters unsuitable candidate presets only. It does not select final coefficients.",
        "",
        "## Run Config",
        "",
    ]
    for key, value in analysis["run_config"].items():
        lines.append(f"- {key}: `{_format_value(value)}`")

    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            f"- eps: `{analysis['thresholds']['eps']}`",
            f"- slow_fast_max_ratio: `{analysis['thresholds']['slow_fast_max_ratio']}`",
            "",
            "## Category Counts",
            "",
            f"- viable: `{analysis['counts']['viable']}`",
            f"- caution: `{analysis['counts']['caution']}`",
            f"- reject: `{analysis['counts']['reject']}`",
            "",
            "## Presets",
            "",
            "| preset | category | sanity | checkpoint | dpp_terms | finite | SoC | entropy | reward_constant | slow_fast_scale | scale_warnings |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for item in analysis["results"]:
        checks = item["checks"]
        slow_fast = checks.get("slow_fast_reward_scale", {})
        slow_fast_status = slow_fast.get("status") if isinstance(slow_fast, dict) else slow_fast
        lines.append(
            "| "
            f"{item['preset_name']} | {item['category']} | "
            f"{checks.get('sanity_status', 'UNKNOWN')} | {checks.get('checkpoint_exists', 'UNKNOWN')} | "
            f"{checks.get('dpp_terms_present', 'UNKNOWN')} | {checks.get('finite_loss_reward', 'UNKNOWN')} | "
            f"{checks.get('uav_soc_bounds', 'UNKNOWN')} | {checks.get('entropy_not_collapsed', 'UNKNOWN')} | "
            f"{checks.get('total_reward_not_constant', 'UNKNOWN')} | {slow_fast_status} | "
            f"{_format_value(item['metrics'].get('scale_warning_count'))} |"
        )

    lines.extend(["", "## Reasons", ""])
    for item in analysis["results"]:
        lines.append(f"### {item['preset_name']} - {item['category']}")
        if item["reject_reasons"]:
            lines.append("")
            lines.append("Reject reasons:")
            for reason in item["reject_reasons"]:
                lines.append(f"- {reason}")
        if item["caution_reasons"]:
            lines.append("")
            lines.append("Caution reasons:")
            for reason in item["caution_reasons"]:
                lines.append(f"- {reason}")
        if not item["reject_reasons"] and not item["caution_reasons"]:
            lines.append("")
            lines.append("- No reject or caution rule fired.")
        lines.append("")

    lines.extend(["## Note", "", analysis["interpretation_note"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify candidate reward presets from comparison artifacts.")
    parser.add_argument("--summary-json", required=True, help="Comparison summary JSON from compare_reward_presets.py.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--slow-fast-max-ratio", type=float, default=1000.0)
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()

    if args.slow_fast_max_ratio <= 0.0:
        raise ValueError("slow-fast-max-ratio must be positive.")

    summary_path = Path(args.summary_json).expanduser()
    analysis = build_preset_analysis(
        summary_path=summary_path,
        eps=float(args.eps),
        slow_fast_max_ratio=float(args.slow_fast_max_ratio),
    )
    markdown = format_markdown(analysis)
    print(markdown)

    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json
        else summary_path.with_name(f"{summary_path.stem}_viability.json")
    )
    output_md = (
        Path(args.output_md).expanduser()
        if args.output_md
        else summary_path.with_name(f"{summary_path.stem}_viability.md")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown + "\n", encoding="utf-8")

    raise SystemExit(1 if args.fail_on_reject and analysis["counts"]["reject"] > 0 else 0)


if __name__ == "__main__":
    main()
