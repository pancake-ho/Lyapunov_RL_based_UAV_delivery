from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence


CATEGORY_LONG_RUN = "long_run_candidate"
CATEGORY_SHORT_ONLY = "short_run_only"
CATEGORY_REJECT = "reject"

REQUIRED_SANITY_CHECKS = (
    "finite_loss_reward",
    "dpp_terms_present",
    "uav_soc_bounds",
    "checkpoint_exists",
)

NUMERIC_METRICS = (
    "warning_count",
    "final_total_reward",
    "mean_total_reward",
    "mean_slow_reward",
    "mean_fast_reward",
    "mean_total_dpp_reward",
    "mean_quality_reward",
    "mean_video_delivery_pressure",
    "mean_battery_service_pressure",
    "mean_charging_effect",
    "mean_actor_loss",
    "mean_critic_loss",
    "mean_entropy",
)

REPORT_SUMMARY_KEYS = (
    "uav_soc",
    "queue",
    "virtual_queue",
    "uav_virtual_queue",
)

REPORT_FINAL_METRICS = (
    "total_reward",
    "actor_loss",
    "critic_loss",
    "entropy",
)


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


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    mean = float(sum(values) / len(values))
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _stats(values: Sequence[float]) -> Dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "std": _std(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "final": float(values[-1]) if values else None,
    }


def _summary_from_reports(reports: Sequence[Dict[str, Any]], key: str) -> Dict[str, float | int | None]:
    mins = []
    means = []
    maxes = []
    for report in reports:
        summary = report.get(key)
        if not isinstance(summary, dict):
            continue
        min_value = _finite_number(summary.get("min"))
        mean_value = _finite_number(summary.get("mean"))
        max_value = _finite_number(summary.get("max"))
        if min_value is not None:
            mins.append(min_value)
        if mean_value is not None:
            means.append(mean_value)
        if max_value is not None:
            maxes.append(max_value)
    return {
        "count": len(means),
        "min": min(mins) if mins else None,
        "mean": _mean(means),
        "mean_std": _std(means),
        "max": max(maxes) if maxes else None,
    }


def _report_final_metric_stats(reports: Sequence[Dict[str, Any]], key: str) -> Dict[str, float | int | None]:
    values = []
    for report in reports:
        metric = report.get("metrics", {}).get(key)
        if not isinstance(metric, dict):
            continue
        value = _finite_number(metric.get("final"))
        if value is not None:
            values.append(value)
    return _stats(values)


def _load_optional_json(path_value: Any) -> Dict[str, Any] | None:
    if not path_value:
        return None
    try:
        return _load_json(Path(str(path_value)))
    except Exception:
        return None


def _sanity_lookup(report: Dict[str, Any]) -> Dict[str, str]:
    results = report.get("sanity", {}).get("results", [])
    if not isinstance(results, list):
        return {}
    return {
        str(item.get("check")): str(item.get("status"))
        for item in results
        if isinstance(item, dict) and item.get("check") is not None
    }


def _scale_warning_statuses(scale_analysis: Dict[str, Any]) -> Dict[str, str]:
    warnings = scale_analysis.get("warnings", [])
    if not isinstance(warnings, list):
        return {}
    return {
        str(item.get("name")): str(item.get("status"))
        for item in warnings
        if isinstance(item, dict) and item.get("name") is not None
    }


def _count_warning(scale_analyses: Sequence[Dict[str, Any]], name: str) -> int:
    return sum(1 for analysis in scale_analyses if _scale_warning_statuses(analysis).get(name) == "WARN")


def _metric_values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = _finite_number(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _quality_ratio_values(rows: Sequence[Dict[str, Any]]) -> List[float]:
    ratios = []
    for row in rows:
        quality = _finite_number(row.get("mean_quality_reward"))
        total = _finite_number(row.get("mean_total_dpp_reward"))
        if quality is None or total is None or abs(total) <= 1e-8:
            continue
        ratios.append(abs(quality) / abs(total))
    return ratios


def _all_near_zero(values: Sequence[float], eps: float) -> bool:
    return bool(values) and all(abs(value) <= eps for value in values)


def _candidate_score(summary: Dict[str, Any]) -> float:
    metrics = summary["metrics"]
    sanity_rate = float(summary["sanity_pass_rate"] or 0.0)
    checkpoint_rate = float(summary["checkpoint_success_rate"] or 0.0)
    warning_mean = float(metrics["warning_count"]["mean"] or 0.0)
    entropy_mean = abs(float(metrics["mean_entropy"]["mean"] or 0.0))
    reward_mean = float(metrics["mean_total_reward"]["mean"] or 0.0)
    return reward_mean + 0.01 * entropy_mean + 100.0 * sanity_rate + 50.0 * checkpoint_rate - 10.0 * warning_mean


def _classify_preset(
    *,
    preset: str,
    rows: Sequence[Dict[str, Any]],
    reports: Sequence[Dict[str, Any]],
    scale_analyses: Sequence[Dict[str, Any]],
    repeated_warning_threshold: float,
    high_warning_mean: float,
    quality_dominance_threshold: float,
    near_zero_eps: float,
) -> Dict[str, Any]:
    seed_count = len(rows)
    viability_counts = {
        "viable": sum(1 for row in rows if row.get("viability") == "viable"),
        "caution": sum(1 for row in rows if row.get("viability") == "caution"),
        "reject": sum(1 for row in rows if row.get("viability") == "reject"),
    }
    sanity_pass_count = sum(1 for row in rows if row.get("sanity_status") == "PASS")
    checkpoint_count = sum(1 for row in rows if row.get("checkpoint_exists") is True)

    metrics = {key: _stats(_metric_values(rows, key)) for key in NUMERIC_METRICS}
    report_summaries = {key: _summary_from_reports(reports, key) for key in REPORT_SUMMARY_KEYS}
    final_metrics = {key: _report_final_metric_stats(reports, key) for key in REPORT_FINAL_METRICS}

    required_check_failures: Dict[str, int] = {name: 0 for name in REQUIRED_SANITY_CHECKS}
    for report in reports:
        checks = _sanity_lookup(report)
        for name in REQUIRED_SANITY_CHECKS:
            if checks.get(name) != "PASS":
                required_check_failures[name] += 1

    scale_warning_counts: Dict[str, int] = {}
    for analysis in scale_analyses:
        for name, status in _scale_warning_statuses(analysis).items():
            if status == "WARN":
                scale_warning_counts[name] = scale_warning_counts.get(name, 0) + 1

    entropy_warn_count = _count_warning(scale_analyses, "entropy_fast_collapse")
    quality_dominance_count = _count_warning(scale_analyses, "quality_reward_dominates_total_dpp_reward")
    battery_zero_count = _count_warning(scale_analyses, "battery_service_pressure_nearly_always_zero")
    repeated_warning_count = sum(
        1
        for count in scale_warning_counts.values()
        if seed_count > 0 and count / seed_count >= repeated_warning_threshold
    )

    quality_ratios = _quality_ratio_values(rows)
    battery_values = _metric_values(rows, "mean_battery_service_pressure")

    reject_reasons: List[str] = []
    caution_reasons: List[str] = []

    if viability_counts["reject"] > 0:
        reject_reasons.append(f"reject_viability_count={viability_counts['reject']}")
    if sanity_pass_count < seed_count:
        reject_reasons.append(f"sanity_pass_rate={sanity_pass_count}/{seed_count}")
    if checkpoint_count < seed_count:
        reject_reasons.append(f"checkpoint_success_rate={checkpoint_count}/{seed_count}")
    for name, count in required_check_failures.items():
        if count > 0:
            reject_reasons.append(f"{name}_fail_or_missing_count={count}")

    soc = report_summaries["uav_soc"]
    if soc["min"] is None or soc["max"] is None:
        reject_reasons.append("uav_soc_summary_missing")
    elif float(soc["min"]) < 0.0 or float(soc["max"]) > 100.0:
        reject_reasons.append(f"uav_soc_out_of_range=[{soc['min']}, {soc['max']}]")

    if float(metrics["warning_count"]["mean"] or 0.0) >= high_warning_mean and repeated_warning_count > 0:
        caution_reasons.append(
            f"repeated_high_warning_count={repeated_warning_count}, warning_mean={metrics['warning_count']['mean']}"
        )
    if entropy_warn_count > 0:
        caution_reasons.append(f"entropy_collapse_warning_count={entropy_warn_count}/{seed_count}")
        if seed_count > 0 and entropy_warn_count / seed_count >= repeated_warning_threshold:
            caution_reasons.append("entropy collapse repeated across seeds; exclude if reproduced in longer warmup.")
    if quality_dominance_count > 0:
        caution_reasons.append(f"quality_dominance_warning_count={quality_dominance_count}/{seed_count}")
    if (
        preset == "quality_oriented"
        and quality_ratios
        and _mean(quality_ratios) is not None
        and float(_mean(quality_ratios)) >= quality_dominance_threshold
    ):
        caution_reasons.append(
            f"quality_oriented_hold: mean_quality_over_total_dpp={_mean(quality_ratios):.6g}"
        )
    if battery_zero_count > 0 or _all_near_zero(battery_values, near_zero_eps):
        caution_reasons.append("battery_service_pressure_near_zero; battery feasibility signal may be underrepresented.")

    if reject_reasons:
        category = CATEGORY_REJECT
    elif caution_reasons:
        category = CATEGORY_SHORT_ONLY
    else:
        category = CATEGORY_LONG_RUN

    summary = {
        "preset_name": preset,
        "category": category,
        "seed_count": seed_count,
        "seeds": [int(row["seed"]) for row in rows if "seed" in row],
        "sanity_pass_rate": sanity_pass_count / seed_count if seed_count else 0.0,
        "checkpoint_success_rate": checkpoint_count / seed_count if seed_count else 0.0,
        "viability_counts": viability_counts,
        "metrics": metrics,
        "final_metrics": final_metrics,
        "queue_and_battery": report_summaries,
        "scale_warning_counts": scale_warning_counts,
        "quality_over_total_dpp": _stats(quality_ratios),
        "reject_reasons": reject_reasons,
        "caution_reasons": caution_reasons,
    }
    summary["score"] = _candidate_score(summary)
    return summary


def analyze_candidates(
    *,
    aggregate_path: Path,
    max_candidates: int,
    repeated_warning_threshold: float,
    high_warning_mean: float,
    quality_dominance_threshold: float,
    near_zero_eps: float,
) -> Dict[str, Any]:
    aggregate = _load_json(aggregate_path)
    rows = [row for row in aggregate.get("results", []) if isinstance(row, dict)]
    if not rows:
        raise ValueError("aggregate summary must contain non-empty results.")

    by_preset = sorted({str(row["preset_name"]) for row in rows})
    preset_summaries = []
    for preset in by_preset:
        preset_rows = [row for row in rows if row.get("preset_name") == preset]
        reports = [
            report
            for report in (_load_optional_json(row.get("report_json_path")) for row in preset_rows)
            if report is not None
        ]
        scale_analyses = [
            analysis
            for analysis in (_load_optional_json(row.get("scale_analysis_json_path")) for row in preset_rows)
            if analysis is not None
        ]
        preset_summaries.append(
            _classify_preset(
                preset=preset,
                rows=preset_rows,
                reports=reports,
                scale_analyses=scale_analyses,
                repeated_warning_threshold=repeated_warning_threshold,
                high_warning_mean=high_warning_mean,
                quality_dominance_threshold=quality_dominance_threshold,
                near_zero_eps=near_zero_eps,
            )
        )

    long_candidates = [item for item in preset_summaries if item["category"] == CATEGORY_LONG_RUN]
    long_candidates.sort(key=lambda item: item["score"], reverse=True)
    recommended = long_candidates[:max_candidates]

    return {
        "event": "reward_preset_long_run_candidate_analysis",
        "aggregate_summary_path": str(aggregate_path.expanduser()),
        "source_run_config": {
            "selected_presets": aggregate.get("selected_presets"),
            "seeds": aggregate.get("seeds"),
            "max_steps": aggregate.get("max_steps"),
            "max_updates": aggregate.get("max_updates"),
            "rollout_steps": aggregate.get("rollout_steps"),
            "device": aggregate.get("device"),
        },
        "thresholds": {
            "max_candidates": max_candidates,
            "repeated_warning_threshold": repeated_warning_threshold,
            "high_warning_mean": high_warning_mean,
            "quality_dominance_threshold": quality_dominance_threshold,
            "near_zero_eps": near_zero_eps,
        },
        "recommended_long_run_candidates": [item["preset_name"] for item in recommended],
        "preset_summaries": preset_summaries,
        "status": "PASS" if recommended else "NO_CANDIDATE",
        "interpretation_note": (
            "This analysis narrows presets for the next longer training candidate run. "
            "It does not claim final performance, tune coefficients, or compare baselines."
        ),
    }


def format_markdown(analysis: Dict[str, Any]) -> str:
    lines = [
        "# Reward Preset Long-Run Candidate Analysis",
        "",
        "This report narrows multi-seed short-run presets for the next longer candidate run only.",
        "",
        "## Recommendation",
        "",
        f"- recommended_long_run_candidates: `{', '.join(analysis['recommended_long_run_candidates']) or 'none'}`",
        f"- status: `{analysis['status']}`",
        "",
        "## Source",
        "",
    ]
    for key, value in analysis["source_run_config"].items():
        lines.append(f"- {key}: `{_format_value(value)}`")

    lines.extend(
        [
            "",
            "## Preset Summary",
            "",
            "| preset | category | seeds | sanity_rate | checkpoint_rate | viable/caution/reject | warn_mean | total_reward_mean | total_reward_std | total_reward_final | total_dpp_mean | quality_mean | video_pressure_mean | battery_pressure_mean | entropy_mean | entropy_final | score |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in analysis["preset_summaries"]:
        counts = item["viability_counts"]
        metrics = item["metrics"]
        final_metrics = item["final_metrics"]
        lines.append(
            "| "
            f"{item['preset_name']} | {item['category']} | {item['seed_count']} | "
            f"{_format_value(item['sanity_pass_rate'])} | {_format_value(item['checkpoint_success_rate'])} | "
            f"{counts['viable']}/{counts['caution']}/{counts['reject']} | "
            f"{_format_value(metrics['warning_count']['mean'])} | "
            f"{_format_value(metrics['mean_total_reward']['mean'])} | "
            f"{_format_value(metrics['mean_total_reward']['std'])} | "
            f"{_format_value(final_metrics['total_reward']['mean'])} | "
            f"{_format_value(metrics['mean_total_dpp_reward']['mean'])} | "
            f"{_format_value(metrics['mean_quality_reward']['mean'])} | "
            f"{_format_value(metrics['mean_video_delivery_pressure']['mean'])} | "
            f"{_format_value(metrics['mean_battery_service_pressure']['mean'])} | "
            f"{_format_value(metrics['mean_entropy']['mean'])} | "
            f"{_format_value(final_metrics['entropy']['mean'])} | {_format_value(item['score'])} |"
        )

    lines.extend(
        [
            "",
            "## Loss And Entropy Final Metrics",
            "",
            "| preset | actor_loss_mean | actor_loss_std | actor_loss_final | critic_loss_mean | critic_loss_std | critic_loss_final | entropy_mean | entropy_std | entropy_final |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in analysis["preset_summaries"]:
        metrics = item["metrics"]
        final_metrics = item["final_metrics"]
        lines.append(
            "| "
            f"{item['preset_name']} | "
            f"{_format_value(metrics['mean_actor_loss']['mean'])} | "
            f"{_format_value(metrics['mean_actor_loss']['std'])} | "
            f"{_format_value(final_metrics['actor_loss']['mean'])} | "
            f"{_format_value(metrics['mean_critic_loss']['mean'])} | "
            f"{_format_value(metrics['mean_critic_loss']['std'])} | "
            f"{_format_value(final_metrics['critic_loss']['mean'])} | "
            f"{_format_value(metrics['mean_entropy']['mean'])} | "
            f"{_format_value(metrics['mean_entropy']['std'])} | "
            f"{_format_value(final_metrics['entropy']['mean'])} |"
        )

    lines.extend(["", "## Queue And Battery", ""])
    for item in analysis["preset_summaries"]:
        lines.append(f"### {item['preset_name']}")
        for key, summary in item["queue_and_battery"].items():
            lines.append(
                f"- {key}: min=`{_format_value(summary['min'])}`, "
                f"mean=`{_format_value(summary['mean'])}`, max=`{_format_value(summary['max'])}`"
            )
        lines.append("")

    lines.extend(["## Reasons", ""])
    for item in analysis["preset_summaries"]:
        lines.append(f"### {item['preset_name']} - {item['category']}")
        if item["reject_reasons"]:
            lines.append("Reject reasons:")
            for reason in item["reject_reasons"]:
                lines.append(f"- {reason}")
        if item["caution_reasons"]:
            lines.append("Caution reasons:")
            for reason in item["caution_reasons"]:
                lines.append(f"- {reason}")
        if not item["reject_reasons"] and not item["caution_reasons"]:
            lines.append("- No reject or caution rule fired.")
        lines.append("")

    lines.extend(["## Note", "", analysis["interpretation_note"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose 1-2 long-run candidate presets from repeat short runs.")
    parser.add_argument("--aggregate-json", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--repeated-warning-threshold", type=float, default=1.0)
    parser.add_argument("--high-warning-mean", type=float, default=2.0)
    parser.add_argument("--quality-dominance-threshold", type=float, default=0.8)
    parser.add_argument("--near-zero-eps", type=float, default=1e-8)
    args = parser.parse_args()

    if args.max_candidates <= 0 or args.max_candidates > 2:
        raise ValueError("max-candidates must be 1 or 2.")
    if args.repeated_warning_threshold <= 0.0 or args.repeated_warning_threshold > 1.0:
        raise ValueError("repeated-warning-threshold must be in (0, 1].")

    aggregate_path = Path(args.aggregate_json).expanduser()
    analysis = analyze_candidates(
        aggregate_path=aggregate_path,
        max_candidates=int(args.max_candidates),
        repeated_warning_threshold=float(args.repeated_warning_threshold),
        high_warning_mean=float(args.high_warning_mean),
        quality_dominance_threshold=float(args.quality_dominance_threshold),
        near_zero_eps=float(args.near_zero_eps),
    )
    markdown = format_markdown(analysis)
    print(markdown)

    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json
        else aggregate_path.with_name(f"{aggregate_path.stem}_candidate_analysis.json")
    )
    output_md = (
        Path(args.output_md).expanduser()
        if args.output_md
        else aggregate_path.with_name(f"{aggregate_path.stem}_candidate_analysis.md")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
