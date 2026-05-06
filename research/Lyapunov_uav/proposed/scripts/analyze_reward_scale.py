from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from check_short_train_sanity import DPP_KEYS, _is_finite_number, _load_jsonl
from report_short_train import _resolve_log_path


SCALE_TERMS = (
    "slow_reward",
    "fast_reward",
    "total_reward",
    *DPP_KEYS,
    "actor_loss",
    "critic_loss",
    "entropy",
    "approx_kl",
    "clip_frac",
)
RELATIVE_DPP_KEYS = (
    "quality_reward",
    "video_delivery_pressure",
    "battery_service_pressure",
    "charging_effect",
)


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


def _is_finite(value: Any) -> bool:
    return _is_finite_number(value)


def _extract_series(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if key in DPP_KEYS:
            terms = row.get("dpp_terms")
            value = terms.get(key) if isinstance(terms, dict) else None
        else:
            value = row.get(key)
        if _is_finite(value):
            values.append(float(value))
    return values


def _stats(values: Sequence[float]) -> Dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "std": _std(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "final": float(values[-1]) if values else None,
    }


def _ratio_series(rows: Sequence[Dict[str, Any]], numerator_key: str, denominator_key: str, eps: float) -> List[float]:
    ratios: List[float] = []
    for row in rows:
        terms = row.get("dpp_terms")
        if not isinstance(terms, dict):
            continue
        numerator = terms.get(numerator_key)
        denominator = terms.get(denominator_key)
        if not _is_finite(numerator) or not _is_finite(denominator):
            continue
        denominator_abs = abs(float(denominator))
        if denominator_abs <= eps:
            continue
        ratios.append(abs(float(numerator)) / denominator_abs)
    return ratios


def _fraction(values: Sequence[float], predicate) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if predicate(value)) / len(values))


def _add_warning(
    warnings: List[Dict[str, Any]],
    *,
    name: str,
    status: str,
    detail: str,
) -> None:
    warnings.append({"name": name, "status": status, "detail": detail})


def _charge_event_count(rows: Sequence[Dict[str, Any]]) -> int | None:
    observed = False
    count = 0
    for row in rows:
        for key in ("uav_charge_effective_sum", "charging_event_count", "charge_event_count"):
            value = row.get(key)
            if _is_finite(value):
                observed = True
                count += int(float(value))
                break
        if "uav_charge_effective" in row and isinstance(row["uav_charge_effective"], list):
            observed = True
            count += sum(1 for value in row["uav_charge_effective"] if _is_finite(value) and float(value) > 0.0)
    return count if observed else None


def _warning_rules(
    *,
    rows: Sequence[Dict[str, Any]],
    term_stats: Dict[str, Dict[str, Any]],
    ratios: Dict[str, Dict[str, Any]],
    eps: float,
    dominance_threshold: float,
    sustained_fraction: float,
    near_zero_fraction: float,
    reward_loss_min_ratio: float,
    reward_loss_max_ratio: float,
    loss_scale_eps: float,
    entropy_collapse_ratio: float,
    entropy_min: float,
    approx_kl_max: float,
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []

    quality_ratios = ratios["quality_reward_over_total_dpp_reward"]["values"]
    quality_fraction = _fraction(quality_ratios, lambda value: value >= dominance_threshold)
    _add_warning(
        warnings,
        name="quality_reward_dominates_total_dpp_reward",
        status="WARN" if quality_fraction is not None and quality_fraction >= sustained_fraction else "OK",
        detail=f"fraction={quality_fraction}, threshold={dominance_threshold}, sustained_fraction={sustained_fraction}",
    )

    video_values = [abs(value) for value in _extract_series(rows, "video_delivery_pressure")]
    video_zero_fraction = _fraction(video_values, lambda value: value <= eps)
    _add_warning(
        warnings,
        name="video_delivery_pressure_nearly_always_zero",
        status="WARN" if video_zero_fraction is not None and video_zero_fraction >= near_zero_fraction else "OK",
        detail=f"zero_fraction={video_zero_fraction}, eps={eps}, near_zero_fraction={near_zero_fraction}",
    )

    battery_values = [abs(value) for value in _extract_series(rows, "battery_service_pressure")]
    battery_zero_fraction = _fraction(battery_values, lambda value: value <= eps)
    _add_warning(
        warnings,
        name="battery_service_pressure_nearly_always_zero",
        status="WARN" if battery_zero_fraction is not None and battery_zero_fraction >= near_zero_fraction else "OK",
        detail=f"zero_fraction={battery_zero_fraction}, eps={eps}, near_zero_fraction={near_zero_fraction}",
    )

    battery_ratios = ratios["battery_service_pressure_over_total_dpp_reward"]["values"]
    battery_dominance_fraction = _fraction(battery_ratios, lambda value: value >= dominance_threshold)
    _add_warning(
        warnings,
        name="battery_service_pressure_dominates_total_dpp_reward",
        status="WARN"
        if battery_dominance_fraction is not None and battery_dominance_fraction >= sustained_fraction
        else "OK",
        detail=(
            f"fraction={battery_dominance_fraction}, threshold={dominance_threshold}, "
            f"sustained_fraction={sustained_fraction}"
        ),
    )

    charging_ratios = ratios["charging_effect_over_total_dpp_reward"]["values"]
    charging_large_fraction = _fraction(charging_ratios, lambda value: value >= dominance_threshold)
    charge_count = _charge_event_count(rows)
    charging_status = "OK"
    if charging_large_fraction is not None and charging_large_fraction >= sustained_fraction and not charge_count:
        charging_status = "WARN"
    _add_warning(
        warnings,
        name="charging_effect_large_without_charging_event",
        status=charging_status,
        detail=(
            f"large_fraction={charging_large_fraction}, threshold={dominance_threshold}, "
            f"sustained_fraction={sustained_fraction}, observed_charge_event_count={charge_count}"
        ),
    )

    reward_scale = abs(float(term_stats["total_reward"]["mean"] or 0.0))
    critic_scale = abs(float(term_stats["critic_loss"]["mean"] or 0.0))
    reward_critic_ratio = reward_scale / critic_scale if critic_scale > loss_scale_eps else None
    reward_critic_warn = (
        reward_critic_ratio is not None
        and (reward_critic_ratio < reward_loss_min_ratio or reward_critic_ratio > reward_loss_max_ratio)
    )
    _add_warning(
        warnings,
        name="total_reward_scale_vs_critic_loss",
        status="INFO" if reward_critic_ratio is None else ("WARN" if reward_critic_warn else "OK"),
        detail=(
            "critic_loss scale is near zero; ratio skipped"
            if reward_critic_ratio is None
            else (
                f"abs(mean_total_reward)/abs(mean_critic_loss)={reward_critic_ratio}, "
                f"expected_range=[{reward_loss_min_ratio}, {reward_loss_max_ratio}]"
            )
        ),
    )

    actor_scale = abs(float(term_stats["actor_loss"]["mean"] or 0.0))
    reward_actor_ratio = reward_scale / actor_scale if actor_scale > loss_scale_eps else None
    _add_warning(
        warnings,
        name="total_reward_scale_vs_actor_loss",
        status="INFO" if reward_actor_ratio is None else (
            "WARN" if reward_actor_ratio < reward_loss_min_ratio or reward_actor_ratio > reward_loss_max_ratio else "OK"
        ),
        detail=(
            "actor_loss scale is near zero; ratio skipped"
            if reward_actor_ratio is None
            else (
                f"abs(mean_total_reward)/abs(mean_actor_loss)={reward_actor_ratio}, "
                f"expected_range=[{reward_loss_min_ratio}, {reward_loss_max_ratio}]"
            )
        ),
    )

    entropy_values = _extract_series(rows, "entropy")
    entropy_status = "OK"
    entropy_detail = "not enough entropy values"
    if entropy_values:
        final_entropy = entropy_values[-1]
        initial_entropy = entropy_values[0]
        ratio = final_entropy / initial_entropy if abs(initial_entropy) > eps else None
        if final_entropy <= entropy_min or (ratio is not None and ratio <= entropy_collapse_ratio):
            entropy_status = "WARN"
        entropy_detail = (
            f"initial={initial_entropy}, final={final_entropy}, final_over_initial={ratio}, "
            f"entropy_min={entropy_min}, collapse_ratio={entropy_collapse_ratio}"
        )
    _add_warning(warnings, name="entropy_fast_collapse", status=entropy_status, detail=entropy_detail)

    approx_kl_values = _extract_series(rows, "approx_kl")
    approx_kl_status = "OK"
    approx_kl_detail = "approx_kl not present"
    if approx_kl_values:
        max_kl = max(approx_kl_values)
        approx_kl_status = "WARN" if max_kl > approx_kl_max else "OK"
        approx_kl_detail = f"max_approx_kl={max_kl}, approx_kl_max={approx_kl_max}"
    _add_warning(warnings, name="approx_kl_too_large", status=approx_kl_status, detail=approx_kl_detail)

    return warnings


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_scale_analysis(
    *,
    log_path: Path,
    eps: float,
    dominance_threshold: float,
    sustained_fraction: float,
    near_zero_fraction: float,
    reward_loss_min_ratio: float,
    reward_loss_max_ratio: float,
    loss_scale_eps: float,
    entropy_collapse_ratio: float,
    entropy_min: float,
    approx_kl_max: float,
) -> Dict[str, Any]:
    rows = _load_jsonl(log_path)
    start = next((row for row in rows if row.get("event") == "short_train_start"), {})
    done = next((row for row in reversed(rows) if row.get("event") == "short_train_done"), {})
    updates = [row for row in rows if row.get("event") == "short_train_update"]
    if not updates:
        raise ValueError(f"No short_train_update rows found in {log_path}")

    term_stats = {key: _stats(_extract_series(updates, key)) for key in SCALE_TERMS}
    relative_ratios: Dict[str, Dict[str, Any]] = {}
    for key in RELATIVE_DPP_KEYS:
        ratio_key = f"{key}_over_total_dpp_reward"
        values = _ratio_series(updates, key, "total_dpp_reward", eps)
        relative_ratios[ratio_key] = {"values": values, "stats": _stats(values)}

    warnings = _warning_rules(
        rows=updates,
        term_stats=term_stats,
        ratios=relative_ratios,
        eps=eps,
        dominance_threshold=dominance_threshold,
        sustained_fraction=sustained_fraction,
        near_zero_fraction=near_zero_fraction,
        reward_loss_min_ratio=reward_loss_min_ratio,
        reward_loss_max_ratio=reward_loss_max_ratio,
        loss_scale_eps=loss_scale_eps,
        entropy_collapse_ratio=entropy_collapse_ratio,
        entropy_min=entropy_min,
        approx_kl_max=approx_kl_max,
    )
    warning_count = sum(1 for warning in warnings if warning["status"] == "WARN")

    run_name = str(start.get("run_name") or updates[-1].get("run_name") or log_path.stem)
    analysis = {
        "run_name": run_name,
        "log_path": str(log_path),
        "device": start.get("device"),
        "seed": start.get("seed"),
        "total_steps": int(done.get("step") or updates[-1].get("step") or 0),
        "total_updates": int(done.get("updates") or updates[-1].get("update") or len(updates)),
        "term_stats": term_stats,
        "relative_dpp_ratios": relative_ratios,
        "warnings": warnings,
        "warning_count": warning_count,
        "status": "PASS" if warning_count == 0 else "WARN",
        "thresholds": {
            "eps": eps,
            "dominance_threshold": dominance_threshold,
            "sustained_fraction": sustained_fraction,
            "near_zero_fraction": near_zero_fraction,
            "reward_loss_min_ratio": reward_loss_min_ratio,
            "reward_loss_max_ratio": reward_loss_max_ratio,
            "loss_scale_eps": loss_scale_eps,
            "entropy_collapse_ratio": entropy_collapse_ratio,
            "entropy_min": entropy_min,
            "approx_kl_max": approx_kl_max,
        },
        "interpretation_note": (
            "This is a scale diagnosis only. It does not tune reward coefficients, "
            "start long training, or compare baselines."
        ),
    }
    return _sanitize_for_json(analysis)


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_markdown(analysis: Dict[str, Any]) -> str:
    lines = [
        "# Reward And DPP Scale Analysis",
        "",
        "## Run",
        "",
        f"- run_name: `{analysis['run_name']}`",
        f"- device: `{_format_value(analysis.get('device'))}`",
        f"- seed: `{_format_value(analysis.get('seed'))}`",
        f"- total_steps: `{analysis['total_steps']}`",
        f"- total_updates: `{analysis['total_updates']}`",
        f"- status: `{analysis['status']}`",
        f"- warning_count: `{analysis['warning_count']}`",
        "",
        "## Term Stats",
        "",
        "| term | mean | std | min | max | final | count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in SCALE_TERMS:
        stat = analysis["term_stats"][key]
        lines.append(
            "| "
            f"{key} | {_format_value(stat['mean'])} | {_format_value(stat['std'])} | "
            f"{_format_value(stat['min'])} | {_format_value(stat['max'])} | "
            f"{_format_value(stat['final'])} | {stat['count']} |"
        )

    lines.extend(
        [
            "",
            "## Relative DPP Ratios",
            "",
            "| ratio | mean | std | min | max | final | count |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, payload in analysis["relative_dpp_ratios"].items():
        stat = payload["stats"]
        lines.append(
            "| "
            f"{key} | {_format_value(stat['mean'])} | {_format_value(stat['std'])} | "
            f"{_format_value(stat['min'])} | {_format_value(stat['max'])} | "
            f"{_format_value(stat['final'])} | {stat['count']} |"
        )

    lines.extend(["", "## Warning Rules", "", "| status | rule | detail |", "| --- | --- | --- |"])
    for warning in analysis["warnings"]:
        lines.append(f"| {warning['status']} | {warning['name']} | {warning['detail']} |")

    lines.extend(["", "## Note", "", analysis["interpretation_note"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose short HRL train reward and DPP term scales.")
    parser.add_argument("--log-path", default=None, help="Exact JSONL log path.")
    parser.add_argument("--log-dir", default=None, help="Directory containing short train JSONL logs.")
    parser.add_argument("--run-name", default=None, help="Run name without the .jsonl suffix.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--dominance-threshold", type=float, default=0.8)
    parser.add_argument("--sustained-fraction", type=float, default=0.8)
    parser.add_argument("--near-zero-fraction", type=float, default=0.8)
    parser.add_argument("--reward-loss-min-ratio", type=float, default=1e-3)
    parser.add_argument("--reward-loss-max-ratio", type=float, default=1e3)
    parser.add_argument("--loss-scale-eps", type=float, default=1e-6)
    parser.add_argument("--entropy-collapse-ratio", type=float, default=0.1)
    parser.add_argument("--entropy-min", type=float, default=1e-8)
    parser.add_argument("--approx-kl-max", type=float, default=0.05)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    log_path = _resolve_log_path(
        Path(args.log_path) if args.log_path else None,
        Path(args.log_dir) if args.log_dir else None,
        args.run_name,
    )
    analysis = build_scale_analysis(
        log_path=log_path,
        eps=float(args.eps),
        dominance_threshold=float(args.dominance_threshold),
        sustained_fraction=float(args.sustained_fraction),
        near_zero_fraction=float(args.near_zero_fraction),
        reward_loss_min_ratio=float(args.reward_loss_min_ratio),
        reward_loss_max_ratio=float(args.reward_loss_max_ratio),
        loss_scale_eps=float(args.loss_scale_eps),
        entropy_collapse_ratio=float(args.entropy_collapse_ratio),
        entropy_min=float(args.entropy_min),
        approx_kl_max=float(args.approx_kl_max),
    )

    markdown = format_markdown(analysis)
    print(markdown)

    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md).expanduser()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown + "\n", encoding="utf-8")

    raise SystemExit(1 if args.fail_on_warning and analysis["warning_count"] > 0 else 0)


if __name__ == "__main__":
    main()
