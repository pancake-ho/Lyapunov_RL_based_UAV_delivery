from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence


def _add_import_paths() -> None:
    script_path = Path(__file__).resolve()
    proposed_dir = script_path.parents[1]
    lyapunov_dir = proposed_dir.parent
    for path in (lyapunov_dir, proposed_dir, script_path.parent):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_add_import_paths()

from analyze_preset_comparison import build_preset_analysis, format_markdown as format_viability_markdown
from compare_reward_presets import format_comparison_markdown, run_comparison


NUMERIC_AGGREGATE_KEYS = (
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
    "warning_count",
)


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


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _selected_presets(
    viability: Dict[str, Any],
    *,
    include_categories: Sequence[str],
    explicit_presets: Sequence[str] | None,
) -> List[Dict[str, Any]]:
    include = set(include_categories)
    explicit = set(explicit_presets or [])
    selected = []
    for row in viability.get("results", []):
        if not isinstance(row, dict):
            continue
        preset = str(row.get("preset_name", ""))
        category = str(row.get("category", ""))
        if not preset or category not in include:
            continue
        if explicit and preset not in explicit:
            continue
        selected.append(
            {
                "preset_name": preset,
                "source_viability": category,
                "source_reject_reasons": row.get("reject_reasons", []),
                "source_caution_reasons": row.get("caution_reasons", []),
            }
        )
    return selected


def _comparison_args_for_seed(
    args: argparse.Namespace,
    *,
    presets: Sequence[str],
    seed: int,
    seed_name: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        presets=list(presets),
        seed=int(seed),
        max_steps=int(args.max_steps),
        max_updates=int(args.max_updates),
        rollout_steps=int(args.rollout_steps),
        device=str(args.device),
        hidden_dim=int(args.hidden_dim),
        checkpoint_interval=int(args.checkpoint_interval),
        run_name=f"{_safe_name(args.run_name)}_{seed_name}",
        log_dir=str(Path(args.log_dir).expanduser() / seed_name),
        checkpoint_dir=str(Path(args.checkpoint_dir).expanduser() / seed_name),
        output_dir=str(Path(args.output_dir).expanduser() / seed_name),
        max_queue=float(args.max_queue),
        entropy_min=float(args.entropy_min),
        reward_tol=float(args.reward_tol),
        eps=float(args.eps),
        dominance_threshold=float(args.dominance_threshold),
        sustained_fraction=float(args.sustained_fraction),
        near_zero_fraction=float(args.near_zero_fraction),
        reward_loss_min_ratio=float(args.reward_loss_min_ratio),
        reward_loss_max_ratio=float(args.reward_loss_max_ratio),
        loss_scale_eps=float(args.loss_scale_eps),
        entropy_collapse_ratio=float(args.entropy_collapse_ratio),
        approx_kl_max=float(args.approx_kl_max),
        stop_on_fail=bool(args.stop_on_fail),
    )


def _write_seed_artifacts(
    *,
    summary: Dict[str, Any],
    seed_output_dir: Path,
    run_name: str,
    analyzer_eps: float,
    slow_fast_max_ratio: float,
) -> Dict[str, Any]:
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = seed_output_dir / f"{run_name}_summary.json"
    summary_md = seed_output_dir / f"{run_name}_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md.write_text(format_comparison_markdown(summary), encoding="utf-8")

    analysis = build_preset_analysis(
        summary_path=summary_json,
        eps=analyzer_eps,
        slow_fast_max_ratio=slow_fast_max_ratio,
    )
    viability_json = seed_output_dir / f"{run_name}_viability.json"
    viability_md = seed_output_dir / f"{run_name}_viability.md"
    viability_markdown = format_viability_markdown(analysis)
    viability_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    viability_md.write_text(viability_markdown + "\n", encoding="utf-8")
    return {
        "summary_json_path": str(summary_json),
        "summary_md_path": str(summary_md),
        "viability_json_path": str(viability_json),
        "viability_md_path": str(viability_md),
        "analysis": analysis,
    }


def _row_by_preset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("preset_name")): row for row in rows if isinstance(row, dict)}


def _aggregate_row(
    *,
    preset: str,
    seed: int,
    source_viability: str,
    comparison_row: Dict[str, Any] | None,
    viability_row: Dict[str, Any] | None,
    seed_artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    comparison_row = comparison_row or {}
    viability_row = viability_row or {}
    return {
        "preset_name": preset,
        "seed": int(seed),
        "source_viability": source_viability,
        "viability": viability_row.get("category", "reject"),
        "sanity_status": comparison_row.get("sanity_status", "FAIL"),
        "warning_count": comparison_row.get("warning_count"),
        "final_total_reward": comparison_row.get("final_total_reward"),
        "mean_total_reward": comparison_row.get("mean_total_reward"),
        "mean_slow_reward": comparison_row.get("mean_slow_reward"),
        "mean_fast_reward": comparison_row.get("mean_fast_reward"),
        "mean_total_dpp_reward": comparison_row.get("mean_total_dpp_reward"),
        "mean_quality_reward": comparison_row.get("mean_quality_reward"),
        "mean_video_delivery_pressure": comparison_row.get("mean_video_delivery_pressure"),
        "mean_battery_service_pressure": comparison_row.get("mean_battery_service_pressure"),
        "mean_charging_effect": comparison_row.get("mean_charging_effect"),
        "mean_actor_loss": comparison_row.get("mean_actor_loss"),
        "mean_critic_loss": comparison_row.get("mean_critic_loss"),
        "mean_entropy": comparison_row.get("mean_entropy"),
        "checkpoint_exists": comparison_row.get("checkpoint_exists", False),
        "run_name": comparison_row.get("run_name"),
        "log_path": comparison_row.get("log_path"),
        "checkpoint_dir": comparison_row.get("checkpoint_dir"),
        "report_json_path": comparison_row.get("report_json_path"),
        "scale_analysis_json_path": comparison_row.get("scale_analysis_json_path"),
        "seed_summary_json_path": seed_artifacts["summary_json_path"],
        "seed_viability_json_path": seed_artifacts["viability_json_path"],
        "reject_reasons": viability_row.get("reject_reasons", []),
        "caution_reasons": viability_row.get("caution_reasons", []),
    }


def _preset_stats(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    presets = sorted({str(row["preset_name"]) for row in rows})
    stats = []
    for preset in presets:
        preset_rows = [row for row in rows if row["preset_name"] == preset]
        payload: Dict[str, Any] = {
            "preset_name": preset,
            "run_count": len(preset_rows),
            "sanity_pass_count": sum(1 for row in preset_rows if row.get("sanity_status") == "PASS"),
            "checkpoint_exists_count": sum(1 for row in preset_rows if row.get("checkpoint_exists") is True),
            "viability_counts": {
                "viable": sum(1 for row in preset_rows if row.get("viability") == "viable"),
                "caution": sum(1 for row in preset_rows if row.get("viability") == "caution"),
                "reject": sum(1 for row in preset_rows if row.get("viability") == "reject"),
            },
        }
        for key in NUMERIC_AGGREGATE_KEYS:
            values = [
                number
                for number in (_finite_number(row.get(key)) for row in preset_rows)
                if number is not None
            ]
            payload[f"{key}_mean"] = _mean(values)
            payload[f"{key}_std"] = _std(values)
        stats.append(payload)
    return stats


def _format_aggregate_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Reward Preset Repeat Short Run Summary",
        "",
        "This is a multi-seed short-run stability check only. It does not select final coefficients, run long training, or compare baselines.",
        "",
        "## Run Config",
        "",
        f"- source_viability_json: `{summary['source_viability_json_path']}`",
        f"- selected_presets: `{', '.join(summary['selected_presets'])}`",
        f"- excluded_reject_presets: `{', '.join(summary['excluded_reject_presets'])}`",
        f"- seeds: `{', '.join(str(seed) for seed in summary['seeds'])}`",
        f"- max_steps: `{summary['max_steps']}`",
        f"- max_updates: `{summary['max_updates']}`",
        f"- rollout_steps: `{summary['rollout_steps']}`",
        f"- device: `{summary['device']}`",
        "",
        "## Preset-Seed Results",
        "",
        "| preset | seed | viability | sanity | warnings | checkpoint | final_total_reward | mean_total_reward | mean_slow_reward | mean_fast_reward | mean_total_dpp | mean_quality | mean_video_pressure | mean_battery_pressure | mean_charging | mean_actor_loss | mean_critic_loss | mean_entropy |",
        "| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["results"]:
        lines.append(
            "| "
            f"{row['preset_name']} | {row['seed']} | {row['viability']} | {row['sanity_status']} | "
            f"{_format_value(row['warning_count'])} | {_format_value(row['checkpoint_exists'])} | "
            f"{_format_value(row['final_total_reward'])} | {_format_value(row['mean_total_reward'])} | "
            f"{_format_value(row['mean_slow_reward'])} | {_format_value(row['mean_fast_reward'])} | "
            f"{_format_value(row['mean_total_dpp_reward'])} | {_format_value(row['mean_quality_reward'])} | "
            f"{_format_value(row['mean_video_delivery_pressure'])} | "
            f"{_format_value(row['mean_battery_service_pressure'])} | "
            f"{_format_value(row['mean_charging_effect'])} | {_format_value(row['mean_actor_loss'])} | "
            f"{_format_value(row['mean_critic_loss'])} | {_format_value(row['mean_entropy'])} |"
        )

    lines.extend(
        [
            "",
            "## Preset Statistics",
            "",
            "| preset | runs | viable | caution | reject | sanity_pass | checkpoint | mean_total_reward_mean | mean_total_reward_std | mean_total_dpp_mean | mean_entropy_mean | warning_count_mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["preset_stats"]:
        counts = row["viability_counts"]
        lines.append(
            "| "
            f"{row['preset_name']} | {row['run_count']} | {counts['viable']} | {counts['caution']} | "
            f"{counts['reject']} | {row['sanity_pass_count']} | {row['checkpoint_exists_count']} | "
            f"{_format_value(row['mean_total_reward_mean'])} | {_format_value(row['mean_total_reward_std'])} | "
            f"{_format_value(row['mean_total_dpp_reward_mean'])} | {_format_value(row['mean_entropy_mean'])} | "
            f"{_format_value(row['warning_count_mean'])} |"
        )

    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- seed={failure['seed']}: {failure['error']}")

    lines.extend(["", "## Note", "", summary["interpretation_note"], ""])
    return "\n".join(lines) + "\n"


def run_repeats(args: argparse.Namespace) -> Dict[str, Any]:
    viability_path = Path(args.viability_json).expanduser()
    source_viability = _load_json(viability_path)
    selected = _selected_presets(
        source_viability,
        include_categories=args.include_categories,
        explicit_presets=args.presets,
    )
    if not selected:
        raise ValueError("No presets selected. Check --viability-json, --include-categories, and --presets.")

    selected_presets = [row["preset_name"] for row in selected]
    selected_by_preset = {row["preset_name"]: row for row in selected}
    source_results = source_viability.get("results", [])
    excluded_reject_presets = [
        str(row.get("preset_name"))
        for row in source_results
        if isinstance(row, dict) and row.get("category") == "reject"
    ]

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    seed_artifacts: List[Dict[str, Any]] = []

    for seed in args.seeds:
        seed_int = int(seed)
        seed_name = f"seed_{seed_int:04d}" if seed_int >= 0 else f"seed_neg{abs(seed_int):04d}"
        comparison_args = _comparison_args_for_seed(
            args,
            presets=selected_presets,
            seed=seed_int,
            seed_name=seed_name,
        )
        try:
            comparison = run_comparison(comparison_args)
        except Exception as exc:  # noqa: BLE001 - aggregate should preserve seed-level failures
            failures.append({"seed": seed_int, "error": repr(exc)})
            if args.stop_on_fail:
                break
            continue

        artifact_payload = _write_seed_artifacts(
            summary=comparison,
            seed_output_dir=Path(comparison_args.output_dir).expanduser().resolve(),
            run_name=str(comparison_args.run_name),
            analyzer_eps=float(args.analyzer_eps),
            slow_fast_max_ratio=float(args.slow_fast_max_ratio),
        )
        seed_artifacts.append(
            {
                "seed": seed_int,
                "summary_json_path": artifact_payload["summary_json_path"],
                "summary_md_path": artifact_payload["summary_md_path"],
                "viability_json_path": artifact_payload["viability_json_path"],
                "viability_md_path": artifact_payload["viability_md_path"],
            }
        )

        comparison_by_preset = _row_by_preset(comparison.get("results", []))
        viability_by_preset = _row_by_preset(artifact_payload["analysis"].get("results", []))
        for preset in selected_presets:
            results.append(
                _aggregate_row(
                    preset=preset,
                    seed=seed_int,
                    source_viability=selected_by_preset[preset]["source_viability"],
                    comparison_row=comparison_by_preset.get(preset),
                    viability_row=viability_by_preset.get(preset),
                    seed_artifacts=artifact_payload,
                )
            )

        if comparison.get("failures"):
            failures.extend({"seed": seed_int, **failure} for failure in comparison["failures"])
            if args.stop_on_fail:
                break

    summary = {
        "event": "reward_preset_repeat_short_runs",
        "source_viability_json_path": str(viability_path),
        "source_viability_counts": source_viability.get("counts"),
        "include_categories": list(args.include_categories),
        "selected_presets": selected_presets,
        "excluded_reject_presets": excluded_reject_presets,
        "seeds": [int(seed) for seed in args.seeds],
        "max_steps": int(args.max_steps),
        "max_updates": int(args.max_updates),
        "rollout_steps": int(args.rollout_steps),
        "device": str(args.device),
        "log_dir": str(Path(args.log_dir).expanduser().resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
        "output_dir": str(output_root),
        "seed_artifacts": seed_artifacts,
        "results": results,
        "preset_stats": _preset_stats(results),
        "failures": failures,
        "status": "PASS" if not failures and all(row["viability"] != "reject" for row in results) else "FAIL",
        "interpretation_note": (
            "This repeat comparison only checks short-run multi-seed stability for presets that were "
            "previously classified as viable or caution. It does not select final coefficients, "
            "run long training, or compare baselines."
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat short HRL reward-preset runs over multiple seeds.")
    parser.add_argument("--viability-json", required=True, help="Output JSON from analyze_preset_comparison.py.")
    parser.add_argument("--include-categories", nargs="+", default=["viable", "caution"])
    parser.add_argument("--presets", nargs="+", default=None, help="Optional subset after reject filtering.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-updates", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--run-name", default="reward_preset_repeat")
    parser.add_argument("--log-dir", default="Lyapunov_uav/proposed/outputs/preset_repeat/logs")
    parser.add_argument("--checkpoint-dir", default="Lyapunov_uav/proposed/outputs/preset_repeat/checkpoints")
    parser.add_argument("--output-dir", default="Lyapunov_uav/proposed/outputs/preset_repeat/reports")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--max-queue", type=float, default=100.0)
    parser.add_argument("--entropy-min", type=float, default=1e-8)
    parser.add_argument("--reward-tol", type=float, default=1e-8)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--dominance-threshold", type=float, default=0.8)
    parser.add_argument("--sustained-fraction", type=float, default=0.8)
    parser.add_argument("--near-zero-fraction", type=float, default=0.8)
    parser.add_argument("--reward-loss-min-ratio", type=float, default=1e-3)
    parser.add_argument("--reward-loss-max-ratio", type=float, default=1e3)
    parser.add_argument("--loss-scale-eps", type=float, default=1e-6)
    parser.add_argument("--entropy-collapse-ratio", type=float, default=0.1)
    parser.add_argument("--approx-kl-max", type=float, default=0.05)
    parser.add_argument("--analyzer-eps", type=float, default=1e-8)
    parser.add_argument("--slow-fast-max-ratio", type=float, default=1000.0)
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive.")
    if args.max_updates <= 0:
        raise ValueError("max-updates must be positive.")
    if args.rollout_steps <= 1:
        raise ValueError("rollout-steps must be greater than 1.")
    if args.slow_fast_max_ratio <= 0.0:
        raise ValueError("slow-fast-max-ratio must be positive.")

    summary = run_repeats(args)
    markdown = _format_aggregate_markdown(summary)
    print(markdown)

    output_root = Path(args.output_dir).expanduser()
    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json
        else output_root / f"{_safe_name(args.run_name)}_aggregate_summary.json"
    )
    output_md = (
        Path(args.output_md).expanduser()
        if args.output_md
        else output_root / f"{_safe_name(args.run_name)}_aggregate_summary.md"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")

    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
