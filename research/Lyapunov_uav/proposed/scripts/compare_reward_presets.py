from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List


def _add_import_paths() -> None:
    script_path = Path(__file__).resolve()
    proposed_dir = script_path.parents[1]
    lyapunov_dir = proposed_dir.parent
    for path in (lyapunov_dir, proposed_dir, script_path.parent):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_add_import_paths()

from analyze_reward_scale import build_scale_analysis, format_markdown as format_scale_markdown
from check_short_train_sanity import run_checks
from report_short_train import build_report, format_markdown as format_report_markdown
from short_hrl_train import run_short_train


DEFAULT_PRESETS = ("conservative_queue", "balanced", "quality_oriented")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _sanity_status(results: List[Dict[str, str]]) -> str:
    return "PASS" if all(result["status"] == "PASS" for result in results) else "FAIL"


def _metric_mean(report: Dict[str, Any], key: str) -> float | None:
    metric = report.get("metrics", {}).get(key, {})
    return metric.get("mean") if isinstance(metric, dict) else None


def _metric_final(report: Dict[str, Any], key: str) -> float | None:
    metric = report.get("metrics", {}).get(key, {})
    return metric.get("final") if isinstance(metric, dict) else None


def _dpp_mean(report: Dict[str, Any], key: str) -> float | None:
    term = report.get("dpp_terms", {}).get(key, {})
    return term.get("mean") if isinstance(term, dict) else None


def _checkpoint_exists(report: Dict[str, Any]) -> bool:
    return bool(report.get("checkpoint_path")) and int(report.get("checkpoint_count") or 0) > 0


def _make_train_args(
    *,
    preset: str,
    run_name: str,
    log_dir: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_steps=int(args.max_steps),
        max_updates=int(args.max_updates),
        rollout_steps=int(args.rollout_steps),
        seed=int(args.seed),
        device=str(args.device),
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir),
        run_name=run_name,
        hidden_dim=int(args.hidden_dim),
        reward_preset=preset,
        checkpoint_interval=int(args.checkpoint_interval),
        save_checkpoint=True,
        append_log=False,
    )


def _summarize_one(
    *,
    preset: str,
    run_name: str,
    log_path: Path,
    checkpoint_dir: Path,
    report: Dict[str, Any],
    scale_analysis: Dict[str, Any],
    sanity_results: List[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "preset_name": preset,
        "run_name": run_name,
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "max_updates": int(args.max_updates),
        "rollout_steps": int(args.rollout_steps),
        "device": report.get("device"),
        "log_path": str(log_path),
        "checkpoint_dir": str(checkpoint_dir),
        "sanity_status": _sanity_status(sanity_results),
        "final_total_reward": _metric_final(report, "total_reward"),
        "mean_total_reward": _metric_mean(report, "total_reward"),
        "mean_slow_reward": _metric_mean(report, "slow_reward"),
        "mean_fast_reward": _metric_mean(report, "fast_reward"),
        "mean_video_delivery_pressure": _dpp_mean(report, "video_delivery_pressure"),
        "mean_quality_reward": _dpp_mean(report, "quality_reward"),
        "mean_battery_service_pressure": _dpp_mean(report, "battery_service_pressure"),
        "mean_charging_effect": _dpp_mean(report, "charging_effect"),
        "mean_total_dpp_reward": _dpp_mean(report, "total_dpp_reward"),
        "mean_actor_loss": _metric_mean(report, "actor_loss"),
        "mean_critic_loss": _metric_mean(report, "critic_loss"),
        "mean_entropy": _metric_mean(report, "entropy"),
        "warning_count": int(scale_analysis.get("warning_count") or 0),
        "scale_status": scale_analysis.get("status"),
        "checkpoint_exists": _checkpoint_exists(report),
        "reward_coefficients": report.get("reward_coefficients"),
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_comparison_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Reward Preset Short Train Comparison",
        "",
        "This is a short smoke comparison only. It does not select final coefficients, run long training, or compare baselines.",
        "",
        "## Run Config",
        "",
        f"- presets: `{', '.join(summary['presets'])}`",
        f"- seed: `{summary['seed']}`",
        f"- max_steps: `{summary['max_steps']}`",
        f"- max_updates: `{summary['max_updates']}`",
        f"- rollout_steps: `{summary['rollout_steps']}`",
        f"- device: `{summary['device']}`",
        "",
        "## Summary",
        "",
        "| preset | sanity | scale | warnings | checkpoint | final_total_reward | mean_total_reward | mean_slow_reward | mean_fast_reward | mean_video_pressure | mean_quality | mean_battery_pressure | mean_charging | mean_total_dpp | mean_actor_loss | mean_critic_loss | mean_entropy |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["results"]:
        lines.append(
            "| "
            f"{row['preset_name']} | {row['sanity_status']} | {row['scale_status']} | "
            f"{row['warning_count']} | {row['checkpoint_exists']} | "
            f"{_format_value(row['final_total_reward'])} | {_format_value(row['mean_total_reward'])} | "
            f"{_format_value(row['mean_slow_reward'])} | {_format_value(row['mean_fast_reward'])} | "
            f"{_format_value(row['mean_video_delivery_pressure'])} | {_format_value(row['mean_quality_reward'])} | "
            f"{_format_value(row['mean_battery_service_pressure'])} | {_format_value(row['mean_charging_effect'])} | "
            f"{_format_value(row['mean_total_dpp_reward'])} | {_format_value(row['mean_actor_loss'])} | "
            f"{_format_value(row['mean_critic_loss'])} | {_format_value(row['mean_entropy'])} |"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- {failure['preset_name']}: {failure['error']}")
    return "\n".join(lines) + "\n"


def run_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    presets = tuple(args.presets)
    log_root = Path(args.log_dir).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for preset in presets:
        safe_preset = _safe_name(preset)
        run_name = f"{_safe_name(args.run_name)}_{safe_preset}"
        preset_log_dir = log_root / safe_preset
        preset_checkpoint_dir = checkpoint_root / safe_preset
        preset_output_dir = output_root / safe_preset
        preset_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            train_args = _make_train_args(
                preset=preset,
                run_name=run_name,
                log_dir=preset_log_dir,
                checkpoint_dir=preset_checkpoint_dir,
                args=args,
            )
            log_path = run_short_train(train_args)
            sanity_results = run_checks(
                log_path=log_path,
                checkpoint_dir=preset_checkpoint_dir,
                max_queue=float(args.max_queue),
                entropy_min=float(args.entropy_min),
                reward_tol=float(args.reward_tol),
            )
            report = build_report(
                log_path=log_path,
                checkpoint_dir=preset_checkpoint_dir,
                max_queue=float(args.max_queue),
                entropy_min=float(args.entropy_min),
                reward_tol=float(args.reward_tol),
            )
            scale_analysis = build_scale_analysis(
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

            (preset_output_dir / f"{run_name}_report.md").write_text(format_report_markdown(report), encoding="utf-8")
            (preset_output_dir / f"{run_name}_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (preset_output_dir / f"{run_name}_scale_analysis.md").write_text(
                format_scale_markdown(scale_analysis) + "\n",
                encoding="utf-8",
            )
            (preset_output_dir / f"{run_name}_scale_analysis.json").write_text(
                json.dumps(scale_analysis, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            results.append(
                _summarize_one(
                    preset=preset,
                    run_name=run_name,
                    log_path=log_path,
                    checkpoint_dir=preset_checkpoint_dir,
                    report=report,
                    scale_analysis=scale_analysis,
                    sanity_results=sanity_results,
                    args=args,
                )
            )
        except Exception as exc:  # noqa: BLE001 - comparison should report per-preset failures
            failures.append({"preset_name": preset, "error": repr(exc)})
            if args.stop_on_fail:
                break

    summary = {
        "event": "reward_preset_comparison",
        "presets": list(presets),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "max_updates": int(args.max_updates),
        "rollout_steps": int(args.rollout_steps),
        "device": str(args.device),
        "log_dir": str(log_root),
        "checkpoint_dir": str(checkpoint_root),
        "output_dir": str(output_root),
        "results": results,
        "failures": failures,
        "status": "PASS" if not failures and all(row["sanity_status"] == "PASS" for row in results) else "FAIL",
        "interpretation_note": (
            "Preset comparison is a short execution gate only; do not choose final coefficients "
            "or claim performance ranking from this output."
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short HRL train smoke comparison across reward presets.")
    parser.add_argument("--presets", nargs="+", default=list(DEFAULT_PRESETS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-updates", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--run-name", default="reward_preset_compare")
    parser.add_argument("--log-dir", default="Lyapunov_uav/proposed/outputs/preset_compare/logs")
    parser.add_argument("--checkpoint-dir", default="Lyapunov_uav/proposed/outputs/preset_compare/checkpoints")
    parser.add_argument("--output-dir", default="Lyapunov_uav/proposed/outputs/preset_compare/reports")
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
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive.")
    if args.max_updates <= 0:
        raise ValueError("max-updates must be positive.")
    if args.rollout_steps <= 1:
        raise ValueError("rollout-steps must be greater than 1.")

    summary = run_comparison(args)
    markdown = format_comparison_markdown(summary)
    print(markdown)

    output_json = Path(args.output_json).expanduser() if args.output_json else Path(args.output_dir).expanduser() / f"{_safe_name(args.run_name)}_summary.json"
    output_md = Path(args.output_md).expanduser() if args.output_md else Path(args.output_dir).expanduser() / f"{_safe_name(args.run_name)}_summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown, encoding="utf-8")

    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
