from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_MAX_STEPS = 1000
DEFAULT_MAX_UPDATES = 50
DEFAULT_ROLLOUT_STEPS = 32
DEFAULT_SEEDS = (0, 1, 2)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _selected_candidates(report: Dict[str, Any], explicit_presets: Sequence[str] | None) -> List[str]:
    candidates = [str(item) for item in report.get("recommended_long_run_candidates", [])]
    if explicit_presets:
        requested = set(explicit_presets)
        candidates = [preset for preset in candidates if preset in requested]
    if not candidates:
        raise ValueError("No long_run_candidate presets selected from candidate analysis report.")
    return candidates[:2]


def _default_seeds(report: Dict[str, Any]) -> List[int]:
    source_seeds = report.get("source_run_config", {}).get("seeds")
    if isinstance(source_seeds, list) and source_seeds:
        return [int(seed) for seed in source_seeds[:3]]
    return list(DEFAULT_SEEDS)


def build_manifest(
    *,
    candidate_report_path: Path,
    presets: Sequence[str] | None,
    seeds: Sequence[int] | None,
    run_name: str,
    log_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    max_steps: int,
    max_updates: int,
    rollout_steps: int,
    hidden_dim: int,
    device: str,
    checkpoint_interval: int,
) -> Dict[str, Any]:
    report = _load_json(candidate_report_path)
    selected_presets = _selected_candidates(report, presets)
    selected_seeds = [int(seed) for seed in (seeds if seeds else _default_seeds(report))]
    if not 1 <= len(selected_seeds) <= 3:
        raise ValueError("medium candidate runs should use 1 to 3 seeds.")

    jobs: List[Dict[str, Any]] = []
    for preset in selected_presets:
        preset_name = _safe_name(preset)
        for seed in selected_seeds:
            seed_name = f"seed_{seed:04d}" if seed >= 0 else f"seed_neg{abs(seed):04d}"
            job_run_name = f"{_safe_name(run_name)}_{preset_name}_{seed_name}"
            jobs.append(
                {
                    "job_index": len(jobs),
                    "preset_name": preset,
                    "seed": seed,
                    "run_name": job_run_name,
                    "log_dir": str(log_dir.expanduser() / preset_name / seed_name),
                    "checkpoint_dir": str(checkpoint_dir.expanduser() / preset_name / seed_name),
                    "output_dir": str(output_dir.expanduser() / preset_name / seed_name),
                }
            )

    return {
        "event": "medium_candidate_run_manifest",
        "candidate_report_path": str(candidate_report_path.expanduser()),
        "selected_presets": selected_presets,
        "seeds": selected_seeds,
        "job_count": len(jobs),
        "medium_config": {
            "max_steps": int(max_steps),
            "max_updates": int(max_updates),
            "rollout_steps": int(rollout_steps),
            "hidden_dim": int(hidden_dim),
            "device": str(device),
            "checkpoint_interval": int(checkpoint_interval),
        },
        "jobs": jobs,
        "interpretation_note": (
            "Medium candidate runs are for training stability and log-trend inspection only. "
            "They are not final long training and do not compare baselines."
        ),
    }


def format_markdown(manifest: Dict[str, Any]) -> str:
    config = manifest["medium_config"]
    lines = [
        "# Medium Candidate Run Manifest",
        "",
        "This manifest prepares medium-length stability runs only.",
        "",
        "## Config",
        "",
        f"- selected_presets: `{', '.join(manifest['selected_presets'])}`",
        f"- seeds: `{', '.join(str(seed) for seed in manifest['seeds'])}`",
        f"- job_count: `{manifest['job_count']}`",
        f"- max_steps: `{config['max_steps']}`",
        f"- max_updates: `{config['max_updates']}`",
        f"- rollout_steps: `{config['rollout_steps']}`",
        f"- hidden_dim: `{config['hidden_dim']}`",
        f"- device: `{config['device']}`",
        f"- checkpoint_interval: `{config['checkpoint_interval']}`",
        "",
        "## Jobs",
        "",
        "| index | preset | seed | run_name |",
        "| ---: | --- | ---: | --- |",
    ]
    for job in manifest["jobs"]:
        lines.append(f"| {job['job_index']} | {job['preset_name']} | {job['seed']} | {job['run_name']} |")
    lines.extend(["", "## Note", "", manifest["interpretation_note"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Seraph medium-run manifest for selected reward presets.")
    parser.add_argument("--candidate-report-json", required=True)
    parser.add_argument("--presets", nargs="+", default=None, help="Optional subset of recommended candidates.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--run-name", default="reward_preset_medium")
    parser.add_argument("--log-dir", default="Lyapunov_uav/proposed/outputs/preset_medium/logs")
    parser.add_argument("--checkpoint-dir", default="Lyapunov_uav/proposed/outputs/preset_medium/checkpoints")
    parser.add_argument("--output-dir", default="Lyapunov_uav/proposed/outputs/preset_medium/reports")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive.")
    if args.max_updates <= 0:
        raise ValueError("max-updates must be positive.")
    if args.rollout_steps <= 1:
        raise ValueError("rollout-steps must be greater than 1.")
    if args.checkpoint_interval <= 0:
        raise ValueError("checkpoint-interval must be positive.")

    candidate_report_path = Path(args.candidate_report_json).expanduser()
    manifest = build_manifest(
        candidate_report_path=candidate_report_path,
        presets=args.presets,
        seeds=args.seeds,
        run_name=str(args.run_name),
        log_dir=Path(args.log_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        max_steps=int(args.max_steps),
        max_updates=int(args.max_updates),
        rollout_steps=int(args.rollout_steps),
        hidden_dim=int(args.hidden_dim),
        device=str(args.device),
        checkpoint_interval=int(args.checkpoint_interval),
    )
    markdown = format_markdown(manifest)
    print(markdown)

    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json
        else Path(args.output_dir).expanduser() / f"{_safe_name(args.run_name)}_manifest.json"
    )
    output_md = (
        Path(args.output_md).expanduser()
        if args.output_md
        else Path(args.output_dir).expanduser() / f"{_safe_name(args.run_name)}_manifest.md"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
