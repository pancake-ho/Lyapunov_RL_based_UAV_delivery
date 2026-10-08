from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EPISODE_REQUIRED = {
    "episode",
    "global_round",
    "episode_formulation_reward",
    "episode_realized_round_cost",
    "episode_hiring_cost",
    "episode_delivery",
    "episode_stall",
    "episode_scheduled_stall",
    "episode_unscheduled_stall",
    "episode_scheduled_stall_rate",
    "episode_quality_per_chunk",
    "episode_quality_degradation_per_chunk",
    "episode_outage_slots",
    "episode_min_soc",
    "episode_service_rate",
    "episode_requested_chunks",
    "episode_active_action_dims",
}
UPDATE_REQUIRED = {
    "update",
    "global_round",
    "approx_kl_post",
    "clipfrac_post",
    "explained_variance",
    "value_rmse",
    "value_mean",
    "return_mean",
    "categorical_entropy",
    "actor_grad_norm",
    "critic_grad_norm",
    "early_stopped",
    "completed_minibatches",
    "expected_minibatches",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and diagnose a completed Fast-PPO H1 training run. "
            "This script does not select a checkpoint."
        )
    )
    parser.add_argument("--episodes-csv", type=Path, required=True)
    parser.add_argument("--updates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-window", type=int, default=50)
    parser.add_argument("--critic-tail-updates", type=int, default=250)
    parser.add_argument("--minimum-allowed-soc", type=float, default=19.95)
    return parser.parse_args()


def _read_numeric_csv(path: Path) -> List[Dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, float]] = []
        for row_number, row in enumerate(reader, start=2):
            converted: Dict[str, float] = {}
            for key, value in row.items():
                if key is None or value is None:
                    raise ValueError(
                        f"Malformed CSV field at {path}:{row_number}."
                    )
                text = value.strip()
                if text == "":
                    converted[str(key)] = float("nan")
                    continue
                try:
                    converted[str(key)] = float(text)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric CSV value at {path}:{row_number}, "
                        f"column={key!r}: {value!r}"
                    ) from exc
            rows.append(converted)
    if not rows:
        raise RuntimeError(f"CSV has no data rows: {path}")
    return rows


def _require_columns(
    rows: Sequence[Mapping[str, float]],
    required: Iterable[str],
    name: str,
) -> None:
    missing = set(required) - set(rows[0])
    if missing:
        raise RuntimeError(f"{name} is missing columns: {sorted(missing)}")


def _values(
    rows: Sequence[Mapping[str, float]],
    key: str,
) -> List[float]:
    return [float(row[key]) for row in rows]


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else float("nan")


def _std(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.pstdev(finite) if len(finite) > 1 else 0.0


def _quantile(values: Iterable[float], probability: float) -> float:
    finite = sorted(
        float(value) for value in values if math.isfinite(float(value))
    )
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def _correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return float("nan")
    x_mean = statistics.fmean(x for x, _ in pairs)
    y_mean = statistics.fmean(y for _, y in pairs)
    covariance = sum(
        (x - x_mean) * (y - y_mean) for x, y in pairs
    )
    x_ss = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_ss = sum((y - y_mean) ** 2 for _, y in pairs)
    denominator = math.sqrt(x_ss * y_ss)
    return covariance / denominator if denominator > 0.0 else float("nan")


def _ols_slope(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> Dict[str, float]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return {"slope": float("nan"), "slope_se": float("nan"), "t": float("nan")}
    x_mean = statistics.fmean(x for x, _ in pairs)
    y_mean = statistics.fmean(y for _, y in pairs)
    x_ss = sum((x - x_mean) ** 2 for x, _ in pairs)
    if x_ss <= 0.0:
        return {"slope": float("nan"), "slope_se": float("nan"), "t": float("nan")}
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in pairs
    ) / x_ss
    intercept = y_mean - slope * x_mean
    residual_ss = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in pairs
    )
    residual_variance = residual_ss / float(len(pairs) - 2)
    slope_se = math.sqrt(residual_variance / x_ss)
    return {
        "slope": slope,
        "slope_se": slope_se,
        "t": slope / slope_se if slope_se > 0.0 else float("nan"),
    }


def _window_change(
    values: Sequence[float],
    window: int,
) -> Dict[str, float]:
    if len(values) < 2 * int(window):
        raise ValueError(
            f"Need at least {2 * int(window)} rows for window comparison."
        )
    first = _mean(values[:window])
    last = _mean(values[-window:])
    percentage = (
        100.0 * (last - first) / abs(first)
        if math.isfinite(first) and first != 0.0
        else float("nan")
    )
    return {
        "first_mean": first,
        "last_mean": last,
        "relative_change_percent": percentage,
        "last_std": _std(values[-window:]),
    }


def _check_contiguous(rows: Sequence[Mapping[str, float]], key: str) -> bool:
    actual = [int(float(row[key])) for row in rows]
    return actual == list(range(1, len(rows) + 1))


def analyze(
    episodes: Sequence[Mapping[str, float]],
    updates: Sequence[Mapping[str, float]],
    *,
    window: int,
    critic_tail: int,
    minimum_allowed_soc: float,
) -> Dict[str, object]:
    _require_columns(episodes, EPISODE_REQUIRED, "episodes CSV")
    _require_columns(updates, UPDATE_REQUIRED, "updates CSV")
    if window <= 0 or critic_tail <= 0:
        raise ValueError("Window sizes must be positive.")

    reward = _values(episodes, "episode_formulation_reward")
    fast_cost = [-value for value in reward]
    active_dims = _values(episodes, "episode_active_action_dims")
    delivery = _values(episodes, "episode_delivery")
    stall = _values(episodes, "episode_stall")
    scheduled_stall = _values(episodes, "episode_scheduled_stall")
    unscheduled_stall = _values(episodes, "episode_unscheduled_stall")
    cost_per_active_dimension = [
        cost / active if active > 0.0 else float("nan")
        for cost, active in zip(fast_cost, active_dims)
    ]
    identity_errors = [
        abs(
            float(row["episode_realized_round_cost"])
            - (
                -float(row["episode_formulation_reward"])
                + float(row["episode_hiring_cost"])
            )
        )
        for row in episodes
    ]

    expected_minibatches = _values(updates, "expected_minibatches")
    completed_minibatches = _values(updates, "completed_minibatches")
    minibatch_mismatches = sum(
        int(completed != expected)
        for completed, expected in zip(
            completed_minibatches,
            expected_minibatches,
        )
    )
    post_kl = _values(updates, "approx_kl_post")
    post_clip = _values(updates, "clipfrac_post")
    actor_grad = _values(updates, "actor_grad_norm")
    critic_grad = _values(updates, "critic_grad_norm")
    explained_variance = _values(updates, "explained_variance")
    tail_count = min(int(critic_tail), len(updates))

    correlation_rows: Dict[str, Dict[str, float]] = {}
    for label, series in (
        ("active_action_dims", active_dims),
        ("delivery", delivery),
        ("total_stall", stall),
        ("scheduled_stall", scheduled_stall),
        ("unscheduled_stall", unscheduled_stall),
    ):
        correlation = _correlation(series, fast_cost)
        correlation_rows[label] = {
            "pearson_r_with_fast_cost": correlation,
            "r_squared": correlation * correlation,
        }

    episode_number = _values(episodes, "episode")
    integrity = {
        "episode_rows": len(episodes),
        "update_rows": len(updates),
        "episodes_contiguous": _check_contiguous(episodes, "episode"),
        "updates_contiguous": _check_contiguous(updates, "update"),
        "last_episode_global_round": int(episodes[-1]["global_round"]),
        "last_update_global_round": int(updates[-1]["global_round"]),
        "reward_cost_identity_max_abs_error": max(identity_errors),
        "minibatch_mismatches": minibatch_mismatches,
        "early_stop_count": sum(
            int(value != 0.0)
            for value in _values(updates, "early_stopped")
        ),
        "outage_slots_total": sum(
            _values(episodes, "episode_outage_slots")
        ),
        "minimum_soc": min(_values(episodes, "episode_min_soc")),
        "minimum_allowed_soc": float(minimum_allowed_soc),
    }
    integrity["safe_completion"] = bool(
        integrity["episodes_contiguous"]
        and integrity["updates_contiguous"]
        and integrity["minibatch_mismatches"] == 0
        and integrity["early_stop_count"] == 0
        and integrity["outage_slots_total"] == 0.0
        and integrity["minimum_soc"] >= float(minimum_allowed_soc)
    )

    comparisons = {
        "fast_cost": _window_change(fast_cost, window),
        "delivery": _window_change(delivery, window),
        "total_stall": _window_change(stall, window),
        "scheduled_stall": _window_change(scheduled_stall, window),
        "scheduled_stall_rate": _window_change(
            _values(episodes, "episode_scheduled_stall_rate"),
            window,
        ),
        "quality_per_chunk": _window_change(
            _values(episodes, "episode_quality_per_chunk"),
            window,
        ),
        "quality_degradation_per_chunk": _window_change(
            _values(
                episodes,
                "episode_quality_degradation_per_chunk",
            ),
            window,
        ),
        "service_rate": _window_change(
            _values(episodes, "episode_service_rate"),
            window,
        ),
        "requested_chunks": _window_change(
            _values(episodes, "episode_requested_chunks"),
            window,
        ),
        "active_action_dims": _window_change(active_dims, window),
        "fast_cost_per_active_action_dim_proxy": _window_change(
            cost_per_active_dimension,
            window,
        ),
    }

    trust_region = {
        "post_kl_mean": _mean(post_kl),
        "post_kl_median": _quantile(post_kl, 0.5),
        "post_kl_p90": _quantile(post_kl, 0.9),
        "post_kl_p95": _quantile(post_kl, 0.95),
        "post_kl_max": max(post_kl),
        "post_kl_over_0_03_count": sum(value > 0.03 for value in post_kl),
        "post_kl_over_0_05_count": sum(value > 0.05 for value in post_kl),
        "post_clip_mean": _mean(post_clip),
        "post_clip_median": _quantile(post_clip, 0.5),
        "post_clip_max": max(post_clip),
        "post_clip_over_0_25_count": sum(value > 0.25 for value in post_clip),
        "post_clip_over_0_50_count": sum(value > 0.50 for value in post_clip),
        "actor_preclip_grad_over_0_5_count": sum(
            value > 0.5 for value in actor_grad
        ),
        "critic_preclip_grad_over_0_5_count": sum(
            value > 0.5 for value in critic_grad
        ),
        "update_count": len(updates),
    }

    critic = {
        "explained_variance_mean": _mean(explained_variance),
        "explained_variance_negative_count": sum(
            value < 0.0 for value in explained_variance
        ),
        "explained_variance_below_minus_1_count": sum(
            value < -1.0 for value in explained_variance
        ),
        "tail_update_count": tail_count,
        "tail_explained_variance_mean": _mean(
            explained_variance[-tail_count:]
        ),
        "tail_value_rmse_mean": _mean(
            _values(updates, "value_rmse")[-tail_count:]
        ),
        "tail_value_mean": _mean(
            _values(updates, "value_mean")[-tail_count:]
        ),
        "tail_return_mean": _mean(
            _values(updates, "return_mean")[-tail_count:]
        ),
    }

    trends = {
        "fast_cost_vs_episode": _ols_slope(episode_number, fast_cost),
        "fast_cost_per_active_action_dim_proxy_vs_episode": _ols_slope(
            episode_number,
            cost_per_active_dimension,
        ),
        "scheduled_stall_rate_vs_episode": _ols_slope(
            episode_number,
            _values(episodes, "episode_scheduled_stall_rate"),
        ),
        "service_rate_vs_episode": _ols_slope(
            episode_number,
            _values(episodes, "episode_service_rate"),
        ),
        "requested_chunks_vs_episode": _ols_slope(
            episode_number,
            _values(episodes, "episode_requested_chunks"),
        ),
    }

    conclusions = {
        "training_run_is_structurally_valid": integrity["safe_completion"],
        "raw_episode_reward_is_workload_confounded": bool(
            abs(
                correlation_rows["active_action_dims"][
                    "pearson_r_with_fast_cost"
                ]
            )
            >= 0.8
        ),
        "ppo_updates_exceed_screening_levels": bool(
            trust_region["post_clip_over_0_25_count"]
            / max(len(updates), 1)
            > 0.5
            or trust_region["post_kl_over_0_03_count"]
            / max(len(updates), 1)
            > 0.1
        ),
        "checkpoint_sweep_required": True,
        "slow_dpp_ready": False,
        "conditional_h2_actor_lr": 7.5e-6,
    }
    return {
        "integrity": integrity,
        "first_last_window": comparisons,
        "workload_correlations": correlation_rows,
        "episode_trends": trends,
        "trust_region": trust_region,
        "critic": critic,
        "conclusions": conclusions,
    }


def _format(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def _write_report(path: Path, result: Mapping[str, object]) -> None:
    integrity = result["integrity"]
    comparisons = result["first_last_window"]
    correlations = result["workload_correlations"]
    trust = result["trust_region"]
    critic = result["critic"]
    conclusions = result["conclusions"]
    assert isinstance(integrity, Mapping)
    assert isinstance(comparisons, Mapping)
    assert isinstance(correlations, Mapping)
    assert isinstance(trust, Mapping)
    assert isinstance(critic, Mapping)
    assert isinstance(conclusions, Mapping)

    lines = [
        "# Fast-PPO H1 training diagnostics",
        "",
        "## Verdict",
        "",
        (
            "The run completed safely, but training reward alone cannot "
            "identify the best checkpoint."
            if conclusions["training_run_is_structurally_valid"]
            else "The run failed one or more structural safety checks."
        ),
        "",
        "- Slow DPP ready: **NO** until the deterministic paired checkpoint "
        "sweep passes.",
        "- Raw reward workload-confounded: "
        f"**{_format(conclusions['raw_episode_reward_is_workload_confounded'])}**",
        "- PPO update screening exceeded: "
        f"**{_format(conclusions['ppo_updates_exceed_screening_levels'])}**",
        "",
        "## Integrity",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    for key in (
        "episode_rows",
        "update_rows",
        "episodes_contiguous",
        "updates_contiguous",
        "reward_cost_identity_max_abs_error",
        "minibatch_mismatches",
        "early_stop_count",
        "outage_slots_total",
        "minimum_soc",
        "safe_completion",
    ):
        lines.append(f"| {key} | {_format(integrity[key])} |")

    lines.extend(
        [
            "",
            "## First window versus last window",
            "",
            "| metric | first mean | last mean | change | last std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, raw_change in comparisons.items():
        assert isinstance(raw_change, Mapping)
        lines.append(
            f"| {metric} | {_format(raw_change['first_mean'])} | "
            f"{_format(raw_change['last_mean'])} | "
            f"{_format(raw_change['relative_change_percent'])}% | "
            f"{_format(raw_change['last_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Workload confounding",
            "",
            "| variable | Pearson r with Fast cost | R² |",
            "|---|---:|---:|",
        ]
    )
    for variable, raw_correlation in correlations.items():
        assert isinstance(raw_correlation, Mapping)
        lines.append(
            f"| {variable} | "
            f"{_format(raw_correlation['pearson_r_with_fast_cost'])} | "
            f"{_format(raw_correlation['r_squared'])} |"
        )

    lines.extend(
        [
            "",
            "## PPO and critic diagnostics",
            "",
            f"- Post-update KL: mean {_format(trust['post_kl_mean'])}, "
            f"p95 {_format(trust['post_kl_p95'])}, max "
            f"{_format(trust['post_kl_max'])}.",
            f"- Post-update clip fraction: mean "
            f"{_format(trust['post_clip_mean'])}; "
            f"{_format(trust['post_clip_over_0_25_count'])}/"
            f"{_format(trust['update_count'])} updates exceeded 0.25.",
            f"- Explained variance: overall mean "
            f"{_format(critic['explained_variance_mean'])}; tail mean "
            f"{_format(critic['tail_explained_variance_mean'])}.",
            f"- Tail value RMSE: {_format(critic['tail_value_rmse_mean'])}.",
            "",
            "## Required next action",
            "",
            "Run Initial/Ep25/.../Ep200 with common seeds using "
            "`fast_checkpoint_sweep.py`. Connect a frozen Fast checkpoint to "
            "Slow DPP only when `selection.json` reports "
            "`slow_dpp_gate_passed: true`. If no checkpoint passes, run H2 "
            "from scratch with actor LR 7.5e-6 while retaining critic LR "
            "3e-5, then repeat the same sweep.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_optional_plots(
    output_dir: Path,
    episodes: Sequence[Mapping[str, float]],
    updates: Sequence[Mapping[str, float]],
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is unavailable; plots were skipped"

    episode = _values(episodes, "episode")
    cost = [
        -value
        for value in _values(episodes, "episode_formulation_reward")
    ]
    active = _values(episodes, "episode_active_action_dims")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(episode, cost, linewidth=0.8)
    axes[0].set_title("Raw Fast cost")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cost")
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(active, cost, s=12, alpha=0.6)
    axes[1].set_title("Cost versus workload proxy")
    axes[1].set_xlabel("Mean active action dimensions")
    axes[1].set_ylabel("Cost")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "h1_workload_confounding.png", dpi=200)
    plt.close(fig)

    update = _values(updates, "update")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(update, _values(updates, "approx_kl_post"), linewidth=0.7)
    axes[0].axhline(0.03, color="tab:red", linestyle="--")
    axes[0].set_ylabel("Post KL")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(update, _values(updates, "clipfrac_post"), linewidth=0.7)
    axes[1].axhline(0.25, color="tab:orange", linestyle="--")
    axes[1].axhline(0.50, color="tab:red", linestyle="--")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Post clip fraction")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "h1_ppo_trust_region.png", dpi=200)
    plt.close(fig)
    return None


def main() -> None:
    args = _parse_args()
    episodes = _read_numeric_csv(args.episodes_csv.expanduser().resolve())
    updates = _read_numeric_csv(args.updates_csv.expanduser().resolve())
    result = analyze(
        episodes,
        updates,
        window=int(args.comparison_window),
        critic_tail=int(args.critic_tail_updates),
        minimum_allowed_soc=float(args.minimum_allowed_soc),
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "h1_diagnostics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(output_dir / "H1_DIAGNOSTICS.md", result)
    plot_note = _write_optional_plots(output_dir, episodes, updates)
    if plot_note:
        (output_dir / "PLOT_NOTE.txt").write_text(
            plot_note + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result["conclusions"], indent=2), flush=True)


if __name__ == "__main__":
    main()
