from __future__ import annotations

import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


PROPOSED_ROOT = Path(__file__).resolve().parents[3]
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))

from config import EnvConfig  # noqa: E402
from env.env import Env  # noqa: E402
from agent.PPO.common import (  # noqa: E402
    ScalarLogger,
    infer_fast_obs_dim,
    set_seed,
    split_env_reset,
    split_env_step,
)
from agent.PPO.common.utils import save_json  # noqa: E402
from agent.PPO.fast.fast_agent import (  # noqa: E402
    FastPPOAgent,
    FastPPOConfig,
)
from agent.PPO.slow.slow_config import (  # noqa: E402
    SlowDPPConfig,
    get_slow_dpp_config,
)
from agent.PPO.slow.slow_dpp_controller import (  # noqa: E402
    SlowDPPController,
)
from agent.PPO.slow.slow_metrics import HRLMetrics  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = PROPOSED_ROOT / result
    return result.resolve()


def _make_env_config(
    run_cfg: SlowDPPConfig,
    seed: int,
) -> EnvConfig:
    base = EnvConfig()
    return replace(
        base,
        seed=int(seed),
        move_prob=float(run_cfg.move_prob),
        episode_slots=(
            int(base.slow_T) * int(run_cfg.rounds_per_episode)
        ),
    )


def _load_trusted_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    """
    Read metadata from a checkpoint produced by this repository.

    weights_only=False is intentional only for the user's own trusted
    checkpoints because extra may contain NumPy arrays.
    """
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # PyTorch before weights_only
        checkpoint = torch.load(path, map_location="cpu")
    return dict(checkpoint.get("extra", {}))


def _build_frozen_fast_agent(
    run_cfg: SlowDPPConfig,
    env_cfg: EnvConfig,
    sample_fast_obs: Dict[str, Any],
) -> FastPPOAgent:
    checkpoint_path = _resolve_path(run_cfg.fast_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"fast checkpoint does not exist: {checkpoint_path}"
        )

    extra = _load_trusted_checkpoint_metadata(checkpoint_path)
    saved_config = dict(extra.get("fast_ppo_config", {}))
    allowed = {field.name for field in fields(FastPPOConfig)}
    config_values = {
        key: value
        for key, value in saved_config.items()
        if key in allowed
    }
    config_values["device"] = str(run_cfg.device)
    config_values.setdefault(
        "hidden_dims",
        tuple(int(x) for x in run_cfg.fast_hidden_dims_fallback),
    )
    config_values.setdefault(
        "init_log_std",
        float(run_cfg.fast_init_log_std_fallback),
    )
    fast_ppo_config = FastPPOConfig(**config_values)

    obs_dim = infer_fast_obs_dim(sample_fast_obs)
    saved_obs_dim = int(extra.get("obs_dim", obs_dim))
    if saved_obs_dim != obs_dim:
        raise ValueError(
            "fast checkpoint/environment observation mismatch: "
            f"checkpoint={saved_obs_dim}, environment={obs_dim}"
        )

    agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=obs_dim,
        ppo_cfg=fast_ppo_config,
    )
    checkpoint = agent.load(
        checkpoint_path,
        strict=True,
        load_optimizer=False,
    )
    loaded_extra = checkpoint.get("extra", {})
    if (
        agent.obs_normalizer is not None
        and "obs_normalizer" not in loaded_extra
    ):
        raise RuntimeError(
            "fast checkpoint has no observation-normalizer state. "
            "A normalized fast policy cannot be evaluated safely without it."
        )

    agent.model.eval()
    for parameter in agent.model.parameters():
        parameter.requires_grad_(False)

    return agent


def _run_fast_round(
    env: Env,
    fast_agent: FastPPOAgent,
) -> Tuple[HRLMetrics, Dict[str, Any], bool, bool]:
    fast_obs = env.get_fast_obs()
    metrics = HRLMetrics(
        fast_layer_ratios=np.zeros(int(env.cfg.layer), dtype=np.float64)
    )
    boundary_info: Optional[Dict[str, Any]] = None
    terminated = False
    truncated = False

    for _ in range(int(env.slow_T)):
        fast_selected = fast_agent.select_action(
            fast_obs,
            deterministic=True,
            update_norm=False,
        )
        next_obs, _, terminated, truncated, info = split_env_step(
            env.step(fast_selected["env_action"])
        )
        metrics.add_slot(
            info,
            fast_selected,
            num_layers=int(env.cfg.layer),
        )
        fast_obs = next_obs

        if bool(info.get("is_round_boundary", False)):
            boundary_info = info
            break
        if terminated or truncated:
            raise RuntimeError(
                "Episode ended before a complete slow round. "
                "episode_slots must be a multiple of slow_T."
            )

    if boundary_info is None:
        raise RuntimeError("Fast loop did not reach a slow round boundary.")
    if metrics.slots != int(env.slow_T):
        raise RuntimeError(
            f"Incomplete round: expected={env.slow_T}, got={metrics.slots}."
        )

    return metrics, boundary_info, bool(terminated), bool(truncated)


def _extract_slow_reward(
    boundary_info: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    reward_components = boundary_info.get("reward_components", {})
    slow_reward = float(reward_components.get("slow_reward", 0.0))
    slow_components = dict(
        reward_components.get("slow_reward_components", {})
    )

    if not bool(slow_components.get("is_round_boundary", False)):
        raise RuntimeError("Boundary info has no completed round DPP value.")

    component_reward = float(
        slow_components.get("slow_reward", slow_reward)
    )
    if not np.isclose(
        slow_reward,
        component_reward,
        rtol=1e-6,
        atol=1e-3,
    ):
        raise RuntimeError(
            "Slow reward fields disagree at the round boundary."
        )

    return float(slow_reward), slow_components


def run(run_cfg: SlowDPPConfig) -> None:
    set_seed(
        int(run_cfg.seed),
        deterministic=bool(run_cfg.deterministic_torch),
    )

    env_cfg = _make_env_config(run_cfg, seed=int(run_cfg.seed))
    probe_env = Env(env_cfg)
    sample_fast_obs, _ = split_env_reset(probe_env.reset())
    fast_agent = _build_frozen_fast_agent(
        run_cfg=run_cfg,
        env_cfg=env_cfg,
        sample_fast_obs=sample_fast_obs,
    )

    controller = SlowDPPController(
        env_cfg=env_cfg,
        dpp_cfg=run_cfg,
    )
    env = Env(env_cfg)

    run_dir = _resolve_path(Path(run_cfg.output_root) / run_cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_cfg.to_dict(), run_dir / "slow_dpp_config.json")
    save_json(env_cfg.as_dict(), run_dir / "env_config.json")

    episode_logger = ScalarLogger(run_dir / "episode_metrics.csv")
    round_logger = ScalarLogger(run_dir / "round_metrics.csv")

    aggregate = HRLMetrics(
        fast_layer_ratios=np.zeros(int(env_cfg.layer), dtype=np.float64)
    )
    global_round = 0
    global_slot = 0

    print("=" * 96, flush=True)
    print("[SLOW ROUND-DPP RUN START]", flush=True)
    print(f"run_dir           : {run_dir}", flush=True)
    print(
        f"fast_checkpoint   : {_resolve_path(run_cfg.fast_checkpoint)}",
        flush=True,
    )
    print("fast_policy       : frozen deterministic PPO", flush=True)
    print("slow_policy       : none", flush=True)
    print("slow_controller   : round DPP coordinate minimization", flush=True)
    print(f"slow_T            : {env_cfg.slow_T}", flush=True)
    print(
        f"forecast_scenarios: {run_cfg.forecast_scenarios}",
        flush=True,
    )
    print(
        f"max_coord_sweeps  : {run_cfg.max_coordinate_sweeps}",
        flush=True,
    )
    print(f"move_prob         : {env_cfg.move_prob}", flush=True)
    print("=" * 96, flush=True)

    for episode in range(1, int(run_cfg.num_episodes) + 1):
        split_env_reset(env.reset())
        episode_metrics = HRLMetrics(
            fast_layer_ratios=np.zeros(
                int(env_cfg.layer),
                dtype=np.float64,
            )
        )

        for round_in_episode in range(
            1,
            int(run_cfg.rounds_per_episode) + 1,
        ):
            selected = controller.select_action(
                env=env,
                fast_agent=fast_agent,
            )
            env.apply_slow_action(selected["env_action"])

            round_metrics, boundary_info, terminated, truncated = (
                _run_fast_round(
                    env=env,
                    fast_agent=fast_agent,
                )
            )
            slow_reward, slow_components = _extract_slow_reward(
                boundary_info
            )
            realized_round_dpp_cost = -float(slow_reward)
            predicted_round_dpp_cost = float(
                selected["predicted_round_dpp_cost"]
            )

            round_metrics.add_round(
                slow_reward=slow_reward,
                slow_components=slow_components,
                slow_selected=selected,
            )
            episode_metrics.merge(round_metrics)
            aggregate.merge(round_metrics)

            global_round += 1
            global_slot += int(round_metrics.slots)

            round_logger.write(
                {
                    "episode": int(episode),
                    "round_in_episode": int(round_in_episode),
                    "global_round": int(global_round),
                    "global_slot": int(global_slot),
                    "predicted_round_dpp_cost": (
                        predicted_round_dpp_cost
                    ),
                    "realized_round_dpp_cost": (
                        realized_round_dpp_cost
                    ),
                    "round_dpp_prediction_error": (
                        realized_round_dpp_cost
                        - predicted_round_dpp_cost
                    ),
                    **selected["action_info"],
                    **round_metrics.summary(),
                }
            )

            print(
                "[SLOW DPP ROUND] "
                f"ep={episode}/{run_cfg.num_episodes} "
                f"round={round_in_episode}/{run_cfg.rounds_per_episode} "
                f"pred_cost={predicted_round_dpp_cost:.3f} "
                f"real_cost={realized_round_dpp_cost:.3f} "
                f"hired={selected['action_info']['effective_hired_uav']} "
                f"rsu_links={selected['action_info']['effective_rsu_links']} "
                f"uav_links={selected['action_info']['effective_uav_links']} "
                f"candidates={selected['action_info']['unique_candidates']}",
                flush=True,
            )

            if terminated or truncated:
                break

        episode_summary = episode_metrics.summary()
        episode_logger.write(
            {
                "episode": int(episode),
                "global_round": int(global_round),
                "global_slot": int(global_slot),
                **episode_summary,
            }
        )

        if episode % int(run_cfg.log_every_episodes) == 0:
            print(
                "[SLOW DPP EPISODE] "
                f"ep={episode}/{run_cfg.num_episodes} "
                f"dpp_cost/round="
                f"{-episode_summary['slow_reward_per_round']:.3f} "
                f"delivery/slot={episode_summary['delivery_per_slot']:.4f} "
                f"degradation/chunk="
                f"{episode_summary['quality_degradation_per_chunk']:.4f} "
                f"scheduled_stall="
                f"{episode_summary['scheduled_stall_rate']:.6f} "
                f"hired/round="
                f"{episode_summary['hired_uav_per_round']:.3f}",
                flush=True,
            )

    final_summary = aggregate.summary()
    save_json(final_summary, run_dir / "final_summary.json")

    print("[SLOW ROUND-DPP RUN DONE]", flush=True)
    print(final_summary, flush=True)


def main() -> None:
    run(get_slow_dpp_config())


if __name__ == "__main__":
    main()