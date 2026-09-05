from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config_p3 import P3Config
try:
    import torch

    from agent.P3.ppo_agent import PPOAgent, PPOTransition, finish_trajectory
    from run.p3_train_ppo import (
        atomic_write_csv,
        atomic_write_json,
        build_episode_summary,
        parse_seed_list,
        plot_training_curve,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    PPOAgent = None
    PPOTransition = None
    finish_trajectory = None
    atomic_write_csv = None
    atomic_write_json = None
    build_episode_summary = None
    parse_seed_list = None
    plot_training_curve = None
    TORCH_AVAILABLE = False


TOTAL_KEYS = (
    "scaled_reward",
    "dpp_cost",
    "original_cost",
    "stall_user_slots",
    "served_user_slots",
    "delivered_chunks",
    "quality_utility",
    "degradation",
    "quality_switches",
    "quality_transitions",
    "queue_sum",
    "z_sum",
    "max_queue",
    "large_queue_violation_user_slots",
    "hired_uav_frames",
    "charging_slots",
    "charging_need_events",
    "stranded_before_charge_events",
    "return_to_charge_events",
    "precharge_depletion_events",
    "precharge_reserve_breach_events",
    "energy_consumed_j",
    "uav_distance_sum_m",
    "uav_scheduled_user_slots",
    "battery_reserve_violations",
    "power_violations",
    "provider_violations",
    "hire_probability_sum",
    "policy_decisions",
)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for PPO monitoring tests")
class P3TrainingMonitoringTests(unittest.TestCase):
    def test_seed_parser_and_extended_episode_metrics(self) -> None:
        self.assertEqual(parse_seed_list("92026, 92027,92028"), (92026, 92027, 92028))
        with self.assertRaises(Exception):
            parse_seed_list("92026,92026")

        cfg = P3Config(num_frames=2)
        totals = {key: 0.0 for key in TOTAL_KEYS}
        totals.update(
            dpp_cost=100.0,
            delivered_chunks=10.0,
            quality_utility=8.0,
            served_user_slots=10.0,
            hired_uav_frames=1.0,
            return_to_charge_events=2.0,
            precharge_reserve_breach_events=1.0,
        )
        summary = build_episode_summary(
            totals,
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            min_battery_soc=0.5,
            peak_uav_total_power_w=2.5,
            processed_frames=2,
            cfg=cfg,
        )
        self.assertAlmostEqual(summary["average_quality_utility"], 0.8)
        self.assertAlmostEqual(summary["precharge_reserve_breach_rate"], 0.5)
        self.assertEqual(summary["processed_frames"], 2)

    def test_ppo_update_exports_stability_diagnostics(self) -> None:
        cfg = P3Config(
            seed=7,
            ppo_hidden_dim=16,
            ppo_update_epochs=2,
            ppo_batch_size=4,
        )
        agent = PPOAgent(cfg, device="cpu")
        transitions: list[PPOTransition] = []
        for index in range(8):
            state = np.zeros(cfg.ppo_state_dim, dtype=np.float32)
            state[index % cfg.ppo_state_dim] = 1.0
            signatures = np.zeros((3, cfg.ppo_signature_dim), dtype=np.int64)
            signatures[0, 2] = 1
            signatures[1, 0] = 1
            signatures[1, 1] = 1
            signatures[1, 2] = 2
            signatures[2, 0] = 1
            signatures[2, 1] = 2
            signatures[2, 2] = 1
            choice = agent.select(state, signatures)
            transitions.append(
                PPOTransition(
                    state_features=state,
                    candidate_signatures=signatures,
                    action_index=choice.action_index,
                    old_log_prob=choice.log_prob,
                    old_value=choice.value,
                    reward=-0.1,
                    next_value=0.0,
                    done=index == 7,
                )
            )
        finish_trajectory(transitions, cfg)
        summary = agent.update(transitions)
        required = {
            "normalized_entropy",
            "clip_fraction",
            "grad_norm",
            "explained_variance",
            "update_epochs_completed",
            "stopped_by_kl",
        }
        self.assertTrue(required.issubset(summary))
        self.assertTrue(all(math.isfinite(float(value)) for value in summary.values()))

    def test_validation_points_and_atomic_outputs_are_rendered(self) -> None:
        rows = []
        for episode in range(12):
            row = {
                "episode": episode,
                "dpp_cost_per_user_slot": 1.0 / (episode + 1),
                "validation_dpp_per_user_slot": math.nan,
                "stall_ratio": 0.1,
                "delivered_chunks_per_user_slot": 0.5,
                "original_cost_per_user_slot": 0.2,
                "average_quality_utility": 0.9,
                "quality_p05_utility": 0.72,
                "hire_rate": 0.3,
                "mean_queue": 2.0,
                "mean_uav_user_distance_m": 80.0,
                "min_battery_soc": 0.4,
                "stranded_before_charge_rate": 0.0,
                "precharge_depletion_rate": 0.0,
                "precharge_reserve_breach_rate": 0.0,
                "normalized_entropy": 0.8,
                "approx_kl": 0.01,
                "target_kl": 0.015,
            }
            if episode in (4, 9):
                row["validation_dpp_per_user_slot"] = 0.3 - episode / 100.0
            rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_csv(root / "training_curve.csv", rows)
            atomic_write_json(root / "status.json", {"ok": True})
            plot_training_curve(rows, root / "training_curve.png")
            self.assertGreater((root / "training_curve.csv").stat().st_size, 0)
            self.assertGreater((root / "training_curve.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
