from __future__ import annotations

import itertools
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    import torch

    from agent.P3.ppo_agent import PPOAgent
    from run.p3_train_ppo import train_episode

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    PPOAgent = None
    train_episode = None
    TORCH_AVAILABLE = False

from agent.P3.exact_fast_controller import ExactFastController
from agent.P3.features import build_candidate_feature_matrix, build_state_features
from agent.P3.slow_rollout_controller import SlowRolloutController
from config_p3 import P3Config
from env.p3.battery import (
    battery_power_cap_w,
    diagnose_return_to_charge,
    safe_return_reserve_j,
)
from env.p3.environment import (
    apply_region_result,
    generate_frame_trace,
    simulate_region_frame,
)
from env.p3.topology import (
    enumerate_region_actions,
    initialize_state,
    region_membership,
    validate_region_action,
)
from env.p3.types import RegionAction
from run.p3_common import run_policy


class P3ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = P3Config(num_frames=2, rollout_scenarios=1)
        self.state = initialize_state(self.cfg)
        self.fast = ExactFastController(self.cfg)

    def users(self, region: int) -> list[int]:
        membership = region_membership(self.state, self.cfg)
        return np.flatnonzero(membership == region).tolist()

    def test_p3_capacity_contract(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.cfg, uav_capacity=1)
        with self.assertRaises(ValueError):
            replace(self.cfg, uav_capacity=self.cfg.rsu_capacity)
        with self.assertRaises(ValueError):
            replace(self.cfg, candidate_offsets_m=(-100.0, 0.0, 100.0))

    def test_enumeration_enforces_capacity_and_single_provider(self) -> None:
        users = self.users(0)
        actions = enumerate_region_actions(self.state, 0, users, self.cfg)
        self.assertTrue(any(len(action.uav_users) == 2 for action in actions))
        for action in actions:
            validate_region_action(action, users, self.cfg)
            self.assertLessEqual(len(action.rsu_users), self.cfg.rsu_capacity)
            self.assertLessEqual(len(action.uav_users), self.cfg.uav_capacity)
            self.assertFalse(set(action.rsu_users).intersection(action.uav_users))

    def test_multi_user_uav_solver_matches_full_option_product(self) -> None:
        users = (0, 1)
        z = {0: 97.0, 1: 95.0}
        distances = {0: 35.0, 1: 110.0}
        fading = {0: 1.1, 1: 0.7}
        cap = battery_power_cap_w(
            self.cfg.initial_battery_j,
            self.cfg.frame_slots,
            self.cfg,
        )
        result = self.fast.solve_uav(
            users,
            z,
            distances,
            fading,
            self.cfg.initial_battery_j,
            self.cfg.frame_slots,
        )
        option_sets = [
            self.fast.feasible_uav_options(z[user], distances[user], fading[user], cap)
            for user in users
        ]
        feasible = [
            combo
            for combo in itertools.product(*option_sets)
            if sum(option.power_w for option in combo) <= cap + 1e-12
        ]
        brute_cost = min(
            sum(option.controllable_dpp_cost for option in combo) for combo in feasible
        )
        actual_cost = sum(option.controllable_dpp_cost for option in result.values())
        self.assertAlmostEqual(actual_cost, brute_cost, places=10)
        self.assertLessEqual(sum(option.power_w for option in result.values()), cap + 1e-12)

    def test_return_to_charge_diagnostic_has_explicit_denominator_event(self) -> None:
        depot = self.cfg.depot_x(0)
        away = depot + 50.0
        depleted = diagnose_return_to_charge(
            self.cfg.relocation_energy_j - 1.0,
            away,
            depot,
            True,
            self.cfg,
        )
        self.assertTrue(depleted.is_return_to_charge)
        self.assertTrue(depleted.depletion_before_arrival)
        reserve_breach = diagnose_return_to_charge(
            self.cfg.relocation_energy_j + self.cfg.reserve_battery_j - 1.0,
            away,
            depot,
            True,
            self.cfg,
        )
        self.assertFalse(reserve_breach.depletion_before_arrival)
        self.assertTrue(reserve_breach.reserve_breach_before_charge)

    def test_safe_return_reserve_covers_multiframe_path(self) -> None:
        depot = self.cfg.depot_x(0)
        required = safe_return_reserve_j(depot + 100.0, depot, self.cfg)
        expected = (
            self.cfg.reserve_battery_j
            + 2.0 * self.cfg.relocation_energy_j
            + self.cfg.frame_slots * self.cfg.hover_energy_per_slot_j
        )
        self.assertAlmostEqual(required, expected)

    def test_multiframe_return_records_need_without_stranding(self) -> None:
        depot = self.cfg.depot_x(0)
        self.state.uav_x[0] = depot + 100.0
        self.state.battery_j[0] = safe_return_reserve_j(
            self.state.uav_x[0], depot, self.cfg
        )
        users = self.users(0)
        controller = SlowRolloutController(self.cfg)

        first_candidates = controller.candidate_actions(
            self.state, 0, users, policy="nearest_hotspot", frame=0
        )
        first = first_candidates.actions[0]
        self.assertEqual(first.target_x(self.cfg), depot + 50.0)
        self.assertFalse(first.uav_users)
        first_result = simulate_region_frame(
            self.state,
            first,
            users,
            generate_frame_trace(self.cfg, 101),
            self.cfg,
            self.fast,
        )
        self.assertEqual(first_result.charging_need_events, 1)
        self.assertEqual(first_result.stranded_before_charge_events, 0)
        apply_region_result(self.state, 0, users, first_result)

        second_candidates = controller.candidate_actions(
            self.state, 0, users, policy="nearest_hotspot", frame=1
        )
        second = second_candidates.actions[0]
        self.assertEqual(second.hired, 0)
        second_result = simulate_region_frame(
            self.state,
            second,
            users,
            generate_frame_trace(self.cfg, 102),
            self.cfg,
            self.fast,
        )
        self.assertEqual(second_result.charging_need_events, 1)
        self.assertEqual(second_result.return_to_charge_events, 1)
        self.assertEqual(second_result.precharge_depletion_events, 0)
        self.assertEqual(second_result.precharge_reserve_breach_events, 0)

    def test_frame_tracks_quality_switch_and_distance_accounting(self) -> None:
        users = self.users(0)
        center = self.cfg.candidate_offsets_m.index(0.0)
        action = RegionAction(
            region=0,
            hired=1,
            point_index=center,
            rsu_users=tuple(users[2:]),
            uav_users=tuple(users[:2]),
        )
        result = simulate_region_frame(
            self.state,
            action,
            users,
            generate_frame_trace(self.cfg, 123),
            self.cfg,
            self.fast,
        )
        self.assertEqual(
            int(np.sum(result.distance_opportunities)),
            len(action.uav_users) * self.cfg.frame_slots,
        )
        self.assertAlmostEqual(np.sum(result.quality_histogram), result.delivered_chunks)
        self.assertEqual(result.power_violations, 0)
        self.assertEqual(result.provider_violations, 0)

    def test_rollout_is_deterministic(self) -> None:
        users = self.users(0)
        controller = SlowRolloutController(self.cfg)
        first = controller.select(self.state, 0, users, frame=0, policy="dpp")
        second = controller.select(self.state, 0, users, frame=0, policy="dpp")
        self.assertEqual(first.action, second.action)
        self.assertAlmostEqual(first.estimated_dpp_cost, second.estimated_dpp_cost)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for PPO tests")
    def test_ppo_features_and_choice_are_feasible(self) -> None:
        users = self.users(0)
        controller = SlowRolloutController(self.cfg)
        candidates = controller.candidate_actions(
            self.state, 0, users, policy="ppo", frame=0
        )
        state_features = build_state_features(self.state, 0, users, self.cfg)
        action_features = build_candidate_feature_matrix(
            self.state, candidates.actions, self.cfg
        )
        agent = PPOAgent(self.cfg, device="cpu")
        choice = agent.select(state_features, action_features)
        self.assertTrue(0 <= choice.action_index < len(candidates.actions))
        validate_region_action(candidates.actions[choice.action_index], users, self.cfg)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for PPO tests")
    def test_ppo_update_is_finite(self) -> None:
        agent = PPOAgent(self.cfg, device="cpu")
        _, transitions = train_episode(agent, self.cfg)
        metrics = agent.update(transitions)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            all(torch.isfinite(parameter).all() for parameter in agent.network.parameters())
        )

    def test_short_policy_run_has_requested_metrics_and_no_hard_violations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_policy(self.cfg, "dpp", Path(directory))
        summary = result.summary
        for key in (
            "average_quality_utility",
            "quality_p05_utility",
            "precharge_depletion_rate",
            "precharge_reserve_breach_rate",
            "stranded_before_charge_rate",
            "mean_uav_user_distance_m",
        ):
            self.assertIn(key, summary)
            self.assertTrue(math.isfinite(float(summary[key])))
        self.assertEqual(summary["battery_reserve_violations"], 0)
        self.assertEqual(summary["power_violations"], 0)
        self.assertEqual(summary["provider_violations"], 0)
        self.assertEqual(len(result.distance_rows), self.cfg.num_distance_bins)
        self.assertEqual(len(result.point_rows), len(self.cfg.candidate_offsets_m))

    def test_long_baseline_returns_to_charge_without_dead_end(self) -> None:
        cfg = replace(self.cfg, num_frames=100, seed=3026)
        with tempfile.TemporaryDirectory() as directory:
            result = run_policy(cfg, "nearest_hotspot", Path(directory))
        self.assertGreater(result.summary["return_to_charge_events"], 0)
        self.assertEqual(result.summary["stranded_before_charge_events"], 0)
        self.assertEqual(result.summary["precharge_depletion_events"], 0)
        self.assertEqual(result.summary["battery_reserve_violations"], 0)


if __name__ == "__main__":
    unittest.main()
