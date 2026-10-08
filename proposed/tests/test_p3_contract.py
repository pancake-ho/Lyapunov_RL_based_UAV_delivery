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
from agent.P3.features import build_candidate_signatures, build_state_features
from agent.P3.slow_rollout_controller import SlowRolloutController
from config_p3 import P3Config
from env.p3.battery import (
    activation_energy_required_j,
    apply_active_slot,
    battery_power_cap_w,
    diagnose_return_to_charge,
)
from env.p3.environment import (
    generate_frame_trace,
    simulate_region_frame,
    update_playback_queue,
)
from env.p3.topology import (
    enumerate_region_actions,
    initialize_state,
    region_membership,
    validate_region_action,
)
from env.p3.types import RegionAction
from run.p3_common import array_max_or_current, run_policy


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
        with self.assertRaises(ValueError):
            replace(
                self.cfg,
                reserve_battery_j=self.cfg.relocation_energy_j - 1.0,
            )

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

    def test_battery_equations_count_threshold_exactly_once(self) -> None:
        required = activation_energy_required_j(self.cfg)
        expected = (
            self.cfg.reserve_battery_j
            + self.cfg.frame_slots * self.cfg.hover_energy_per_slot_j
        )
        self.assertAlmostEqual(required, expected)

        remaining = 5
        target_power = 1.25
        battery = (
            self.cfg.reserve_battery_j
            + remaining * self.cfg.hover_energy_per_slot_j
            + self.cfg.slot_duration_s * target_power / self.cfg.pa_efficiency
        )
        cap = battery_power_cap_w(battery, remaining, self.cfg)
        self.assertAlmostEqual(cap, target_power)

        step = apply_active_slot(
            battery,
            target_power,
            remaining,
            self.cfg,
        )
        expected_consumed = (
            self.cfg.hover_energy_per_slot_j
            + self.cfg.slot_duration_s * target_power / self.cfg.pa_efficiency
        )
        self.assertAlmostEqual(step.consumed_j, expected_consumed)
        self.assertAlmostEqual(step.battery_after_j, battery - expected_consumed)

    def test_return_to_charge_diagnostic_checks_pre_return_threshold(self) -> None:
        depot = self.cfg.depot_x(0)
        away = depot + 60.0
        depleted = diagnose_return_to_charge(
            self.cfg.relocation_energy_j - 1.0,
            away,
            depot,
            True,
            self.cfg,
        )
        self.assertTrue(depleted.is_return_to_charge)
        self.assertTrue(depleted.depletion_before_arrival)
        self.assertTrue(depleted.reserve_breach_before_charge)

        reserve_breach = diagnose_return_to_charge(
            self.cfg.reserve_battery_j - 1.0,
            away,
            depot,
            True,
            self.cfg,
        )
        self.assertFalse(reserve_breach.depletion_before_arrival)
        self.assertTrue(reserve_breach.reserve_breach_before_charge)

        safe = diagnose_return_to_charge(
            self.cfg.reserve_battery_j,
            away,
            depot,
            True,
            self.cfg,
        )
        self.assertFalse(safe.depletion_before_arrival)
        self.assertFalse(safe.reserve_breach_before_charge)

    def test_unhired_uav_returns_then_charges_by_equations_4_4_to_4_6(self) -> None:
        depot = self.cfg.depot_x(0)
        self.state.uav_x[0] = depot + 60.0
        self.state.battery_j[0] = self.cfg.reserve_battery_j
        users = self.users(0)
        action = RegionAction(
            region=0,
            hired=0,
            point_index=-1,
            rsu_users=tuple(users[: self.cfg.rsu_capacity]),
            uav_users=tuple(),
        )
        result = simulate_region_frame(
            self.state,
            action,
            users,
            generate_frame_trace(self.cfg, 101),
            self.cfg,
            self.fast,
        )
        expected = min(
            self.cfg.battery_capacity_j,
            self.cfg.reserve_battery_j
            - self.cfg.relocation_energy_j
            + self.cfg.frame_slots * self.cfg.charge_energy_per_slot_j,
        )
        self.assertAlmostEqual(result.battery_after_j, expected)
        self.assertEqual(result.return_to_charge_events, 1)
        self.assertEqual(result.precharge_depletion_events, 0)
        self.assertEqual(result.precharge_reserve_breach_events, 0)
        self.assertEqual(result.charging_slots, self.cfg.frame_slots)

    def test_queue_update_preserves_virtual_queue_identity(self) -> None:
        cases = ((0.0, 0.0), (0.4, 1.0), (5.0, 1.0), (100.5, 0.0))
        for queue_before, delivered in cases:
            queue_after, z_before, z_after, actual_departure = (
                update_playback_queue(queue_before, delivered, self.cfg)
            )
            self.assertAlmostEqual(
                queue_after,
                max(queue_before - self.cfg.playback_chunks_per_slot, 0.0)
                + delivered,
            )
            self.assertAlmostEqual(
                z_after,
                z_before + actual_departure - delivered,
            )
            self.assertAlmostEqual(
                z_after,
                self.cfg.large_queue_level - queue_after,
            )

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

    def test_factorized_state_and_signatures_preserve_global_user_ids(self) -> None:
        membership = region_membership(self.state, self.cfg)
        users = np.flatnonzero(membership == 1).tolist()
        controller = SlowRolloutController(self.cfg)
        candidates = controller.candidate_actions(
            self.state, 1, users, policy="ppo", frame=0
        )
        signatures = build_candidate_signatures(candidates.actions, self.cfg)
        self.assertEqual(signatures.shape[1], self.cfg.ppo_signature_dim)
        self.assertEqual(np.unique(signatures, axis=0).shape[0], len(candidates.actions))

        features = build_state_features(self.state, 1, users, self.cfg)
        user_matrix = features[self.cfg.ppo_global_feature_dim :].reshape(
            self.cfg.num_users, self.cfg.ppo_user_feature_dim
        )
        present = set(np.flatnonzero(user_matrix[:, 0] > 0.5).tolist())
        self.assertEqual(present, set(users))

    def test_empty_region_frame_has_safe_queue_statistics(self) -> None:
        empty_region = 0
        self.state.user_x[:] = self.cfg.rsu_x(1)
        users = self.users(empty_region)
        self.assertEqual(users, [])
        actions = enumerate_region_actions(
            self.state, empty_region, users, self.cfg
        )
        action = next(candidate for candidate in actions if candidate.hired == 0)
        result = simulate_region_frame(
            self.state,
            action,
            users,
            generate_frame_trace(self.cfg, 456),
            self.cfg,
            self.fast,
        )
        self.assertEqual(result.queue_samples.size, 0)
        self.assertEqual(result.z_samples.size, 0)
        self.assertEqual(
            array_max_or_current(7.0, result.queue_samples),
            7.0,
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for PPO tests")
    def test_ppo_features_and_choice_are_feasible(self) -> None:
        users = self.users(0)
        controller = SlowRolloutController(self.cfg)
        candidates = controller.candidate_actions(
            self.state, 0, users, policy="ppo", frame=0
        )
        state_features = build_state_features(self.state, 0, users, self.cfg)
        action_signatures = build_candidate_signatures(candidates.actions, self.cfg)
        agent = PPOAgent(self.cfg, device="cpu")
        choice = agent.select(state_features, action_signatures)
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
        self.assertEqual(len(result.user_rows), self.cfg.num_users)
        self.assertGreaterEqual(summary["p95_queue"], summary["mean_queue"])
        self.assertGreaterEqual(summary["jain_service_fairness"], 0.0)

    def test_default_queue_has_nonvacuous_prefetch_range(self) -> None:
        self.assertGreater(
            self.cfg.max_chunks_per_slot,
            self.cfg.playback_chunks_per_slot,
        )
        queue_after, _, _, _ = update_playback_queue(
            self.cfg.initial_playback_queue,
            self.cfg.max_chunks_per_slot,
            self.cfg,
        )
        self.assertGreater(queue_after, self.cfg.initial_playback_queue)

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
