from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent.P1.exact_fast_controller import ExactFastController
from agent.P1.slow_rollout_controller import SlowRolloutController
from config_p1 import P1Config
from env.p1.battery import battery_power_cap_w
from env.p1.environment import generate_frame_trace, simulate_region_frame
from env.p1.radio import capacity_bps, link_gain, required_uav_power_w
from env.p1.topology import (
    enumerate_region_actions,
    initialize_state,
    region_membership,
    validate_region_action,
)
from env.p1.types import RegionAction
from run.p1_labmeeting import run_policy


class P1ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = P1Config(num_frames=2, rollout_scenarios=1)
        self.state = initialize_state(self.cfg)
        self.fast = ExactFastController(self.cfg)

    def users(self, region: int) -> list[int]:
        membership = region_membership(self.state, self.cfg)
        return np.flatnonzero(membership == region).tolist()

    def test_enumeration_enforces_p1_capacity_and_single_provider(self) -> None:
        for region in range(self.cfg.num_regions):
            users = self.users(region)
            actions = enumerate_region_actions(
                self.state,
                region,
                users,
                self.cfg,
            )
            self.assertGreater(len(actions), 0)
            for action in actions:
                validate_region_action(action, users, self.cfg)
                self.assertLessEqual(len(action.rsu_users), self.cfg.rsu_capacity)
                self.assertLessEqual(int(action.uav_user is not None), 1)
                if action.uav_user is not None:
                    self.assertNotIn(action.uav_user, action.rsu_users)

    def test_rsu_exact_solver_matches_full_option_bruteforce(self) -> None:
        z = 90.0
        horizontal = 120.0
        fading = 0.8
        option = self.fast.solve_rsu(z, horizontal, fading)

        bandwidth = self.cfg.rsu_total_bandwidth_hz / self.cfg.rsu_capacity
        power = self.cfg.rsu_total_power_w / self.cfg.rsu_capacity
        vertical = self.cfg.rsu_height_m - self.cfg.user_height_m
        distance = math.hypot(horizontal, vertical)
        gain = link_gain(
            self.cfg.rsu_beta0,
            distance,
            self.cfg.rsu_pathloss_exp,
            fading,
        )
        capacity = capacity_bps(bandwidth, power, gain, self.cfg)
        brute_costs = [0.0]
        for quality_index, chunk_bits in enumerate(self.cfg.chunk_size_bits):
            feasible = min(
                self.cfg.max_chunks_per_slot,
                int(
                    math.floor(
                        capacity * self.cfg.slot_duration_s / chunk_bits
                    )
                ),
            )
            for chunks in range(1, feasible + 1):
                brute_costs.append(
                    self.fast.controllable_cost(z, chunks, quality_index)
                )
        self.assertAlmostEqual(
            option.controllable_dpp_cost,
            min(brute_costs),
            places=10,
        )

    def test_minimum_uav_power_satisfies_rate_equality(self) -> None:
        chunks = 1
        quality_index = 1
        horizontal = 40.0
        fading = 1.2
        power = required_uav_power_w(
            chunks,
            quality_index,
            horizontal,
            fading,
            self.cfg,
        )
        vertical = self.cfg.uav_height_m - self.cfg.user_height_m
        distance = math.hypot(horizontal, vertical)
        gain = link_gain(
            self.cfg.uav_beta0,
            distance,
            self.cfg.uav_pathloss_exp,
            fading,
        )
        capacity = capacity_bps(
            self.cfg.uav_total_bandwidth_hz,
            power,
            gain,
            self.cfg,
        )
        payload = chunks * self.cfg.chunk_size_bits[quality_index]
        self.assertAlmostEqual(
            capacity * self.cfg.slot_duration_s,
            payload,
            delta=max(1e-6, payload * 1e-10),
        )

    def test_battery_power_cap_preserves_remaining_hover_reserve(self) -> None:
        remaining = self.cfg.frame_slots
        cap = battery_power_cap_w(
            self.cfg.reserve_battery_j
            + remaining * self.cfg.hover_energy_per_slot_j,
            remaining,
            self.cfg,
        )
        self.assertAlmostEqual(cap, 0.0)

    def test_unhired_charges_and_hired_preserves_reserve(self) -> None:
        region = 0
        users = self.users(region)
        trace = generate_frame_trace(self.cfg, 123)

        charge_state = self.state.copy()
        charge_state.battery_j[region] = 0.5 * self.cfg.battery_capacity_j
        no_hire = RegionAction(region, 0, -1, tuple(), None)
        charged = simulate_region_frame(
            charge_state,
            no_hire,
            users,
            trace,
            self.cfg,
        )
        self.assertGreater(
            charged.battery_after_j,
            charge_state.battery_j[region],
        )
        self.assertLessEqual(
            charged.battery_after_j,
            self.cfg.battery_capacity_j,
        )

        center_index = self.cfg.candidate_offsets_m.index(0.0)
        hired = RegionAction(
            region,
            1,
            center_index,
            tuple(users[1 : 1 + self.cfg.rsu_capacity]),
            users[0],
        )
        served = simulate_region_frame(
            self.state,
            hired,
            users,
            trace,
            self.cfg,
        )
        self.assertGreaterEqual(
            served.battery_after_j + 1e-7,
            self.cfg.reserve_battery_j,
        )
        self.assertEqual(served.battery_reserve_violations, 0)
        self.assertEqual(served.power_violations, 0)

    def test_queue_update_is_exact_and_not_clipped_at_large_q(self) -> None:
        region = 0
        users = self.users(region)
        self.state.queue[users[0]] = self.cfg.large_queue_level + 2.0
        result = simulate_region_frame(
            self.state,
            RegionAction(region, 0, -1, tuple(), None),
            users,
            generate_frame_trace(self.cfg, 456),
            self.cfg,
        )
        local_index = users.index(users[0])
        expected = (
            self.cfg.large_queue_level
            + 2.0
            - self.cfg.frame_slots * self.cfg.playback_chunks_per_slot
        )
        self.assertAlmostEqual(result.queue_after[local_index], expected)
        self.assertGreater(result.large_queue_violation_user_slots, 0)

    def test_hiring_cost_is_added_once_per_frame(self) -> None:
        region = 0
        users = self.users(region)
        center_index = self.cfg.candidate_offsets_m.index(0.0)
        action = RegionAction(
            region,
            1,
            center_index,
            tuple(users[1 : 1 + self.cfg.rsu_capacity]),
            users[0],
        )
        trace = generate_frame_trace(self.cfg, 789)
        with_cost = simulate_region_frame(
            self.state,
            action,
            users,
            trace,
            self.cfg,
        )
        zero_cost_cfg = replace(self.cfg, hiring_cost_per_frame=0.0)
        without_cost = simulate_region_frame(
            self.state,
            action,
            users,
            trace,
            zero_cost_cfg,
        )
        self.assertAlmostEqual(
            with_cost.original_cost - without_cost.original_cost,
            self.cfg.lambda_h * self.cfg.hiring_cost_per_frame,
        )
        self.assertAlmostEqual(
            with_cost.frame_dpp_cost - without_cost.frame_dpp_cost,
            self.cfg.lyapunov_v
            * self.cfg.lambda_h
            * self.cfg.hiring_cost_per_frame,
        )

    def test_common_random_rollout_is_deterministic(self) -> None:
        region = 0
        users = self.users(region)
        controller = SlowRolloutController(self.cfg)
        first = controller.select(self.state, region, users, frame=0)
        second = controller.select(self.state, region, users, frame=0)
        self.assertEqual(first.action, second.action)
        self.assertAlmostEqual(
            first.estimated_dpp_cost,
            second.estimated_dpp_cost,
        )

    def test_short_run_has_zero_hard_constraint_violations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, rows = run_policy(
                self.cfg,
                "dpp",
                Path(directory),
            )
        self.assertEqual(len(rows), self.cfg.num_frames)
        self.assertEqual(summary["battery_reserve_violations"], 0)
        self.assertEqual(summary["power_violations"], 0)
        self.assertEqual(summary["provider_violations"], 0)


if __name__ == "__main__":
    unittest.main()
