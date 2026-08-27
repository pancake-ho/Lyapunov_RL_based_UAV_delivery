from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from agent.P1.exact_fast_controller import ExactFastController
from config_p1 import P1Config
from env.p1.environment import generate_frame_trace, simulate_region_frame
from env.p1.topology import enumerate_region_actions
from env.p1.types import P1State, RegionAction


SUPPORTED_POLICIES = ("dpp", "rsu_only", "always_hire", "fixed_rsu")


@dataclass(frozen=True)
class SelectionResult:
    action: RegionAction
    estimated_dpp_cost: float
    candidate_count: int


class SlowRolloutController:
    """Finite P1 action enumeration with sample-average exact-fast rollout."""

    def __init__(self, cfg: P1Config) -> None:
        self.cfg = cfg
        self.fast_controller = ExactFastController(cfg)

    def select(
        self,
        state: P1State,
        region: int,
        region_users: Sequence[int],
        frame: int,
        policy: str = "dpp",
    ) -> SelectionResult:
        actions = enumerate_region_actions(
            state,
            region,
            region_users,
            self.cfg,
        )
        actions = self._filter_policy(actions, region, policy)
        if not actions:
            raise RuntimeError(f"policy {policy!r} has no feasible action")

        # Every action sees exactly the same stochastic scenarios.
        traces = [
            generate_frame_trace(
                self.cfg,
                self.rollout_seed(frame, region, scenario),
            )
            for scenario in range(self.cfg.rollout_scenarios)
        ]
        best_action: RegionAction | None = None
        best_score = math.inf
        for action in actions:
            score = float(
                np.mean(
                    [
                        simulate_region_frame(
                            state,
                            action,
                            region_users,
                            trace,
                            self.cfg,
                            self.fast_controller,
                        ).frame_dpp_cost
                        for trace in traces
                    ]
                )
            )
            if self._is_better(action, score, best_action, best_score):
                best_action = action
                best_score = score

        if best_action is None:
            raise RuntimeError("slow action selection failed")
        return SelectionResult(
            action=best_action,
            estimated_dpp_cost=float(best_score),
            candidate_count=len(actions),
        )

    def rollout_seed(self, frame: int, region: int, scenario: int) -> int:
        return int(
            self.cfg.seed
            + 10_000_019
            + frame * 100_003
            + region * 1_009
            + scenario
        )

    def realized_seed(self, frame: int) -> int:
        return int(self.cfg.seed + 20_000_033 + frame * 100_003)

    def _filter_policy(
        self,
        actions: list[RegionAction],
        region: int,
        policy: str,
    ) -> list[RegionAction]:
        if policy == "dpp":
            return actions
        if policy == "rsu_only":
            return [action for action in actions if action.hired == 0]
        if policy == "always_hire":
            hired = [action for action in actions if action.hired == 1]
            return hired or [action for action in actions if action.hired == 0]
        if policy == "fixed_rsu":
            points = self.cfg.candidate_points(region)
            center_index = min(
                range(len(points)),
                key=lambda index: abs(
                    points[index] - self.cfg.rsu_x(region)
                ),
            )
            fixed = [
                action
                for action in actions
                if action.hired == 1 and action.point_index == center_index
            ]
            return fixed or [
                action for action in actions if action.hired == 0
            ]
        raise ValueError(
            f"unknown policy {policy!r}; choose one of {SUPPORTED_POLICIES}"
        )

    @staticmethod
    def _action_key(action: RegionAction) -> tuple:
        return (
            action.hired,
            action.point_index,
            action.uav_user if action.uav_user is not None else -1,
            action.rsu_users,
        )

    def _is_better(
        self,
        action: RegionAction,
        score: float,
        best_action: RegionAction | None,
        best_score: float,
    ) -> bool:
        if score < best_score - 1e-9:
            return True
        if abs(score - best_score) > 1e-9:
            return False
        return (
            best_action is None
            or self._action_key(action) < self._action_key(best_action)
        )
