from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from agent.P3.exact_fast_controller import ExactFastController
from config_p3 import P3Config
from env.p3.environment import generate_frame_trace, simulate_region_frame
from env.p3.topology import (
    enumerate_region_actions,
    shortlist_region_users,
)
from env.p3.types import P3State, RegionAction


SUPPORTED_RULE_POLICIES = (
    "dpp",
    "rsu_only",
    "always_hire",
    "fixed_rsu",
    "nearest_hotspot",
    "load_threshold",
    "random",
)


@dataclass(frozen=True)
class CandidateSet:
    actions: tuple[RegionAction, ...]
    shortlisted_users: tuple[int, ...]
    enumerated_count: int
    evaluated_count: int


@dataclass(frozen=True)
class SelectionResult:
    action: RegionAction
    estimated_dpp_cost: float
    enumerated_count: int
    evaluated_count: int


def _score_action_batch(
    state: P3State,
    actions: tuple[RegionAction, ...],
    region_users: tuple[int, ...],
    traces: tuple,
    cfg: P3Config,
) -> list[float]:
    """Evaluate an independent action batch in a worker process."""

    fast_controller = ExactFastController(cfg)
    return [
        float(
            np.mean(
                [
                    simulate_region_frame(
                        state,
                        action,
                        region_users,
                        trace,
                        cfg,
                        fast_controller,
                    ).frame_dpp_cost
                    for trace in traces
                ]
            )
        )
        for action in actions
    ]


class SlowRolloutController:
    """Equations (8.2)-(8.3): structured frame-level rollout."""

    def __init__(self, cfg: P3Config, rollout_workers: int = 1) -> None:
        self.cfg = cfg
        self.fast_controller = ExactFastController(cfg)
        self.rollout_workers = int(rollout_workers)
        if self.rollout_workers <= 0:
            raise ValueError("rollout_workers must be positive")
        self._executor: ProcessPoolExecutor | None = None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def candidate_actions(
        self,
        state: P3State,
        region: int,
        region_users: Sequence[int],
        policy: str = "dpp",
        frame: int = 0,
    ) -> CandidateSet:
        if policy not in SUPPORTED_RULE_POLICIES and policy != "ppo":
            raise ValueError(f"unknown policy: {policy}")
        shortlist = shortlist_region_users(state, region, region_users, self.cfg)
        all_actions = enumerate_region_actions(
            state,
            region,
            region_users,
            self.cfg,
            candidate_users=shortlist,
        )
        enumerated_count = len(all_actions)
        filtered = self._filter_policy(
            all_actions,
            state,
            region,
            region_users,
            policy,
        )
        if not filtered:
            raise RuntimeError(f"policy {policy!r} has no feasible action")
        pruned = self._prune_actions(filtered, state, region)
        return CandidateSet(
            actions=tuple(pruned),
            shortlisted_users=shortlist,
            enumerated_count=enumerated_count,
            evaluated_count=len(pruned),
        )

    def select(
        self,
        state: P3State,
        region: int,
        region_users: Sequence[int],
        frame: int,
        policy: str = "dpp",
    ) -> SelectionResult:
        candidates = self.candidate_actions(
            state,
            region,
            region_users,
            policy=policy,
            frame=frame,
        )
        actions = candidates.actions
        if policy == "random":
            rng = np.random.default_rng(self.rollout_seed(frame, region, 999_983))
            action = actions[int(rng.integers(0, len(actions)))]
            return SelectionResult(
                action=action,
                estimated_dpp_cost=math.nan,
                enumerated_count=candidates.enumerated_count,
                evaluated_count=candidates.evaluated_count,
            )

        traces = [
            generate_frame_trace(
                self.cfg,
                self.rollout_seed(frame, region, scenario),
            )
            for scenario in range(self.cfg.rollout_scenarios)
        ]
        scores = self._score_actions(
            state,
            actions,
            tuple(int(user) for user in region_users),
            tuple(traces),
        )
        best_action: RegionAction | None = None
        best_score = math.inf
        for action, score in zip(actions, scores):
            if self._is_better(action, score, best_action, best_score):
                best_action = action
                best_score = score
        if best_action is None:
            raise RuntimeError("slow action selection failed")
        return SelectionResult(
            action=best_action,
            estimated_dpp_cost=float(best_score),
            enumerated_count=candidates.enumerated_count,
            evaluated_count=candidates.evaluated_count,
        )

    def _score_actions(
        self,
        state: P3State,
        actions: Sequence[RegionAction],
        region_users: tuple[int, ...],
        traces: tuple,
    ) -> list[float]:
        actions = tuple(actions)
        if self.rollout_workers == 1 or len(actions) <= 1:
            return _score_action_batch(
                state,
                actions,
                region_users,
                traces,
                self.cfg,
            )
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.rollout_workers)
        batch_count = min(len(actions), self.rollout_workers * 2)
        batch_size = int(math.ceil(len(actions) / batch_count))
        batches = [
            actions[start : start + batch_size]
            for start in range(0, len(actions), batch_size)
        ]
        futures = [
            self._executor.submit(
                _score_action_batch,
                state,
                batch,
                region_users,
                traces,
                self.cfg,
            )
            for batch in batches
        ]
        scores: list[float] = []
        for future in futures:
            scores.extend(future.result())
        if len(scores) != len(actions):
            raise RuntimeError("parallel rollout returned an invalid score count")
        return scores

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
        actions: Sequence[RegionAction],
        state: P3State,
        region: int,
        region_users: Sequence[int],
        policy: str,
    ) -> list[RegionAction]:
        actions = list(actions)
        if policy in ("dpp", "ppo", "random"):
            return actions
        no_hire = [action for action in actions if action.hired == 0]
        hired = [action for action in actions if action.hired == 1]
        if policy == "rsu_only":
            return no_hire
        if policy == "always_hire":
            return hired or no_hire
        if policy == "load_threshold":
            return (
                (hired or no_hire)
                if len(region_users) > self.cfg.rsu_capacity
                else no_hire
            )
        if policy == "fixed_rsu":
            points = self.cfg.candidate_points(region)
            center = min(
                range(len(points)),
                key=lambda index: abs(points[index] - self.cfg.rsu_x(region)),
            )
            fixed = [action for action in hired if action.point_index == center]
            return fixed or no_hire
        if policy == "nearest_hotspot":
            if not region_users:
                return no_hire
            hotspot = float(np.mean(state.user_x[list(region_users)]))
            points = self.cfg.candidate_points(region)
            nearest = min(range(len(points)), key=lambda index: abs(points[index] - hotspot))
            selected = [action for action in hired if action.point_index == nearest]
            return selected or no_hire
        raise ValueError(f"unknown policy: {policy}")

    def _prune_actions(
        self,
        actions: Sequence[RegionAction],
        state: P3State,
        region: int,
    ) -> list[RegionAction]:
        if len(actions) <= self.cfg.max_rollout_actions:
            return list(actions)

        def heuristic(action: RegionAction) -> tuple[float, tuple]:
            scheduled = action.rsu_users + action.uav_users
            urgency = sum(
                cfg_z / self.cfg.large_queue_level
                for cfg_z in (
                    self.cfg.large_queue_level - float(state.queue[user])
                    for user in scheduled
                )
            )
            point = action.target_x(self.cfg)
            uav_distance = sum(abs(point - float(state.user_x[user])) for user in action.uav_users)
            move_distance = abs(point - float(state.uav_x[region]))
            score = (
                urgency
                - 0.25 * uav_distance / self.cfg.region_length_m
                - 0.10 * move_distance / max(self.cfg.reachable_distance_m, 1e-9)
                - 0.05 * action.hired
            )
            return score, self._action_key(action)

        ranked = sorted(actions, key=heuristic, reverse=True)
        return ranked[: self.cfg.max_rollout_actions]

    @staticmethod
    def _action_key(action: RegionAction) -> tuple:
        return (
            action.hired,
            action.point_index,
            action.uav_users,
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
        return best_action is None or self._action_key(action) < self._action_key(best_action)
