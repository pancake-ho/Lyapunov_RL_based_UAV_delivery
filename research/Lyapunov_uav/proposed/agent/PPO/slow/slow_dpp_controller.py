from __future__ import annotations

import copy
import math
from itertools import product
from typing import Any, Dict, Iterator, Mapping, Tuple

import numpy as np
import torch

from agent.PPO.common import split_env_step
from agent.PPO.fast.fast_agent import FastPPOAgent
from env.env import Env

from .slow_config import SlowDPPConfig


SlowEnvAction = Dict[str, np.ndarray]


class SlowDPPController:
    """
    Slow policy train 과정이 없는 round-level DPP controller
    """
    UNSCHEDULED = 0
    RSU = 1
    UAV = 2

    def __init__(self, env_cfg: Any, dpp_cfg: SlowDPPConfig) -> None:
        self.env_cfg = env_cfg
        self.dpp_cfg = dpp_cfg

        self.num_rsu = int(env_cfg.num_rsu)
        self.num_user = int(env_cfg.num_user)
        self.num_uav = int(env_cfg.num_uav)

        self._score_cache: Dict[bytes, Tuple[float, Dict[str, Any]]] = {}
        self._candidate_requests = 0
        self._finite_candidate_requests = 0
    
    @staticmethod
    def _binary_copy(value: Any) -> np.ndarray:
        """
        binary-type decision 복사본 반환 수행
        """
        return np.asarray(value, dtype=np.int32).copy()
    
    def _empty_action(self) -> SlowEnvAction:
        """
        모든 decision을 0으로 반환
        """
        return {
            "rsu_scheduling": np.zeros((self.num_rsu, self.num_user), dtype=np.int32),
            "uav_hiring": np.zeros(self.num_uav, dtype=np.int32),
            "uav_scheduling": np.zeros((self.num_uav, self.num_user), dtype=np.int32),
        }
    
    def _copy_action(self, action: Mapping[str, Any]) -> SlowEnvAction:
        return {
            "rsu_scheduling": self._binary_copy(action["rsu_scheduling"]),
            "uav_hiring": self._binary_copy(action["uav_hiring"]),
            "uav_scheduling": self._binary_copy(action["uav_scheduling"]),
        }
    
    @staticmethod
    def _is_binary(name: str, value: np.ndarray) -> None:
        if np.any((value != 0) & (value != 1)):
            raise ValueError(f"{name}은 0 또는 1을 포함해야 합니다.")
    
    @staticmethod
    def _action_key(action: Mapping[str, np.ndarray]) -> bytes:
        return b"|".join(
            np.asarray(action[name], dtype=np.int8).tobytes()
            for name in (
                "rsu_scheduling", "uav_hiring", "uav_scheduling",
            )
        )
    
    def _validate_action(self, env: Env, action: Mapping[str, Any]) -> None:
        rsu = np.asarray(action["rsu_scheduling"], dtype=np.int32)
        hiring = np.asarray(action["uav_hiring"], dtype=np.int32)
        uav = np.asarray(action["uav_scheduling"], dtype=np.int32)

        if rsu.shape != (self.num_rsu, self.num_user):
            raise ValueError(
                "rsu_scheduling has an invalid shape: "
                f"{rsu.shape}"
            )
        if hiring.shape != (self.num_uav,):
            raise ValueError(
                "uav_hiring has an invalid shape: "
                f"{hiring.shape}"
            )
        if uav.shape != (self.num_uav, self.num_user):
            raise ValueError(
                "uav_scheduling has an invalid shape: "
                f"{uav.shape}"
            )

        self._is_binary("rsu_scheduling", rsu)
        self._is_binary("uav_hiring", hiring)
        self._is_binary("uav_scheduling", uav)

        region = np.asarray(env.user_region, dtype=np.int32).reshape(-1)
        requested = np.asarray(env.requested_content, dtype=np.int32).reshape(-1)
        cached = np.asarray(env.uav_cached_content, dtype=np.int32).reshape(-1)

        # mask 및 검증
        rsu_region_mask = (np.arange(self.num_rsu, dtype=np.int32)[:, None] == region[None, :])
        uav_region_mask = (np.arange(self.num_uav, dtype=np.int32)[:, None] == region[None, :])
        cache_match = (cached[:, None] == requested[None, :])

        if np.any((rsu == 1) & ~rsu_region_mask):
            raise ValueError("RSU slow action contains a cross-region link.")
        if np.any((uav == 1) & ~uav_region_mask):
            raise ValueError("UAV slow action contains a cross-region link.")
        if np.any((uav == 1) & ~cache_match):
            raise ValueError("UAV slow action violates the cache constraint.")
        if np.any(uav > hiring[:, None]):
            raise ValueError("UAV scheduling requires uav_hiring=1.")
        if np.any(
            uav.sum(axis=1)
            > int(self.env_cfg.uav_user_cap)
        ):
            raise ValueError(
                "UAV slow candidates exceed uav_user_cap."
            )
        
        provider_count = rsu.sum(axis=0) + uav.sum(axis=0)
        if np.any(provider_count > 1):
            raise ValueError(
                "A user cannot be an RSU and UAV slow candidate together."
            )

    def _initial_rsu_first_action(self, env: Env) -> SlowEnvAction:
        """
        RSU (scheduling) action 초기화 수행
        """
        # 빈 action 생성
        action = self._empty_action()
        region = np.asarray(env.user_region, dtype=np.int32).reshape(-1)

        for user_idx, region_idx in enumerate(region):
            action["rsu_scheduling"][int(region_idx), user_idx] = 1
        self._validate_action(env, action)
        return action
    
    def _region_assignment_count(self, env: Env, region_idx: int) -> int:
        region = np.asarray(env.user_region, dtype=np.int32)
        requested = np.asarray(env.requested_content, dtype=np.int32)
        cached = np.asarray(env.uav_cached_content, dtype=np.int32)
        users = np.flatnonzero(region == int(region_idx))

        compatible = int(
            np.sum(requested[users] == cached[region_idx])
        )
        incompatible = int(users.size) - compatible
        cap = min(
            compatible,
            int(self.env_cfg.uav_user_cap),
        )

        assignment_count = 0
        for uav_count in range(cap + 1):
            assignment_count += (
                math.comb(compatible, uav_count)
                * (
                    2
                    ** (
                        compatible
                        - uav_count
                        + incompatible
                    )
                )
            )

        # When no user is assigned to the UAV, both mu=0 and mu=1 are
        # feasible. The assignment_count above contains that case once.
        no_uav_assignments = 2 ** int(users.size)
        return int(
            assignment_count + no_uav_assignments
        )

    def _iter_region_candidates(self, env: Env, base_action: Mapping[str, np.ndarray], region_idx: int) -> Iterator[SlowEnvAction]:
        region = np.asarray(env.user_region, dtype=np.int32)
        requested = np.asarray(env.requested_content, dtype=np.int32)
        cached = np.asarray(env.uav_cached_content, dtype=np.int32)
        users = np.flatnonzero(region == int(region_idx))

        candidate_count = self._region_assignment_count(env, region_idx)
        candidate_limit = int(
            getattr(
                self.dpp_cfg,
                "max_exact_region_candidates",
                getattr(
                    self.dpp_cfg,
                    "max_region_candidates",
                    0,
                ),
            )
        )
        if candidate_count > candidate_limit:
            raise RuntimeError(
                "The exact local assignment set exceeds the configured "
                "safety limit: "
                f"region={region_idx}, users={users.size}, "
                f"candidates={candidate_count}, "
                f"limit={candidate_limit}."
            )

        choices = []
        for user_idx in users:
            user_choices = [self.UNSCHEDULED, self.RSU]
            if int(requested[user_idx]) == int(cached[region_idx]):
                user_choices.append(self.UAV)
            choices.append(tuple(user_choices))

        assignments = product(*choices) if choices else [tuple()]

        for labels in assignments:
            uav_count = sum(
                int(label) == self.UAV
                for label in labels
            )
            if uav_count > int(self.env_cfg.uav_user_cap):
                continue

            # phi=1 requires mu=1. With phi=0, mu remains an independent
            # feasible decision and both employment values are enumerated.
            hiring_choices = (1,) if uav_count > 0 else (0, 1)
            for hiring_value in hiring_choices:
                action = self._copy_action(base_action)
                action["rsu_scheduling"][region_idx, :] = 0
                action["uav_scheduling"][region_idx, :] = 0
                action["uav_hiring"][region_idx] = int(
                    hiring_value
                )

                for user_idx, label in zip(users, labels):
                    if int(label) == self.RSU:
                        action["rsu_scheduling"][
                            region_idx,
                            user_idx,
                        ] = 1
                    elif int(label) == self.UAV:
                        action["uav_scheduling"][
                            region_idx,
                            user_idx,
                        ] = 1
                    elif int(label) != self.UNSCHEDULED:
                        raise RuntimeError(
                            f"Unknown provider label: {label}"
                        )

                self._validate_action(env, action)
                yield action
    
    def _forecast_seed(self, env: Env, scenario_idx: int) -> int:
        # Forecast randomness is independent from env.rng, so candidate
        # evaluation cannot peek at the actual environment's future samples.
        # All candidates use the same scenario seeds for fair comparison.
        round_number = int(env.t) // max(1, int(env.slow_T))
        value = (
            int(self.dpp_cfg.forecast_seed_offset)
            + 1_000_003 * int(self.dpp_cfg.seed)
            + 10_007 * int(env.episode)
            + 1_009 * int(round_number)
            + int(scenario_idx)
        )
        return int(value % (2**63 - 1))

    @staticmethod
    def _extract_boundary_cost(
        boundary_info: Mapping[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        reward_components = boundary_info.get("reward_components", {})
        slow_reward = float(reward_components.get("slow_reward", math.nan))
        slow_components = dict(
            reward_components.get("slow_reward_components", {})
        )

        if not bool(slow_components.get("is_round_boundary", False)):
            raise RuntimeError(
                "Forecast rollout ended without a complete round DPP value."
            )
        if not math.isfinite(slow_reward):
            raise RuntimeError("Forecast slow_reward is NaN or Inf.")

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
                "Forecast slow reward fields disagree at round boundary."
            )

        # Env stores R_H(r) = -J_S(r). The controller minimizes J_S(r).
        return -slow_reward, slow_components

    def _run_forecast_round(
        self,
        env: Env,
        fast_agent: FastPPOAgent,
        action: Mapping[str, np.ndarray],
        scenario_idx: int,
    ) -> Tuple[float, Dict[str, Any]]:
        trial = copy.deepcopy(env)
        trial.rng = np.random.default_rng(
            self._forecast_seed(env, scenario_idx)
        )

        trial.apply_slow_action(self._copy_action(action))
        fast_obs = trial.get_fast_obs()
        boundary_info: Dict[str, Any] | None = None
        outage_slots = 0

        deterministic = bool(
            getattr(
                self.dpp_cfg,
                "forecast_fast_deterministic",
                True,
            )
        )
        device = getattr(
            fast_agent,
            "device",
            torch.device("cpu"),
        )
        cuda_devices = []
        if (
            torch.cuda.is_available()
            and getattr(device, "type", "cpu") == "cuda"
        ):
            cuda_devices = [
                (
                    torch.cuda.current_device()
                    if device.index is None
                    else int(device.index)
                )
            ]

        forecast_seed = self._forecast_seed(
            env,
            scenario_idx,
        )
        # A scenario uses common environment and Fast-policy randomness for
        # every candidate. fork_rng also prevents forecast rollouts from
        # consuming the actual joint-training RNG stream.
        with torch.random.fork_rng(
            devices=cuda_devices,
            enabled=True,
        ):
            torch.manual_seed(forecast_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(
                    forecast_seed
                )

            for _ in range(int(trial.slow_T)):
                selected = fast_agent.select_action(
                    fast_obs,
                    deterministic=deterministic,
                    update_norm=False,
                )
                (
                    next_obs,
                    _,
                    terminated,
                    truncated,
                    info,
                ) = split_env_step(
                    trial.step(selected["env_action"])
                )
                outage_slots += int(
                    np.asarray(
                        info.get("outage", []),
                        dtype=np.int32,
                    ).sum()
                )
                fast_obs = next_obs

                if bool(
                    info.get("is_round_boundary", False)
                ):
                    boundary_info = info
                    break
                if terminated or truncated:
                    raise RuntimeError(
                        "Forecast episode ended before a complete "
                        "slow round."
                    )

        if boundary_info is None:
            raise RuntimeError(
                "Forecast loop did not reach the slow round boundary."
            )

        cost, slow_components = self._extract_boundary_cost(boundary_info)

        # Battery queue pressure and rule-based charging remain in J_S.
        # A predicted physical outage is an infeasible candidate because the
        # scenario requires no depletion within the selected round.
        if outage_slots > 0:
            cost = math.inf

        return float(cost), {
            "scenario": int(scenario_idx),
            "round_dpp_cost": float(cost),
            "outage_slots": int(outage_slots),
            "round_fast_reward_sum": float(
                slow_components.get("round_fast_reward_sum", 0.0)
            ),
            "hire_cost_raw": float(
                slow_components.get("hire_cost_raw", 0.0)
            ),
            "hire_weight": float(
                slow_components.get("hire_weight", 1.0)
            ),
        }

    def _score_action(
        self,
        env: Env,
        fast_agent: FastPPOAgent,
        action: Mapping[str, np.ndarray],
    ) -> Tuple[float, Dict[str, Any]]:
        self._validate_action(env, action)
        self._candidate_requests += 1

        key = self._action_key(action)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached

        scenario_costs = []
        scenario_info = []
        for scenario_idx in range(int(self.dpp_cfg.forecast_scenarios)):
            cost, info = self._run_forecast_round(
                env=env,
                fast_agent=fast_agent,
                action=action,
                scenario_idx=scenario_idx,
            )
            scenario_costs.append(float(cost))
            scenario_info.append(info)

        if any(not math.isfinite(value) for value in scenario_costs):
            mean_cost = math.inf
            std_cost = math.inf
        else:
            mean_cost = float(np.mean(scenario_costs))
            std_cost = float(np.std(scenario_costs))
            self._finite_candidate_requests += 1

        result = (
            float(mean_cost),
            {
                "mean_round_dpp_cost": float(mean_cost),
                "std_round_dpp_cost": float(std_cost),
                "scenario_costs": tuple(float(x) for x in scenario_costs),
                "scenarios": scenario_info,
            },
        )
        self._score_cache[key] = result
        return result

    def _tie_break_key(
        self,
        action: Mapping[str, np.ndarray],
    ) -> Tuple[Any, ...]:
        rsu = np.asarray(action["rsu_scheduling"], dtype=np.int32)
        hiring = np.asarray(action["uav_hiring"], dtype=np.int32)
        uav = np.asarray(action["uav_scheduling"], dtype=np.int32)

        # Near-equal DPP costs follow a deterministic structural priority:
        # fewer hired UAVs, more RSU candidates, fewer UAV candidates.
        return (
            int(hiring.sum()),
            -int(rsu.sum()),
            int(uav.sum()),
            self._action_key(action),
        )

    def _candidate_is_better(
        self,
        candidate_score: float,
        candidate: Mapping[str, np.ndarray],
        incumbent_score: float,
        incumbent: Mapping[str, np.ndarray],
    ) -> bool:
        if not math.isfinite(candidate_score):
            return False
        if not math.isfinite(incumbent_score):
            return True

        atol = float(getattr(self.dpp_cfg, "dpp_tie_atol", 1e-3))
        rtol = float(getattr(self.dpp_cfg, "dpp_tie_rtol", 1e-9))
        tolerance = atol + rtol * max(
            abs(float(candidate_score)),
            abs(float(incumbent_score)),
        )

        if float(candidate_score) < float(incumbent_score) - tolerance:
            return True
        if abs(float(candidate_score) - float(incumbent_score)) <= tolerance:
            return self._tie_break_key(candidate) < self._tie_break_key(
                incumbent
            )
        return False

    @staticmethod
    def _same_action(
        lhs: Mapping[str, np.ndarray],
        rhs: Mapping[str, np.ndarray],
    ) -> bool:
        return all(
            np.array_equal(lhs[name], rhs[name])
            for name in (
                "rsu_scheduling",
                "uav_hiring",
                "uav_scheduling",
            )
        )

    def _build_action_info(
        self,
        env: Env,
        action: Mapping[str, np.ndarray],
        score: float,
        score_std: float,
        sweeps_completed: int,
    ) -> Dict[str, Any]:
        rsu = np.asarray(action["rsu_scheduling"], dtype=np.int32)
        hiring = np.asarray(action["uav_hiring"], dtype=np.int32)
        uav = np.asarray(action["uav_scheduling"], dtype=np.int32)
        provider_count = rsu.sum(axis=0) + uav.sum(axis=0)

        action_dim = int(rsu.size + hiring.size + uav.size)
        active_dims = int(rsu.sum() + hiring.sum() + uav.sum())

        return {
            "controller": "round_dpp_coordinate_descent",
            "predicted_round_dpp_cost": float(score),
            "predicted_round_dpp_cost_std": float(score_std),
            "coordinate_sweeps_completed": int(sweeps_completed),
            "coordinate_converged": 1,
            "candidate_requests": int(self._candidate_requests),
            "unique_candidates": int(len(self._score_cache)),
            "finite_candidate_requests": int(
                self._finite_candidate_requests
            ),
            "forecast_scenarios": int(self.dpp_cfg.forecast_scenarios),
            "raw_rsu_links": int(rsu.sum()),
            "effective_rsu_links": int(rsu.sum()),
            "raw_hired_uav": int(hiring.sum()),
            "effective_hired_uav": int(hiring.sum()),
            "raw_uav_links": int(uav.sum()),
            "effective_uav_links": int(uav.sum()),
            "num_scheduled_users": int(np.sum(provider_count > 0)),
            "num_residual_users": int(np.sum(provider_count == 0)),
            "active_action_dims": int(active_dims),
            "active_action_ratio": float(active_dims / max(action_dim, 1)),
            "projection_count": 0,
            "round_idx": int(env.t) // max(1, int(env.slow_T)),
        }

    def select_action(
        self,
        env: Env,
        fast_agent: FastPPOAgent,
    ) -> Dict[str, Any]:
        """Select the slow action by direct predicted round-DPP minimization."""
        if int(env.round_slot) != 0:
            raise RuntimeError(
                "Slow DPP decision must be made only at a round boundary: "
                f"round_slot={env.round_slot}."
            )

        self._score_cache = {}
        self._candidate_requests = 0
        self._finite_candidate_requests = 0

        current = self._initial_rsu_first_action(env)
        current_score, _ = self._score_action(env, fast_agent, current)
        if not math.isfinite(current_score):
            raise RuntimeError(
                "The RSU-first initial slow action is not DPP-feasible."
            )

        sweeps_completed = 0
        coordinate_converged = False
        for _ in range(int(self.dpp_cfg.max_coordinate_sweeps)):
            changed_in_sweep = False

            for region_idx in range(self.num_rsu):
                best_action = current
                best_score = current_score

                for candidate in self._iter_region_candidates(
                    env=env,
                    base_action=current,
                    region_idx=region_idx,
                ):
                    score, _ = self._score_action(
                        env=env,
                        fast_agent=fast_agent,
                        action=candidate,
                    )
                    if self._candidate_is_better(
                        candidate_score=score,
                        candidate=candidate,
                        incumbent_score=best_score,
                        incumbent=best_action,
                    ):
                        best_action = candidate
                        best_score = score

                if not self._same_action(current, best_action):
                    changed_in_sweep = True
                    current = self._copy_action(best_action)
                    current_score = float(best_score)

            sweeps_completed += 1
            if not changed_in_sweep:
                coordinate_converged = True
                break

        if not coordinate_converged:
            raise RuntimeError(
                "Slow DPP coordinate minimization did not converge within "
                f"{self.dpp_cfg.max_coordinate_sweeps} sweeps."
            )

        self._validate_action(env, current)
        if not math.isfinite(current_score):
            raise RuntimeError("Selected slow DPP cost is not finite.")

        _, current_score_info = self._score_action(
            env=env,
            fast_agent=fast_agent,
            action=current,
        )
        current_score_std = float(
            current_score_info["std_round_dpp_cost"]
        )
        action_info = self._build_action_info(
            env=env,
            action=current,
            score=current_score,
            score_std=current_score_std,
            sweeps_completed=sweeps_completed,
        )

        return {
            "env_action": self._copy_action(current),
            "action_info": action_info,
            "predicted_round_dpp_cost": float(current_score),
            "predicted_round_dpp_cost_std": float(current_score_std),
        }
