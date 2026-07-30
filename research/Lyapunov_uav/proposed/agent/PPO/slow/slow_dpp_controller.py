from __future__ import annotations

import atexit
import copy
import math
import multiprocessing as mp
import os
import time
from itertools import product
from multiprocessing.connection import Connection
from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch

from agent.PPO.common import split_env_step
from agent.PPO.fast.fast_agent import FastPPOAgent
from env.env import Env

from .slow_config import SlowDPPConfig


SlowEnvAction = Dict[str, np.ndarray]


def _slim_forecast_step_info(
    info: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Keep only information consumed by the Slow-DPP forecast.

    Non-boundary slot ``info`` dictionaries contain many diagnostic arrays.
    Sending all of them through a multiprocessing pipe every slot would add
    substantial serialization overhead. The full boundary record is retained
    because ``_extract_boundary_cost()`` needs its reward components.
    """
    if bool(info.get("is_round_boundary", False)):
        return copy.deepcopy(dict(info))

    return {
        "is_round_boundary": False,
        "outage": np.asarray(
            info.get("outage", []),
            dtype=np.int32,
        ).copy(),
    }


def _forecast_env_worker_main(
    connection: Connection,
    cpu_id: Optional[int],
) -> None:
    """
    Own candidate Env instances inside one spawned process.

    A candidate environment must remain in the same process for all 3,600
    slots. A per-slot ProcessPoolExecutor call would mutate only a temporary
    unpickled copy and break the Markov trajectory. This persistent command
    loop keeps the state local and exchanges only actions/observations.
    """
    base_env: Optional[Env] = None
    trials: Dict[int, Env] = {}

    try:
        if (
            cpu_id is not None
            and hasattr(os, "sched_setaffinity")
        ):
            os.sched_setaffinity(
                0,
                {int(cpu_id)},
            )

        while True:
            command, payload = connection.recv()

            if command == "set_base_env":
                base_env = payload
                trials = {}
                connection.send(("ok", None))
                continue

            if command == "load_trials":
                if base_env is None:
                    raise RuntimeError(
                        "Forecast worker received trials before base Env."
                    )

                indexed_actions, forecast_seed = payload
                trials = {}
                initial_observations = []

                for trial_idx, slow_action in indexed_actions:
                    trial = copy.deepcopy(base_env)
                    trial.rng = np.random.default_rng(
                        int(forecast_seed)
                    )
                    trial.apply_slow_action(
                        {
                            name: np.asarray(
                                slow_action[name],
                                dtype=np.int32,
                            ).copy()
                            for name in (
                                "rsu_scheduling",
                                "uav_hiring",
                                "uav_scheduling",
                            )
                        }
                    )
                    trial_idx = int(trial_idx)
                    trials[trial_idx] = trial
                    initial_observations.append(
                        (
                            trial_idx,
                            trial.get_fast_obs(),
                        )
                    )

                connection.send(
                    ("ok", initial_observations)
                )
                continue

            if command == "step":
                indexed_actions = payload
                step_results = []
                worker_started_at = time.perf_counter()

                for trial_idx, env_action in indexed_actions:
                    trial_idx = int(trial_idx)
                    trial = trials.get(trial_idx)
                    if trial is None:
                        raise RuntimeError(
                            "Forecast worker does not own trial "
                            f"{trial_idx}."
                        )

                    (
                        next_obs,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = split_env_step(
                        trial.step(dict(env_action))
                    )
                    step_results.append(
                        (
                            trial_idx,
                            next_obs,
                            float(reward),
                            bool(terminated),
                            bool(truncated),
                            _slim_forecast_step_info(info),
                        )
                    )

                connection.send(
                    (
                        "ok",
                        {
                            "results": step_results,
                            "worker_compute_seconds": (
                                time.perf_counter()
                                - worker_started_at
                            ),
                        },
                    )
                )
                continue

            if command == "clear":
                trials = {}
                connection.send(("ok", None))
                continue

            if command == "close":
                connection.send(("ok", None))
                return

            raise RuntimeError(
                f"Unknown forecast worker command: {command}"
            )

    except EOFError:
        return
    except BaseException as exc:
        try:
            connection.send(
                (
                    "error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            )
        except BaseException:
            pass
        raise
    finally:
        connection.close()


class _ForecastEnvProcessPool:
    """
    Persistent spawn-based process pool for candidate environment steps.

    CUDA policy inference stays in the parent process. Workers never touch
    CUDA; they only advance independent CPU environment states.
    """

    def __init__(
        self,
        worker_count: int,
        response_timeout_seconds: float = 300.0,
    ) -> None:
        self.worker_count = max(1, int(worker_count))
        self.response_timeout_seconds = float(
            response_timeout_seconds
        )
        self._context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[Any] = []
        self._trial_indices_by_worker: list[list[int]] = [
            []
            for _ in range(self.worker_count)
        ]
        self._closed = False
        if hasattr(os, "sched_getaffinity"):
            available_cpus = sorted(
                int(value)
                for value in os.sched_getaffinity(0)
            )
        else:
            available_cpus = list(
                range(os.cpu_count() or self.worker_count)
            )
        if len(available_cpus) < self.worker_count:
            raise RuntimeError(
                "forecast_env_workers exceeds the CPUs available to the "
                "Slurm task: "
                f"workers={self.worker_count}, "
                f"available_cpus={available_cpus}."
            )
        self.available_cpus = tuple(available_cpus)
        self.last_step_roundtrip_seconds = 0.0
        self.last_step_worker_compute_sum_seconds = 0.0
        self.last_step_worker_compute_max_seconds = 0.0
        self.last_step_overhead_seconds = 0.0

        for worker_idx in range(self.worker_count):
            parent_connection, child_connection = (
                self._context.Pipe(duplex=True)
            )
            process = self._context.Process(
                target=_forecast_env_worker_main,
                args=(
                    child_connection,
                    self.available_cpus[worker_idx],
                ),
                name=f"slow-dpp-env-{worker_idx}",
                daemon=True,
            )
            process.start()
            child_connection.close()
            self._connections.append(parent_connection)
            self._processes.append(process)

    def _receive(
        self,
        worker_idx: int,
    ) -> Any:
        connection = self._connections[worker_idx]
        process = self._processes[worker_idx]

        if not connection.poll(
            self.response_timeout_seconds
        ):
            raise TimeoutError(
                "Timed out waiting for forecast environment worker "
                f"{worker_idx}; alive={process.is_alive()}, "
                f"exitcode={process.exitcode}."
            )

        status, payload = connection.recv()
        if status == "ok":
            return payload
        if status == "error":
            raise RuntimeError(
                "Forecast environment worker failed: "
                f"worker={worker_idx}, "
                f"type={payload.get('type')}, "
                f"message={payload.get('message')}"
            )
        raise RuntimeError(
            "Forecast worker returned an invalid response: "
            f"worker={worker_idx}, status={status}"
        )

    def set_base_env(self, env: Env) -> None:
        if self._closed:
            raise RuntimeError(
                "Forecast process pool is already closed."
            )

        for connection in self._connections:
            connection.send(
                ("set_base_env", env)
            )
        for worker_idx in range(self.worker_count):
            self._receive(worker_idx)

    def load_trials(
        self,
        actions: Sequence[Mapping[str, np.ndarray]],
        forecast_seed: int,
    ) -> list[Dict[str, np.ndarray]]:
        if not actions:
            return []

        active_workers = min(
            self.worker_count,
            len(actions),
        )
        self._trial_indices_by_worker = [
            []
            for _ in range(self.worker_count)
        ]
        indexed_actions_by_worker: list[list[Any]] = [
            []
            for _ in range(active_workers)
        ]

        for trial_idx, action in enumerate(actions):
            worker_idx = trial_idx % active_workers
            self._trial_indices_by_worker[
                worker_idx
            ].append(trial_idx)
            indexed_actions_by_worker[
                worker_idx
            ].append(
                (
                    trial_idx,
                    {
                        name: np.asarray(
                            action[name],
                            dtype=np.int32,
                        ).copy()
                        for name in (
                            "rsu_scheduling",
                            "uav_hiring",
                            "uav_scheduling",
                        )
                    },
                )
            )

        for worker_idx in range(active_workers):
            self._connections[worker_idx].send(
                (
                    "load_trials",
                    (
                        indexed_actions_by_worker[worker_idx],
                        int(forecast_seed),
                    ),
                )
            )

        observations: list[
            Optional[Dict[str, np.ndarray]]
        ] = [
            None
            for _ in actions
        ]
        for worker_idx in range(active_workers):
            for trial_idx, observation in self._receive(
                worker_idx
            ):
                observations[int(trial_idx)] = observation

        if any(value is None for value in observations):
            raise RuntimeError(
                "Forecast workers did not return every initial "
                "observation."
            )
        return [
            value
            for value in observations
            if value is not None
        ]

    def step(
        self,
        env_actions: Sequence[Mapping[str, Any]],
    ) -> list[
        Tuple[
            Dict[str, np.ndarray],
            float,
            bool,
            bool,
            Dict[str, Any],
        ]
    ]:
        if not env_actions:
            return []

        active_workers = min(
            self.worker_count,
            len(env_actions),
        )
        step_started_at = time.perf_counter()
        for worker_idx in range(active_workers):
            indexed_actions = [
                (
                    trial_idx,
                    env_actions[trial_idx],
                )
                for trial_idx in self._trial_indices_by_worker[
                    worker_idx
                ]
            ]
            self._connections[worker_idx].send(
                ("step", indexed_actions)
            )

        ordered_results: list[Optional[Any]] = [
            None
            for _ in env_actions
        ]
        worker_compute_seconds = []
        for worker_idx in range(active_workers):
            response = self._receive(worker_idx)
            worker_compute_seconds.append(
                float(
                    response["worker_compute_seconds"]
                )
            )
            for (
                trial_idx,
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) in response["results"]:
                ordered_results[int(trial_idx)] = (
                    next_obs,
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                    info,
                )

        if any(value is None for value in ordered_results):
            raise RuntimeError(
                "Forecast workers did not return every step result."
            )
        roundtrip_seconds = (
            time.perf_counter() - step_started_at
        )
        worker_compute_sum = float(
            sum(worker_compute_seconds)
        )
        worker_compute_max = float(
            max(worker_compute_seconds, default=0.0)
        )
        self.last_step_roundtrip_seconds = float(
            roundtrip_seconds
        )
        self.last_step_worker_compute_sum_seconds = (
            worker_compute_sum
        )
        self.last_step_worker_compute_max_seconds = (
            worker_compute_max
        )
        self.last_step_overhead_seconds = max(
            0.0,
            float(roundtrip_seconds) - worker_compute_max,
        )
        return [
            value
            for value in ordered_results
            if value is not None
        ]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for connection, process in zip(
            self._connections,
            self._processes,
        ):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except BaseException:
                    pass

        for worker_idx, (connection, process) in enumerate(
            zip(
                self._connections,
                self._processes,
            )
        ):
            if process.is_alive():
                try:
                    self._receive(worker_idx)
                except BaseException:
                    pass
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            connection.close()


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
        self._forecast_batches = 0
        self._forecast_trial_steps = 0
        self._forecast_wall_seconds = 0.0
        self._forecast_policy_seconds = 0.0
        self._forecast_env_seconds = 0.0
        self._forecast_process_setup_seconds = 0.0
        self._forecast_process_roundtrip_seconds = 0.0
        self._forecast_worker_compute_sum_seconds = 0.0
        self._forecast_worker_critical_seconds = 0.0
        self._forecast_process_overhead_seconds = 0.0
        self._forecast_batch_size_sum = 0
        self._forecast_peak_batch_size = 0
        self._forecast_active_workers_sum = 0
        self._forecast_peak_active_workers = 0
        self._forecast_process_pool: Optional[
            _ForecastEnvProcessPool
        ] = None
        atexit.register(self.close)

    def close(self) -> None:
        if self._forecast_process_pool is not None:
            self._forecast_process_pool.close()
            self._forecast_process_pool = None

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

    def _configured_forecast_batch_size(self) -> int:
        return max(
            1,
            int(
                getattr(
                    self.dpp_cfg,
                    "forecast_candidate_batch_size",
                    1,
                )
            ),
        )

    def _configured_forecast_env_workers(self) -> int:
        return max(
            1,
            int(
                getattr(
                    self.dpp_cfg,
                    "forecast_env_workers",
                    1,
                )
            ),
        )

    @staticmethod
    def _step_forecast_trial(
        payload: Tuple[Env, Mapping[str, Any]],
    ) -> Tuple[
        Dict[str, np.ndarray],
        float,
        bool,
        bool,
        Dict[str, Any],
    ]:
        trial, env_action = payload
        return split_env_step(trial.step(dict(env_action)))

    def _run_forecast_round_batch(
        self,
        env: Env,
        fast_agent: FastPPOAgent,
        actions: Sequence[Mapping[str, np.ndarray]],
        scenario_idx: int,
    ) -> list[Tuple[float, Dict[str, Any]]]:
        """
        Evaluate independent Slow candidates in lock-step.

        Candidate enumeration, one-full-round horizon, DPP extraction,
        feasibility handling, and common-random-number semantics are
        unchanged. Only the actor forward and independent environment steps
        are scheduled in batches.
        """
        if not actions:
            return []

        started_at = time.perf_counter()
        forecast_seed = self._forecast_seed(
            env,
            scenario_idx,
        )
        candidate_count = len(actions)
        active_workers = min(
            self._configured_forecast_env_workers(),
            candidate_count,
        )
        self._forecast_batch_size_sum += candidate_count
        self._forecast_peak_batch_size = max(
            self._forecast_peak_batch_size,
            candidate_count,
        )
        self._forecast_active_workers_sum += active_workers
        self._forecast_peak_active_workers = max(
            self._forecast_peak_active_workers,
            active_workers,
        )

        use_process_workers = bool(
            self._forecast_process_pool is not None
            and candidate_count > 1
        )

        trials: list[Env] = []
        if use_process_workers:
            setup_started_at = time.perf_counter()
            fast_observations = (
                self._forecast_process_pool.load_trials(
                    actions=actions,
                    forecast_seed=forecast_seed,
                )
            )
            self._forecast_process_setup_seconds += (
                time.perf_counter() - setup_started_at
            )
        else:
            fast_observations = []
            for action in actions:
                trial = copy.deepcopy(env)
                trial.rng = np.random.default_rng(
                    forecast_seed
                )
                trial.apply_slow_action(
                    self._copy_action(action)
                )
                trials.append(trial)
                fast_observations.append(
                    trial.get_fast_obs()
                )

        boundary_infos: list[Optional[Dict[str, Any]]] = [
            None
            for _ in actions
        ]
        outage_slots = np.zeros(
            candidate_count,
            dtype=np.int64,
        )
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

        # fork_rng keeps forecast sampling separate from the realized Fast-PPO
        # trajectory. The batch sampler shares each underlying random
        # variate across candidates while retaining candidate-specific policy
        # probabilities.
        with torch.random.fork_rng(
            devices=cuda_devices,
            enabled=True,
        ):
            torch.manual_seed(forecast_seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(forecast_seed)

            for _ in range(int(env.slow_T)):
                policy_started_at = time.perf_counter()
                batch_selector = getattr(
                    fast_agent,
                    "select_env_actions_batch",
                    None,
                )
                if callable(batch_selector):
                    env_actions = batch_selector(
                        observations=fast_observations,
                        deterministic=deterministic,
                        update_norm=False,
                        common_random_across_batch=True,
                    )
                else:
                    if (
                        not deterministic
                        and candidate_count > 1
                    ):
                        raise RuntimeError(
                            "Stochastic batched forecast requires "
                            "FastPPOAgent.select_env_actions_batch()."
                        )
                    env_actions = [
                        fast_agent.select_action(
                            obs,
                            deterministic=deterministic,
                            update_norm=False,
                        )["env_action"]
                        for obs in fast_observations
                    ]
                self._forecast_policy_seconds += (
                    time.perf_counter() - policy_started_at
                )

                env_started_at = time.perf_counter()
                if use_process_workers:
                    step_results = (
                        self._forecast_process_pool.step(
                            env_actions
                        )
                    )
                    self._forecast_process_roundtrip_seconds += (
                        self._forecast_process_pool
                        .last_step_roundtrip_seconds
                    )
                    self._forecast_worker_compute_sum_seconds += (
                        self._forecast_process_pool
                        .last_step_worker_compute_sum_seconds
                    )
                    self._forecast_worker_critical_seconds += (
                        self._forecast_process_pool
                        .last_step_worker_compute_max_seconds
                    )
                    self._forecast_process_overhead_seconds += (
                        self._forecast_process_pool
                        .last_step_overhead_seconds
                    )
                else:
                    payloads = list(
                        zip(trials, env_actions)
                    )
                    step_results = [
                        self._step_forecast_trial(payload)
                        for payload in payloads
                    ]
                self._forecast_env_seconds += (
                    time.perf_counter() - env_started_at
                )
                self._forecast_trial_steps += len(step_results)

                reached_boundary = False
                for trial_idx, (
                    next_obs,
                    _,
                    terminated,
                    truncated,
                    info,
                ) in enumerate(step_results):
                    outage_slots[trial_idx] += int(
                        np.asarray(
                            info.get("outage", []),
                            dtype=np.int32,
                        ).sum()
                    )
                    fast_observations[trial_idx] = next_obs

                    if bool(
                        info.get("is_round_boundary", False)
                    ):
                        boundary_infos[trial_idx] = dict(info)
                        reached_boundary = True
                    elif terminated or truncated:
                        raise RuntimeError(
                            "Forecast episode ended before a complete "
                            "slow round."
                        )

                if reached_boundary:
                    if any(
                        value is None
                        for value in boundary_infos
                    ):
                        raise RuntimeError(
                            "Batched forecast candidates reached different "
                            "round boundaries."
                        )
                    break

        if any(value is None for value in boundary_infos):
            raise RuntimeError(
                "Forecast loop did not reach the slow round boundary."
            )

        results: list[Tuple[float, Dict[str, Any]]] = []
        for trial_idx, boundary_info in enumerate(boundary_infos):
            if boundary_info is None:
                raise RuntimeError(
                    "Internal error: missing forecast boundary info."
                )
            cost, slow_components = self._extract_boundary_cost(
                boundary_info
            )

            # Battery queue pressure and rule-based charging remain in J_S.
            # A predicted physical outage makes the candidate infeasible.
            if int(outage_slots[trial_idx]) > 0:
                cost = math.inf

            results.append(
                (
                    float(cost),
                    {
                        "scenario": int(scenario_idx),
                        "round_dpp_cost": float(cost),
                        "outage_slots": int(
                            outage_slots[trial_idx]
                        ),
                        "round_fast_reward_sum": float(
                            slow_components.get(
                                "round_fast_reward_sum",
                                0.0,
                            )
                        ),
                        "hire_cost_raw": float(
                            slow_components.get(
                                "hire_cost_raw",
                                0.0,
                            )
                        ),
                        "hire_weight": float(
                            slow_components.get(
                                "hire_weight",
                                1.0,
                            )
                        ),
                    },
                )
            )

        self._forecast_batches += 1
        self._forecast_wall_seconds += (
            time.perf_counter() - started_at
        )
        return results

    def _run_forecast_round(
        self,
        env: Env,
        fast_agent: FastPPOAgent,
        action: Mapping[str, np.ndarray],
        scenario_idx: int,
    ) -> Tuple[float, Dict[str, Any]]:
        return self._run_forecast_round_batch(
            env=env,
            fast_agent=fast_agent,
            actions=[action],
            scenario_idx=scenario_idx,
        )[0]

    def _score_actions(
        self,
        env: Env,
        fast_agent: FastPPOAgent,
        actions: Sequence[Mapping[str, np.ndarray]],
    ) -> list[Tuple[float, Dict[str, Any]]]:
        if not actions:
            return []

        results: list[Optional[Tuple[float, Dict[str, Any]]]] = [
            None
            for _ in actions
        ]
        missing_actions: list[Mapping[str, np.ndarray]] = []
        missing_keys: list[bytes] = []
        missing_key_set: set[bytes] = set()

        for action_idx, action in enumerate(actions):
            self._validate_action(env, action)
            self._candidate_requests += 1
            key = self._action_key(action)
            cached = self._score_cache.get(key)
            if cached is not None:
                results[action_idx] = cached
            elif key not in missing_key_set:
                missing_key_set.add(key)
                missing_keys.append(key)
                missing_actions.append(action)

        scenario_costs: Dict[bytes, list[float]] = {
            key: []
            for key in missing_keys
        }
        scenario_infos: Dict[bytes, list[Dict[str, Any]]] = {
            key: []
            for key in missing_keys
        }
        batch_size = self._configured_forecast_batch_size()
        for scenario_idx in range(
            int(self.dpp_cfg.forecast_scenarios)
        ):
            for batch_start in range(
                0,
                len(missing_actions),
                batch_size,
            ):
                batch_actions = missing_actions[
                    batch_start:batch_start + batch_size
                ]
                batch_keys = missing_keys[
                    batch_start:batch_start + batch_size
                ]
                batch_results = self._run_forecast_round_batch(
                    env=env,
                    fast_agent=fast_agent,
                    actions=batch_actions,
                    scenario_idx=scenario_idx,
                )
                for key, (cost, info) in zip(
                    batch_keys,
                    batch_results,
                ):
                    scenario_costs[key].append(float(cost))
                    scenario_infos[key].append(info)

        for key in missing_keys:
            costs = scenario_costs[key]
            if any(not math.isfinite(value) for value in costs):
                mean_cost = math.inf
                std_cost = math.inf
            else:
                mean_cost = float(np.mean(costs))
                std_cost = float(np.std(costs))
                self._finite_candidate_requests += 1

            self._score_cache[key] = (
                float(mean_cost),
                {
                    "mean_round_dpp_cost": float(mean_cost),
                    "std_round_dpp_cost": float(std_cost),
                    "scenario_costs": tuple(
                        float(value)
                        for value in costs
                    ),
                    "scenarios": scenario_infos[key],
                },
            )

        finalized: list[Tuple[float, Dict[str, Any]]] = []
        for action_idx, action in enumerate(actions):
            result = results[action_idx]
            if result is None:
                result = self._score_cache[
                    self._action_key(action)
                ]
            finalized.append(result)
        return finalized

    def _score_action(
        self,
        env: Env,
        fast_agent: FastPPOAgent,
        action: Mapping[str, np.ndarray],
    ) -> Tuple[float, Dict[str, Any]]:
        return self._score_actions(
            env=env,
            fast_agent=fast_agent,
            actions=[action],
        )[0]

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
        throughput = (
            float(self._forecast_trial_steps)
            / max(float(self._forecast_wall_seconds), 1e-12)
        )
        mean_batch_size = (
            float(self._forecast_batch_size_sum)
            / max(int(self._forecast_batches), 1)
        )
        mean_active_workers = (
            float(self._forecast_active_workers_sum)
            / max(int(self._forecast_batches), 1)
        )

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
            "forecast_candidate_batch_size": int(
                self._configured_forecast_batch_size()
            ),
            "forecast_env_workers": int(
                self._configured_forecast_env_workers()
            ),
            "forecast_available_cpus": int(
                len(
                    self._forecast_process_pool.available_cpus
                )
                if self._forecast_process_pool is not None
                else 1
            ),
            "forecast_batches": int(self._forecast_batches),
            "forecast_trial_steps": int(
                self._forecast_trial_steps
            ),
            "forecast_wall_seconds": float(
                self._forecast_wall_seconds
            ),
            "forecast_policy_seconds": float(
                self._forecast_policy_seconds
            ),
            "forecast_env_seconds": float(
                self._forecast_env_seconds
            ),
            "forecast_process_setup_seconds": float(
                self._forecast_process_setup_seconds
            ),
            "forecast_process_roundtrip_seconds": float(
                self._forecast_process_roundtrip_seconds
            ),
            "forecast_worker_compute_sum_seconds": float(
                self._forecast_worker_compute_sum_seconds
            ),
            "forecast_worker_critical_seconds": float(
                self._forecast_worker_critical_seconds
            ),
            "forecast_process_overhead_seconds": float(
                self._forecast_process_overhead_seconds
            ),
            "forecast_mean_batch_size": float(
                mean_batch_size
            ),
            "forecast_peak_batch_size": int(
                self._forecast_peak_batch_size
            ),
            "forecast_mean_active_workers": float(
                mean_active_workers
            ),
            "forecast_peak_active_workers": int(
                self._forecast_peak_active_workers
            ),
            "forecast_env_backend": (
                "spawn_process"
                if self._configured_forecast_env_workers() > 1
                else "serial"
            ),
            "forecast_trial_steps_per_second": float(
                throughput
            ),
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
        self._forecast_batches = 0
        self._forecast_trial_steps = 0
        self._forecast_wall_seconds = 0.0
        self._forecast_policy_seconds = 0.0
        self._forecast_env_seconds = 0.0
        self._forecast_process_setup_seconds = 0.0
        self._forecast_process_roundtrip_seconds = 0.0
        self._forecast_worker_compute_sum_seconds = 0.0
        self._forecast_worker_critical_seconds = 0.0
        self._forecast_process_overhead_seconds = 0.0
        self._forecast_batch_size_sum = 0
        self._forecast_peak_batch_size = 0
        self._forecast_active_workers_sum = 0
        self._forecast_peak_active_workers = 0

        workers = self._configured_forecast_env_workers()
        if workers > 1:
            if self._forecast_process_pool is None:
                self._forecast_process_pool = (
                    _ForecastEnvProcessPool(
                        worker_count=workers,
                    )
                )
            elif (
                self._forecast_process_pool.worker_count
                != workers
            ):
                self.close()
                self._forecast_process_pool = (
                    _ForecastEnvProcessPool(
                        worker_count=workers,
                    )
                )
            self._forecast_process_pool.set_base_env(env)
        else:
            self.close()

        try:
            current = self._initial_rsu_first_action(env)
            current_score, _ = self._score_action(
                env,
                fast_agent,
                current,
            )
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
                    candidates = list(
                        self._iter_region_candidates(
                            env=env,
                            base_action=current,
                            region_idx=region_idx,
                        )
                    )
                    candidate_results = self._score_actions(
                        env=env,
                        fast_agent=fast_agent,
                        actions=candidates,
                    )
                    for candidate, (score, _) in zip(
                        candidates,
                        candidate_results,
                    ):
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
                    "Slow DPP coordinate minimization did not converge "
                    "within "
                    f"{self.dpp_cfg.max_coordinate_sweeps} sweeps."
                )

            self._validate_action(env, current)
            if not math.isfinite(current_score):
                raise RuntimeError(
                    "Selected slow DPP cost is not finite."
                )

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
                "predicted_round_dpp_cost_std": float(
                    current_score_std
                ),
            }
        finally:
            # The spawn workers remain alive across Slow decisions. Reusing
            # them avoids repeatedly importing Python/Torch and recreating
            # 16 processes for every round. ``close()`` is registered with
            # atexit and can also be called explicitly by tests/trainers.
            pass