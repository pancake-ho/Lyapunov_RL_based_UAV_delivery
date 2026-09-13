from __future__ import annotations

"""P3 physical transitions with scheduling-only slow proposals.

A non-learning completion selects hiring/hovering point before begin_frame.
Fast l/k/discrete-power semantics and state/reward formulas follow the supplied
Claude environment. region_scope is used only by isolated candidate rollouts.
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np

from agent.P3.exact_fast_controller import ExactFastController
from agent.P3.features import build_state_features
from config_hppo import HPPOConfig
from env.p3.battery import (
    activation_energy_required_j,
    apply_active_slot,
    apply_relocation,
    apply_unhired_slot,
    battery_power_cap_w,
    diagnose_return_to_charge,
    relocation_energy_j,
)
from env.p3.environment import (
    generate_frame_trace,
    update_playback_queue,
    validate_frame_trace,
)
from env.p3.radio import (
    capacity_bps,
    link_gain,
    required_uav_power_w,
    rsu_link_capacity_bps,
)
from env.p3.topology import (
    initialize_state,
    region_membership,
    validate_region_action,
    validate_state,
)
from env.p3.types import FrameTrace, P3State, RegionAction
from hppo.scheduling import SchedulingProposal, integer_action


PROVIDER_NONE, PROVIDER_RSU, PROVIDER_UAV = 0, 1, 2


def uav_link_capacity_bps(
    horizontal_distance_m: float,
    fading: float,
    power_w: float,
    cfg: HPPOConfig,
) -> tuple[float, float]:
    """Eq. (2.13)-(2.15) with the fixed per-user UAV group W^U/J^U.

    Returns ``(capacity_bps, channel_gain)``. This is the exact inverse of
    :func:`env.p3.radio.required_uav_power_w` (Eq. 7.3).
    """

    vertical = cfg.uav_height_m - cfg.user_height_m
    gain = link_gain(
        cfg.uav_beta0,
        math.hypot(horizontal_distance_m, vertical),
        cfg.uav_pathloss_exp,
        fading,
    )
    return capacity_bps(cfg.uav_user_bandwidth_hz, power_w, gain, cfg), gain


def episode_seed(cfg: HPPOConfig, episode: int) -> int:
    return int(cfg.seed + 1_009 * int(episode))


def realized_trace_seed(cfg: HPPOConfig, episode: int, frame: int) -> int:
    """Same structure as SlowRolloutController.realized_seed, plus episode."""

    return int(episode_seed(cfg, episode) + 20_000_033 + int(frame) * 100_003)


# ----------------------------------------------------------------------
# result containers
# ----------------------------------------------------------------------
@dataclass
class RegionSlotMetrics:
    dpp_slot_cost: float = 0.0          # J_F(t), Eq. (6.12)
    degradation: float = 0.0            # sum_n g^Q_n(t), Eq. (3.7)
    delivered_chunks: float = 0.0
    utility: float = 0.0
    stall_user_slots: int = 0
    served_user_slots: int = 0
    member_count: int = 0
    constraint_cost: float = 0.0        # mean_n [Z_n]^+ / Qe (lagrangian mode)
    stall_fraction: float = 0.0
    q_gt_qe_user_slots: int = 0
    total_uav_power_w: float = 0.0
    p_eff_w: float = 0.0
    reserve_violation: int = 0
    power_violation: int = 0


@dataclass
class SlotStepResult:
    frame_done: bool
    metrics: dict[int, RegionSlotMetrics]
    info: dict


@dataclass
class RegionFrameAccumulator:
    hired: int = 0
    hiring_cost: float = 0.0            # lambda_H * c^H * mu
    dpp_slot_sum: float = 0.0
    degradation_sum: float = 0.0
    delivered_sum: float = 0.0
    utility_sum: float = 0.0
    stall_user_slots: int = 0
    served_user_slots: int = 0
    user_slots: int = 0
    constraint_sum: float = 0.0
    stall_fraction_sum: float = 0.0
    q_gt_qe_user_slots: int = 0
    energy_consumed_j: float = 0.0
    energy_charged_j: float = 0.0
    reserve_violations: int = 0
    power_violations: int = 0
    projection_events: int = 0
    quality_hist: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def frame_dpp_cost(self) -> float:
        # identical to RegionFrameResult.frame_dpp_cost in env/p3/environment.py
        return float(self.dpp_slot_sum) + float(self._v) * float(self.hiring_cost)

    @property
    def original_cost(self) -> float:
        return float(self.degradation_sum + self.hiring_cost)

    _v: float = 0.0


# ----------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------
class P3HierarchicalEnv:
    """Episodic, per-region, two-timescale P3 environment.

    Protocol per episode::

        env.reset(episode)
        for r in range(cfg.num_frames):
            obs = env.prepare_frame()                    # dict region -> obs
            masks = {m: env.frame_action_masks(m)}
            actions, scores = completion.select_all(env, proposals, fast_policy)
            info = env.begin_frame(proposals, actions, scores)
            for t in range(cfg.frame_slots):
                sobs = {m: env.get_slot_obs(m)}
                smasks = {m: env.slot_action_masks(m)}
                step = env.step_slot({m: raw_slot_action_m})
            # step.frame_done is True after the last slot; frame summary is in
            # step.info['frame_summary']
    """

    def __init__(self, cfg: HPPOConfig, region_scope=None) -> None:
        self.cfg = cfg
        self.M = cfg.num_regions
        self.regions = tuple(range(self.M)) if region_scope is None else tuple(region_scope)
        if not self.regions or len(set(self.regions)) != len(self.regions) or any(m not in range(self.M) for m in self.regions):
            raise ValueError("invalid region scope")
        self.N = cfg.num_users
        self.T = cfg.frame_slots
        self.L = cfg.num_candidate_points
        self.K = cfg.num_quality_levels
        self.guard = ExactFastController(cfg)  # only max_queue_admissible_chunks is used
        self.state: P3State | None = None
        self.episode = 0
        self.frame = 0
        self.local_slot = 0
        self.global_slot = 0
        self.frame_prepared = False
        self.frame_active = False
        self.membership = np.zeros(self.N, dtype=np.int32)
        self.region_users: dict[int, tuple[int, ...]] = {}
        self.region_actions: dict[int, RegionAction] = {}
        self.provider = np.zeros(self.N, dtype=np.int64)
        self.trace: FrameTrace | None = None
        self.acc: dict[int, RegionFrameAccumulator] = {}
        self.episode_totals: dict = {}

    # ------------------------------------------------------------------
    # episode / frame lifecycle
    # ------------------------------------------------------------------
    def reset(self, episode: int = 0, seed: int | None = None) -> None:
        self.episode = int(episode)
        init_seed = episode_seed(self.cfg, self.episode) if seed is None else int(seed)
        # initialize_state draws from cfg.seed; re-seed through a config copy so
        # the P3 initial-state generator (positions/speeds/queues/battery) is reused.
        self.state = initialize_state(replace(self.cfg, seed=init_seed))
        validate_state(self.state, self.cfg)
        self.frame = 0
        self.local_slot = 0
        self.global_slot = 0
        self.frame_prepared = False
        self.frame_active = False
        self.region_actions = {}
        self.provider[:] = PROVIDER_NONE
        self.episode_totals = {
            "dpp_cost": 0.0,
            "original_cost": 0.0,
            "degradation": 0.0,
            "hiring_cost": 0.0,
            "delivered_chunks": 0.0,
            "utility": 0.0,
            "stall_user_slots": 0,
            "served_user_slots": 0,
            "user_slots": 0,
            "q_gt_qe_user_slots": 0,
            "hired_uav_frames": 0,
            "uav_frames": 0,
            "projection_events": 0,
            "reserve_violations": 0,
            "power_violations": 0,
            "constraint_sum": 0.0,
            "constraint_samples": 0,
            "stall_fraction_sum": 0.0,
            "energy_consumed_j": 0.0,
            "energy_charged_j": 0.0,
            "min_battery_soc": 1.0,
            "max_queue": float(np.max(self.state.queue)),
        }

    def prepare_frame(self) -> dict[int, np.ndarray]:
        if self.frame_active:
            raise RuntimeError("cannot prepare a new frame while a frame is active")
        assert self.state is not None
        self.membership = region_membership(self.state, self.cfg)
        self.region_users = {
            m: tuple(int(u) for u in np.flatnonzero(self.membership == m))
            for m in self.regions
        }
        # P3 common-random-number convention: the realized fading trace is a
        # deterministic function of (seed, episode, frame). The agents never see
        # it, so it is a hidden random disturbance from their point of view.
        self.trace = generate_frame_trace(self.cfg, realized_trace_seed(self.cfg, self.episode, self.frame))
        validate_frame_trace(self.trace, self.cfg)
        self.frame_prepared = True
        return {m: self.get_frame_obs(m) for m in self.regions}

    # ------------------------------------------------------------------
    # observations (no instantaneous CSI)
    # ------------------------------------------------------------------
    def get_frame_obs(self, region: int) -> np.ndarray:
        if not self.frame_prepared:
            raise RuntimeError("call prepare_frame() first")
        assert self.state is not None
        base = build_state_features(self.state, region, self.region_users[region], self.cfg)
        g = self.cfg.ppo_global_feature_dim  # 7 global features from agent/P3/features.py
        progress = np.asarray([self.frame / max(self.cfg.num_frames, 1)], dtype=np.float32)
        obs = np.concatenate([base[:g], progress, base[g:]]).astype(np.float32)
        if obs.shape != (self.cfg.frame_obs_dim,):
            raise RuntimeError(f"frame obs shape {obs.shape} != {(self.cfg.frame_obs_dim,)}")
        return obs

    def get_slot_obs(self, region: int) -> np.ndarray:
        if not self.frame_active:
            raise RuntimeError("no active frame")
        assert self.state is not None
        cfg = self.cfg
        action = self.region_actions[region]
        users = self.region_users[region]
        center = cfg.rsu_x(region)
        battery = float(self.state.battery_j[region])
        remaining = self.T - self.local_slot
        p_eff = battery_power_cap_w(battery, remaining, cfg) if action.hired else 0.0
        g = [
            region / max(self.M - 1, 1),
            float(action.hired),
            battery / cfg.battery_capacity_j,
            p_eff / cfg.uav_max_total_power_w,
            (float(self.state.uav_x[region]) - center) / cfg.region_length_m,
            self.local_slot / max(self.T - 1, 1),
            remaining / self.T,
            len(users) / self.N,
        ]
        feats = np.zeros((self.N, cfg.slot_obs_user_dim), dtype=np.float32)
        speed_span = max(cfg.vehicle_speed_max_mps - cfg.vehicle_speed_min_mps, 1e-9)
        uav_x = action.target_x(cfg)
        for u in users:
            q = float(self.state.queue[u])
            z = cfg.large_queue_level - q
            x = float(self.state.user_x[u])
            prov = int(self.provider[u])
            one_hot = [float(prov == PROVIDER_NONE), float(prov == PROVIDER_RSU), float(prov == PROVIDER_UAV)]
            feats[u] = np.asarray(
                [
                    1.0,
                    *one_hot,
                    np.clip(q / cfg.large_queue_level, 0.0, 2.0),
                    np.clip(z / cfg.large_queue_level, -1.0, 1.0),
                    np.clip((x - center) / cfg.region_length_m, -2.0, 2.0),
                    np.clip((float(self.state.user_speed[u]) - cfg.vehicle_speed_min_mps) / speed_span, 0.0, 1.0),
                    np.clip(abs(center - x) / cfg.region_length_m, 0.0, 2.0),
                    np.clip(abs(uav_x - x) / cfg.region_length_m, 0.0, 2.0) if action.hired else 0.0,
                    (float(self.state.last_quality_index[u]) + 1.0) / self.K,
                ],
                dtype=np.float32,
            )
        obs = np.concatenate([np.asarray(g, dtype=np.float32), feats.reshape(-1)])
        if obs.shape != (cfg.slot_obs_dim,):
            raise RuntimeError(f"slot obs shape {obs.shape} != {(cfg.slot_obs_dim,)}")
        return obs

    # ------------------------------------------------------------------
    # frame feasibility / masks
    # ------------------------------------------------------------------
    def feasible_hover_points(self, region: int) -> list[int]:
        """Points that satisfy reachability (2.4), relocation causality and activation (4.11)."""

        assert self.state is not None
        cfg = self.cfg
        previous_x = float(self.state.uav_x[region])
        battery = float(self.state.battery_j[region])
        feasible = []
        for idx, point_x in enumerate(cfg.candidate_points(region)):
            if abs(point_x - previous_x) > cfg.reachable_distance_m + 1e-9:
                continue
            after_move = battery - relocation_energy_j(previous_x, point_x, cfg)
            if after_move + 1e-9 < activation_energy_required_j(cfg):
                continue
            feasible.append(idx)
        return feasible

    def frame_action_masks(self, region: int) -> list[np.ndarray]:
        """Base support; the AR policy additionally masks prefix capacities."""
        possible = bool(self.feasible_hover_points(region))
        members = set(self.region_users[region])
        return [np.asarray([True, u in members, u in members and possible], dtype=bool)
                for u in range(self.N)]

    def decode_frame_action(self, raw: np.ndarray) -> dict:
        return {"assoc": integer_action(raw, self.cfg.frame_action_nvec, "frame action")}

    def proposal(self, region, raw):
        return SchedulingProposal.from_tokens(region, raw, self.region_users[region], self.cfg,
                                              bool(self.feasible_hover_points(region)))

    def fork_for_rollout(self, region: int, trace: FrameTrace):
        """Copy the observable frame state; never read/copy the realized trace."""
        if not self.frame_prepared or self.frame_active:
            raise RuntimeError("rollout fork requires a prepared frame")
        validate_frame_trace(trace, self.cfg)
        sim = P3HierarchicalEnv(self.cfg, region_scope=(region,))
        sim.reset(self.episode)  # uses a private numpy Generator only
        sim.state = self.state.copy()
        sim.frame, sim.global_slot = self.frame, self.global_slot
        sim.membership = self.membership.copy()
        sim.region_users = {region: tuple(self.region_users[region])}
        sim.trace = trace
        sim.frame_prepared = True
        return sim

    def begin_frame(self, raw_actions: dict[int, np.ndarray],
                    completed_actions: dict[int, RegionAction], completion_info: dict | None = None) -> dict:
        if not self.frame_prepared or self.frame_active:
            raise RuntimeError("frame must be prepared and inactive")
        assert self.state is not None and self.trace is not None
        cfg = self.cfg
        if set(raw_actions) != set(self.regions) or set(completed_actions) != set(self.regions):
            raise ValueError("frame input must cover the exact region scope")
        proposals = {m: self.proposal(m, raw_actions[m]) for m in self.regions}
        for m, action in completed_actions.items():
            proposals[m].validate_completion(action)
            validate_region_action(action, self.region_users[m], cfg)
            previous = float(self.state.uav_x[m])
            target = action.target_x(cfg)
            if abs(target - previous) > cfg.reachable_distance_m + 1e-9:
                raise ValueError("completion is unreachable")
            after = float(self.state.battery_j[m]) - relocation_energy_j(previous, target, cfg)
            if after < -1e-9 or (action.hired and after + 1e-9 < activation_energy_required_j(cfg)):
                raise ValueError("completion violates battery feasibility")
        info: dict = {
            "event": "frame_start",
            "episode": self.episode,
            "frame": self.frame,
            "global_slot": self.global_slot,
            "membership": self.membership.tolist(),
            "regions": {},
        }
        self.provider[:] = PROVIDER_NONE
        self.acc = {}
        for m in self.regions:
            users = self.region_users[m]
            battery_before = float(self.state.battery_j[m])
            uav_x_before = float(self.state.uav_x[m])
            action = completed_actions[m]
            reasons = []  # scheduling is feasible before execution; no priority projection
            decoded = self.decode_frame_action(raw_actions[m])
            target_x = action.target_x(cfg)
            return_diag = diagnose_return_to_charge(
                battery_j=battery_before,
                previous_x=uav_x_before,
                depot_x=cfg.depot_x(m),
                will_charge=action.hired == 0,
                cfg=cfg,
            )
            charging_needed = battery_before < activation_energy_required_j(cfg) - 1e-9
            relocation = apply_relocation(battery_before, uav_x_before, target_x, cfg)  # raises on violation
            battery_after = relocation.battery_after_j
            if action.hired and battery_after + 1e-9 < activation_energy_required_j(cfg):
                raise RuntimeError("frame activation reserve violation")
            self.state.battery_j[m] = battery_after
            self.state.uav_x[m] = target_x
            self.region_actions[m] = action
            for u in action.rsu_users:
                self.provider[u] = PROVIDER_RSU
            for u in action.uav_users:
                self.provider[u] = PROVIDER_UAV
            hiring_cost = cfg.lambda_h * cfg.hiring_cost_per_frame * action.hired  # once per frame
            acc = RegionFrameAccumulator(hired=action.hired, hiring_cost=hiring_cost,
                                         quality_hist=np.zeros(self.K), _v=cfg.lyapunov_v)
            acc.energy_consumed_j += relocation.consumed_j
            acc.projection_events += len(reasons)
            self.acc[m] = acc
            self.episode_totals["hired_uav_frames"] += action.hired
            self.episode_totals["uav_frames"] += 1
            self.episode_totals["projection_events"] += len(reasons)
            self.episode_totals["hiring_cost"] += hiring_cost

            info["regions"][m] = {
                "members": list(users),
                "user_state": [self._user_snapshot(u, m) for u in users],
                "raw_action": np.asarray(raw_actions[m]).tolist(),
                "raw_assoc": decoded["assoc"].tolist(),
                "proposal_rsu_users": list(proposals[m].rsu_users),
                "proposal_uav_candidates": list(proposals[m].uav_candidates),
                "unhired_uav_candidates": list(proposals[m].uav_candidates) if not action.hired else [],
                "completion": (completion_info or {}).get(m),
                "feasible_points": self.feasible_hover_points_before(m, battery_before, uav_x_before),
                "executed_hire": action.hired,
                "executed_point": action.point_index,
                "executed_target_x": target_x,
                "executed_rsu_users": list(action.rsu_users),
                "executed_uav_users": list(action.uav_users),
                "unserved_users": [u for u in users if self.provider[u] == PROVIDER_NONE],
                "projection_reasons": reasons,
                "uav_x_before": uav_x_before,
                "uav_x_after": target_x,
                "relocation_energy_j": relocation.consumed_j,
                "relocation_distance_m": abs(target_x - uav_x_before),
                "reachable_distance_m": cfg.reachable_distance_m,
                "battery_before_j": battery_before,
                "battery_after_relocation_j": battery_after,
                "battery_soc_after_relocation": battery_after / cfg.battery_capacity_j,
                "activation_energy_required_j": activation_energy_required_j(cfg),
                "activation_ok": bool((not action.hired) or battery_after + 1e-9 >= activation_energy_required_j(cfg)),
                "charging_needed": bool(charging_needed),
                "return_to_charge": bool(return_diag.is_return_to_charge),
                "precharge_depletion": bool(return_diag.depletion_before_arrival),
                "hiring_cost_weighted": hiring_cost,
            }
        self.local_slot = 0
        self.frame_active = True
        self.frame_prepared = False
        return info

    def feasible_hover_points_before(self, region: int, battery: float, previous_x: float) -> list[int]:
        cfg = self.cfg
        out = []
        for idx, point_x in enumerate(cfg.candidate_points(region)):
            if abs(point_x - previous_x) > cfg.reachable_distance_m + 1e-9:
                continue
            if battery - relocation_energy_j(previous_x, point_x, cfg) + 1e-9 < activation_energy_required_j(cfg):
                continue
            out.append(idx)
        return out

    # ------------------------------------------------------------------
    # slot masks / step
    # ------------------------------------------------------------------
    def slot_action_masks(self, region: int) -> list[np.ndarray]:
        cfg = self.cfg
        action = self.region_actions[region]
        scheduled = set(action.rsu_users) | set(action.uav_users)
        uav_set = set(action.uav_users)
        masks: list[np.ndarray] = []
        for u in range(self.N):  # chunks
            m = np.zeros(cfg.max_chunks_per_slot + 1, dtype=bool)
            if u in scheduled:
                m[:] = True
            else:
                m[0] = True
            masks.append(m)
        for u in range(self.N):  # quality
            m = np.zeros(self.K, dtype=bool)
            if u in scheduled:
                m[:] = True
            else:
                m[0] = True
            masks.append(m)
        for u in range(self.N):  # UAV power level
            m = np.zeros(cfg.uav_power_levels, dtype=bool)
            if u in uav_set:
                m[:] = True
            else:
                m[0] = True
            masks.append(m)
        return masks

    def decode_slot_action(self, raw: np.ndarray) -> dict:
        raw = integer_action(raw, self.cfg.slot_action_nvec, "slot action")
        return {
            "chunks": raw[: self.N].copy(),
            "quality": raw[self.N: 2 * self.N].copy(),
            "power_level": raw[2 * self.N: 3 * self.N].copy(),
        }

    def step_slot(self, raw_actions: dict[int, np.ndarray]) -> SlotStepResult:
        if not self.frame_active:
            raise RuntimeError("step_slot without active frame")
        assert self.state is not None and self.trace is not None
        cfg = self.cfg
        slot = self.local_slot
        remaining = self.T - slot  # R_t: remaining slots including the current one
        metrics: dict[int, RegionSlotMetrics] = {}
        info: dict = {
            "event": "slot",
            "episode": self.episode,
            "frame": self.frame,
            "slot_in_frame": slot,
            "global_slot": self.global_slot,
            "remaining_slots_including_current": remaining,
            "regions": {},
        }
        if set(raw_actions) != set(self.regions):
            raise ValueError("slot input must cover the exact region scope")
        for m in self.regions:
            raw = integer_action(raw_actions[m], cfg.slot_action_nvec, "slot action")
            if any(not mask[token] for mask, token in zip(self.slot_action_masks(m), raw)):
                raise ValueError("slot action violates scheduling mask")
        x_before_all = self.state.user_x.copy()

        for m in self.regions:
            action = self.region_actions[m]
            users = self.region_users[m]
            d = self.decode_slot_action(raw_actions[m])
            rm = RegionSlotMetrics(member_count=len(users))
            user_records: list[dict] = []
            uav_x = action.target_x(cfg)
            battery_before = float(self.state.battery_j[m])

            # ---- UAV power request -> total-power / battery projection (Eq. 3.3, 4.9, 4.10)
            p_eff = battery_power_cap_w(battery_before, remaining, cfg) if action.hired else 0.0
            req_power = {u: 0.0 for u in action.uav_users}
            for u in action.uav_users:
                if int(d["chunks"][u]) > 0:
                    req_power[u] = cfg.power_level_to_w(int(d["power_level"][u]))
            total_req = float(sum(req_power.values()))
            scale = 1.0 if total_req <= p_eff + 1e-12 else p_eff / max(total_req, 1e-12)
            exec_power = {u: req_power[u] * scale for u in action.uav_users}
            total_exec = float(sum(exec_power.values()))
            if total_exec > p_eff + 1e-9:
                raise RuntimeError("executed UAV power exceeds P_eff after projection")

            # ---- per-user delivery
            uav_scheduled = set(action.uav_users)
            rsu_scheduled = set(action.rsu_users)
            for u in users:
                q_before = float(self.state.queue[u])
                z_before = cfg.large_queue_level - q_before
                prov = int(self.provider[u])
                req_l = int(d["chunks"][u])
                k = int(d["quality"][u])
                x = float(self.state.user_x[u])
                rec: dict = {
                    "user": u, "provider": prov, "x_m": x,
                    "q_before": q_before, "z_before": z_before,
                    "req_chunks": req_l, "req_quality": k,
                    "req_power_level": int(d["power_level"][u]),
                    "req_power_w": 0.0, "exec_power_w": 0.0,
                    "fading": None, "gain": None, "capacity_bps": 0.0,
                    "rsu_horizontal_distance_m": abs(cfg.rsu_x(m) - x),
                    "uav_horizontal_distance_m": abs(uav_x - x) if action.hired else None,
                    "feasible_by_rate": 0, "queue_admissible_cap": 0,
                    "feasible_chunks": 0, "delivered": 0, "min_required_power_w": 0.0,
                }
                queue_cap = self.guard.max_queue_admissible_chunks(z_before) if cfg.enforce_queue_admissibility else cfg.max_chunks_per_slot
                rec["queue_admissible_cap"] = int(queue_cap)
                chunk_bits = cfg.chunk_size_bits[k]
                capacity = 0.0
                feasible_rate = 0
                if u in rsu_scheduled and req_l > 0:
                    fading = float(self.trace.rsu_fading[slot, m, u])
                    capacity = rsu_link_capacity_bps(rec["rsu_horizontal_distance_m"], fading, cfg)
                    feasible_rate = int(math.floor(capacity * cfg.slot_duration_s / chunk_bits + 1e-9))
                    rec.update(fading=fading, capacity_bps=capacity,
                               exec_power_w=cfg.rsu_total_power_w / cfg.rsu_capacity)
                elif u in uav_scheduled and req_l > 0:
                    fading = float(self.trace.uav_fading[slot, m, action.point_index, u])
                    capacity, gain = uav_link_capacity_bps(rec["uav_horizontal_distance_m"], fading, exec_power[u], cfg)
                    feasible_rate = int(math.floor(capacity * cfg.slot_duration_s / chunk_bits + 1e-9))
                    rec.update(fading=fading, gain=gain, capacity_bps=capacity,
                               req_power_w=req_power[u], exec_power_w=exec_power[u])
                elif u in uav_scheduled:
                    rec.update(req_power_w=0.0, exec_power_w=0.0)
                feasible = int(min(cfg.max_chunks_per_slot, queue_cap, feasible_rate)) if req_l > 0 else 0
                delivered = int(min(req_l, feasible))
                if delivered > 0 and delivered * chunk_bits > capacity * cfg.slot_duration_s + 1e-6:
                    raise RuntimeError("rate feasibility (3.1)/(3.2) violated")
                if u in uav_scheduled and delivered > 0:
                    rec["min_required_power_w"] = required_uav_power_w(
                        delivered, k, rec["uav_horizontal_distance_m"], rec["fading"], cfg)
                utility = delivered * cfg.quality_utility[k]
                degradation = delivered * (cfg.quality_max - cfg.quality_utility[k])  # Eq. (3.7)
                stalled = int(q_before < cfg.playback_chunks_per_slot)               # Eq. (3.11)
                q_after, _, z_after, departure = update_playback_queue(q_before, float(delivered), cfg)
                dpp = cfg.alpha_z * z_before * (departure - delivered) + cfg.lyapunov_v * degradation  # Eq. (6.12)

                rec.update(
                    feasible_by_rate=feasible_rate, feasible_chunks=feasible, delivered=delivered,
                    utility=utility, degradation=degradation, departure=departure,
                    q_after=q_after, z_after=z_after, stall=stalled,
                    q_gt_qe=int(q_before > cfg.large_queue_level), dpp_slot_cost=dpp,
                    identity_ok=bool(abs(z_after - (cfg.large_queue_level - q_after)) < 1e-9),
                    last_quality_before=int(self.state.last_quality_index[u]),
                )
                self.state.queue[u] = q_after
                self.state.last_stalled[u] = bool(stalled)
                if delivered > 0:
                    self.acc[m].quality_hist[k] += delivered
                    self.state.last_quality_index[u] = k
                rm.dpp_slot_cost += dpp
                rm.degradation += degradation
                rm.delivered_chunks += delivered
                rm.utility += utility
                rm.stall_user_slots += stalled
                rm.served_user_slots += int(delivered > 0)
                rm.q_gt_qe_user_slots += int(q_before > cfg.large_queue_level)
                user_records.append(rec)

            # ---- UAV battery update (Eq. 4.2-4.6)
            if action.hired:
                step = apply_active_slot(battery_before, total_exec, remaining, cfg)  # raises on violation
                battery_after = step.battery_after_j
                hover_e = cfg.hover_energy_per_slot_j
                comm_e = step.consumed_j - hover_e
                charge_e = 0.0
                required_after = cfg.reserve_battery_j + (remaining - 1) * cfg.hover_energy_per_slot_j
                rm.reserve_violation = int(battery_after + 1e-7 < required_after)
                self.acc[m].energy_consumed_j += step.consumed_j
            else:
                step = apply_unhired_slot(battery_before, cfg)
                battery_after = step.battery_after_j
                hover_e = comm_e = 0.0
                charge_e = step.charged_j
                required_after = 0.0
                self.acc[m].energy_charged_j += charge_e
            rm.power_violation = int(total_exec > p_eff + 1e-9)
            rm.total_uav_power_w = total_exec
            rm.p_eff_w = p_eff
            self.state.battery_j[m] = battery_after

            # ---- reward ingredients
            if users:
                z_pos = np.maximum(cfg.large_queue_level - self.state.queue[list(users)], 0.0)
                rm.constraint_cost = float(np.mean(z_pos) / cfg.large_queue_level)
                rm.stall_fraction = rm.stall_user_slots / len(users)
            acc = self.acc[m]
            acc.dpp_slot_sum += rm.dpp_slot_cost
            acc.degradation_sum += rm.degradation
            acc.delivered_sum += rm.delivered_chunks
            acc.utility_sum += rm.utility
            acc.stall_user_slots += rm.stall_user_slots
            acc.served_user_slots += rm.served_user_slots
            acc.user_slots += len(users)
            acc.constraint_sum += rm.constraint_cost
            acc.stall_fraction_sum += rm.stall_fraction
            acc.q_gt_qe_user_slots += rm.q_gt_qe_user_slots
            acc.reserve_violations += rm.reserve_violation
            acc.power_violations += rm.power_violation
            metrics[m] = rm

            info["regions"][m] = {
                "hired": action.hired,
                "point_index": action.point_index,
                "uav_x": uav_x,
                "rsu_users": list(action.rsu_users),
                "uav_users": list(action.uav_users),
                "raw_action": np.asarray(raw_actions[m]).tolist(),
                "p_eff_w": p_eff,
                "p_max_w": cfg.uav_max_total_power_w,
                "total_requested_power_w": total_req,
                "power_scale": scale,
                "total_executed_power_w": total_exec,
                "battery_before_j": battery_before,
                "hover_energy_j": hover_e,
                "communication_energy_j": comm_e,
                "charge_accepted_j": charge_e,
                "battery_after_j": battery_after,
                "battery_soc_after": battery_after / cfg.battery_capacity_j,
                "reserve_required_after_j": required_after,
                "reserve_ok": bool(rm.reserve_violation == 0),
                "users": user_records,
                "dpp_slot_cost": rm.dpp_slot_cost,
                "degradation": rm.degradation,
                "delivered_chunks": rm.delivered_chunks,
                "stall_user_slots": rm.stall_user_slots,
                "constraint_cost_z_norm": rm.constraint_cost,
                "stall_fraction": rm.stall_fraction,
            }

        # ---- mobility: one slot of constant-speed motion on the ring road
        self.state.user_x = np.mod(self.state.user_x + self.state.user_speed * cfg.slot_duration_s, cfg.road_length_m)
        info["mobility"] = {
            "x_before": x_before_all.tolist(),
            "x_after": self.state.user_x.tolist(),
            "speed": self.state.user_speed.tolist(),
        }
        validate_state(self.state, cfg)
        self.episode_totals["max_queue"] = max(self.episode_totals["max_queue"], float(np.max(self.state.queue)))
        self.episode_totals["min_battery_soc"] = min(self.episode_totals["min_battery_soc"], float(np.min(self.state.battery_j)) / cfg.battery_capacity_j)

        self.local_slot += 1
        self.global_slot += 1
        frame_done = self.local_slot >= self.T
        if frame_done:
            info["frame_summary"] = self._finish_frame()
            self.frame_active = False
            self.frame += 1
        return SlotStepResult(frame_done=frame_done, metrics=metrics, info=info)

    # ------------------------------------------------------------------
    def _finish_frame(self) -> dict:
        cfg = self.cfg
        summary: dict = {"event": "frame_end", "episode": self.episode, "frame": self.frame, "regions": {}}
        tot_dpp = tot_orig = 0.0
        for m, acc in self.acc.items():
            summary["regions"][m] = {
                "hired": acc.hired,
                "hiring_cost_weighted": acc.hiring_cost,
                "dpp_slot_sum": acc.dpp_slot_sum,
                "frame_dpp_cost": acc.frame_dpp_cost,                 # Eq. (6.11)/(8.4)
                "original_cost": acc.original_cost,                   # Eq. (5.1) frame term
                "degradation_sum": acc.degradation_sum,
                "delivered_chunks": acc.delivered_sum,
                "utility_sum": acc.utility_sum,
                "stall_user_slots": acc.stall_user_slots,
                "served_user_slots": acc.served_user_slots,
                "user_slots": acc.user_slots,
                "stall_ratio": acc.stall_user_slots / max(acc.user_slots, 1),
                "constraint_mean": acc.constraint_sum / self.T,
                "stall_fraction_mean": acc.stall_fraction_sum / self.T,
                "q_gt_qe_user_slots": acc.q_gt_qe_user_slots,
                "energy_consumed_j": acc.energy_consumed_j,
                "energy_charged_j": acc.energy_charged_j,
                "reserve_violations": acc.reserve_violations,
                "power_violations": acc.power_violations,
                "projection_events": acc.projection_events,
                "quality_histogram": acc.quality_hist.tolist(),
                "battery_soc_end": float(self.state.battery_j[m]) / cfg.battery_capacity_j,
                "uav_x_end": float(self.state.uav_x[m]),
            }
            tot_dpp += acc.frame_dpp_cost
            tot_orig += acc.original_cost
            t = self.episode_totals
            t["dpp_cost"] += acc.frame_dpp_cost
            t["original_cost"] += acc.original_cost
            t["degradation"] += acc.degradation_sum
            t["delivered_chunks"] += acc.delivered_sum
            t["utility"] += acc.utility_sum
            t["stall_user_slots"] += acc.stall_user_slots
            t["served_user_slots"] += acc.served_user_slots
            t["user_slots"] += acc.user_slots
            t["q_gt_qe_user_slots"] += acc.q_gt_qe_user_slots
            t["reserve_violations"] += acc.reserve_violations
            t["power_violations"] += acc.power_violations
            t["constraint_sum"] += acc.constraint_sum
            t["constraint_samples"] += self.T
            t["stall_fraction_sum"] += acc.stall_fraction_sum
            t["energy_consumed_j"] += acc.energy_consumed_j
            t["energy_charged_j"] += acc.energy_charged_j
        summary["frame_dpp_cost_total"] = tot_dpp
        summary["original_cost_total"] = tot_orig
        summary["mean_Q"] = float(np.mean(self.state.queue))
        summary["mean_Z"] = float(np.mean(cfg.large_queue_level - self.state.queue))
        return summary

    def episode_summary(self) -> dict:
        t = self.episode_totals
        cfg = self.cfg
        us = max(t["user_slots"], 1)
        return {
            "episode": self.episode,
            "frames": self.frame,
            "dpp_cost_total": t["dpp_cost"],
            "dpp_cost_per_user_slot": t["dpp_cost"] / us,
            "original_cost_total": t["original_cost"],
            "original_cost_per_user_slot": t["original_cost"] / us,
            "degradation_total": t["degradation"],
            "hiring_cost_total": t["hiring_cost"],
            "delivered_chunks_total": t["delivered_chunks"],
            "delivered_chunks_per_user_slot": t["delivered_chunks"] / us,
            "average_quality_utility": t["utility"] / max(t["delivered_chunks"], 1e-9),
            "stall_ratio": t["stall_user_slots"] / us,
            "served_user_ratio": t["served_user_slots"] / us,
            "z_constraint_mean": t["constraint_sum"] / max(t["constraint_samples"], 1),
            "hire_rate": t["hired_uav_frames"] / max(t["uav_frames"], 1),
            "projection_events": t["projection_events"],
            "q_gt_qe_rate": t["q_gt_qe_user_slots"] / us,
            "max_queue": t["max_queue"],
            "min_battery_soc": t["min_battery_soc"],
            "reserve_violations": t["reserve_violations"],
            "power_violations": t["power_violations"],
            "energy_consumed_j": t["energy_consumed_j"],
            "energy_charged_j": t["energy_charged_j"],
            "mean_final_Q": float(np.mean(self.state.queue)),
            "mean_final_Z": float(np.mean(cfg.large_queue_level - self.state.queue)),
            "mean_final_soc": float(np.mean(self.state.battery_j)) / cfg.battery_capacity_j,
        }

    # ------------------------------------------------------------------
    def _user_snapshot(self, u: int, region: int) -> dict:
        cfg = self.cfg
        x = float(self.state.user_x[u])
        q = float(self.state.queue[u])
        return {
            "user": int(u),
            "x_m": x,
            "speed_mps": float(self.state.user_speed[u]),
            "Q": q,
            "Z": cfg.large_queue_level - q,
            "rsu_horizontal_distance_m": abs(cfg.rsu_x(region) - x),
            "candidate_horizontal_distance_m": [abs(p - x) for p in cfg.candidate_points(region)],
            "last_quality_index": int(self.state.last_quality_index[u]),
        }

    def assert_consistency(self) -> None:
        cfg = self.cfg
        assert self.state is not None
        validate_state(self.state, cfg)
        if self.frame_active:
            for m, action in self.region_actions.items():
                validate_region_action(action, self.region_users[m], cfg)
                if action.hired:
                    remaining = self.T - self.local_slot
                    need = cfg.reserve_battery_j + max(remaining, 0) * cfg.hover_energy_per_slot_j
                    if float(self.state.battery_j[m]) + 1e-7 < need:
                        raise AssertionError("remaining-hover reserve violated")


# ----------------------------------------------------------------------
# reward composition (train.py uses these)
# ----------------------------------------------------------------------
def slot_training_reward(cfg: HPPOConfig, rm: RegionSlotMetrics, dual: float) -> tuple[float, float]:
    """Return (base_reward, training_reward) for one region-slot."""

    if cfg.reward_mode == "dpp":
        base = -rm.dpp_slot_cost * cfg.ppo_reward_scale
        return base, base
    base = -rm.degradation * cfg.objective_reward_scale
    r = base
    if cfg.reward_mode == "objective_lagrangian":
        r -= cfg.constraint_reward_scale * dual * rm.constraint_cost
    if cfg.stall_training_penalty != 0.0:
        r -= cfg.stall_training_penalty * rm.stall_fraction
    return base, r


def frame_training_reward(cfg: HPPOConfig, region_summary: dict, dual: float) -> tuple[float, float]:
    if cfg.reward_mode == "dpp":
        base = -region_summary["frame_dpp_cost"] * cfg.ppo_reward_scale  # == run/p3_train_ppo.py reward
        return base, base
    base = -region_summary["original_cost"] * cfg.objective_reward_scale
    r = base
    if cfg.reward_mode == "objective_lagrangian":
        r -= cfg.constraint_reward_scale * dual * region_summary["constraint_mean"] * cfg.frame_slots
    if cfg.stall_training_penalty != 0.0:
        r -= cfg.stall_training_penalty * region_summary["stall_fraction_mean"]
    return base, r
