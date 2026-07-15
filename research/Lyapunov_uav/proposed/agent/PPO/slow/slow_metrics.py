from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class HRLMetrics:
    """Accumulate slot metrics without averaging ratios prematurely."""

    slots: int = 0
    rounds: int = 0
    fast_reward: float = 0.0
    slow_reward: float = 0.0
    hire_cost: float = 0.0

    delivery: float = 0.0
    transmitted_rsu: float = 0.0
    transmitted_uav: float = 0.0
    quality: float = 0.0
    degradation: float = 0.0
    stall: float = 0.0
    scheduled_stall: float = 0.0
    scheduled_playback: float = 0.0
    unscheduled_stall: float = 0.0
    unscheduled_playback: float = 0.0

    consumed_soc: float = 0.0
    charged_soc: float = 0.0
    outage_slots: float = 0.0
    charging_slots: float = 0.0
    min_soc: float = float("inf")

    fast_active_dims: float = 0.0
    fast_active_ratio: float = 0.0
    fast_service_rate: float = 0.0
    fast_requested_chunks: float = 0.0
    fast_action_saturation: float = 0.0
    hired_uav_slots: float = 0.0
    serving_uav_slots: float = 0.0
    idle_hired_uav_slots: float = 0.0
    charging_uav_slots: float = 0.0
    fast_layer_ratios: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )

    hired_uav_rounds: float = 0.0
    rsu_links_rounds: float = 0.0
    uav_links_rounds: float = 0.0
    scheduled_users_rounds: float = 0.0
    slow_active_dims_rounds: float = 0.0
    slow_active_ratio_rounds: float = 0.0
    slow_projection_count: float = 0.0

    def add_slot(
        self,
        info: Dict[str, Any],
        fast_selected: Dict[str, Any],
        num_layers: int,
    ) -> None:
        reward_components = info.get("reward_components", {})
        fast = reward_components.get("fast_reward_components", {})
        self.slots += 1
        self.fast_reward += float(reward_components.get("fast_reward", 0.0))
        self.delivery += float(fast.get("sum_delivery", 0.0))
        self.transmitted_rsu += float(
            np.asarray(
                info.get("transmitted_rsu_per_user", []),
                dtype=np.float32,
            ).sum()
        )
        self.transmitted_uav += float(
            np.asarray(
                info.get("transmitted_uav_per_user", []),
                dtype=np.float32,
            ).sum()
        )
        self.quality += float(fast.get("sum_quality", 0.0))
        self.degradation += float(
            fast.get("sum_quality_degradation", 0.0)
        )
        self.consumed_soc += float(fast.get("sum_consumed_soc", 0.0))
        self.charged_soc += float(fast.get("sum_charged_soc", 0.0))

        stall = np.asarray(info.get("stall", []), dtype=np.float32)
        playback = np.asarray(info.get("playback", []), dtype=np.float32)
        connection_type = np.asarray(
            info.get("prev_connection_state", {}).get("connection_type", []),
            dtype=np.int32,
        )
        if stall.shape == playback.shape:
            self.stall += float(stall.sum())
        if stall.shape == playback.shape == connection_type.shape:
            scheduled = connection_type > 0
            unscheduled = ~scheduled
            self.scheduled_stall += float(stall[scheduled].sum())
            self.scheduled_playback += float(playback[scheduled].sum())
            self.unscheduled_stall += float(stall[unscheduled].sum())
            self.unscheduled_playback += float(playback[unscheduled].sum())

        next_soc = np.asarray(info.get("next_E", []), dtype=np.float32)
        if next_soc.size > 0:
            self.min_soc = min(self.min_soc, float(next_soc.min()))
        self.outage_slots += float(
            np.asarray(info.get("outage", []), dtype=np.float32).sum()
        )
        self.charging_slots += float(
            np.asarray(info.get("charging_state", []), dtype=np.float32).sum()
        )

        hiring = np.asarray(
            info.get("uav_hiring", []), dtype=np.int32
        ).reshape(-1)
        self.hired_uav_slots += float(hiring.sum())
        for index, battery_info in enumerate(
            info.get("battery_step_info", [])
        ):
            if index >= hiring.size or int(hiring[index]) != 1:
                continue
            mode = str(battery_info.get("mode", "")).lower()
            if mode == "serve":
                self.serving_uav_slots += 1.0
            elif mode == "charge":
                self.charging_uav_slots += 1.0
            elif mode == "idle":
                self.idle_hired_uav_slots += 1.0

        self.fast_active_dims += float(
            fast_selected.get("active_action_dims", 0.0)
        )
        self.fast_active_ratio += float(
            fast_selected.get("active_action_ratio", 0.0)
        )
        self.fast_service_rate += float(
            fast_selected.get("service_rate", 0.0)
        )
        self.fast_requested_chunks += float(
            fast_selected.get("mean_requested_chunks", 0.0)
        )
        self.fast_action_saturation += float(
            fast_selected.get("action_saturation_ratio", 0.0)
        )
        if self.fast_layer_ratios.shape != (num_layers,):
            self.fast_layer_ratios = np.zeros(num_layers, dtype=np.float64)
        for layer in range(1, num_layers + 1):
            self.fast_layer_ratios[layer - 1] += float(
                fast_selected.get(f"layer_{layer}_ratio", 0.0)
            )

    def add_round(
        self,
        slow_reward: float,
        slow_components: Dict[str, Any],
        slow_selected: Dict[str, Any],
    ) -> None:
        self.rounds += 1
        self.slow_reward += float(slow_reward)
        self.hire_cost += float(slow_components.get("hire_cost", 0.0))
        action = slow_selected["env_action"]
        self.hired_uav_rounds += float(
            np.asarray(action["uav_hiring"], dtype=np.float32).sum()
        )
        self.rsu_links_rounds += float(
            np.asarray(action["rsu_scheduling"], dtype=np.float32).sum()
        )
        self.uav_links_rounds += float(
            np.asarray(action["uav_scheduling"], dtype=np.float32).sum()
        )
        info = slow_selected.get("action_info", {})
        self.scheduled_users_rounds += float(
            info.get("num_scheduled_users", 0.0)
        )
        self.slow_active_dims_rounds += float(
            info.get("active_action_dims", 0.0)
        )
        self.slow_active_ratio_rounds += float(
            info.get("active_action_ratio", 0.0)
        )
        self.slow_projection_count += float(
            info.get("projection_count", 0.0)
        )

    def merge(self, other: "HRLMetrics") -> None:
        for name in (
            "slots", "rounds", "fast_reward", "slow_reward", "hire_cost",
            "delivery", "transmitted_rsu", "transmitted_uav", "quality",
            "degradation", "stall", "scheduled_stall",
            "scheduled_playback", "unscheduled_stall",
            "unscheduled_playback", "consumed_soc", "charged_soc",
            "outage_slots", "charging_slots", "fast_active_dims",
            "fast_active_ratio", "fast_service_rate", "fast_requested_chunks",
            "fast_action_saturation", "hired_uav_slots",
            "serving_uav_slots", "idle_hired_uav_slots",
            "charging_uav_slots", "hired_uav_rounds",
            "rsu_links_rounds", "uav_links_rounds",
            "scheduled_users_rounds", "slow_active_dims_rounds",
            "slow_active_ratio_rounds", "slow_projection_count",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.min_soc = min(self.min_soc, other.min_soc)
        if self.fast_layer_ratios.shape != other.fast_layer_ratios.shape:
            self.fast_layer_ratios = np.zeros_like(other.fast_layer_ratios)
        self.fast_layer_ratios += other.fast_layer_ratios

    def summary(self) -> Dict[str, float]:
        slots = max(self.slots, 1)
        rounds = max(self.rounds, 1)
        delivery = self.delivery
        result = {
            "slots": float(self.slots),
            "rounds": float(self.rounds),
            "fast_reward": float(self.fast_reward),
            "fast_reward_per_slot": float(self.fast_reward / slots),
            "slow_reward": float(self.slow_reward),
            "slow_reward_per_round": float(self.slow_reward / rounds),
            "hire_cost": float(self.hire_cost),
            "hire_cost_per_round": float(self.hire_cost / rounds),
            "delivery": float(delivery),
            "delivery_per_slot": float(delivery / slots),
            "transmitted_rsu_per_slot": float(
                self.transmitted_rsu / slots
            ),
            "transmitted_uav_per_slot": float(
                self.transmitted_uav / slots
            ),
            "uav_transmission_share": (
                float(
                    self.transmitted_uav
                    / (self.transmitted_rsu + self.transmitted_uav)
                )
                if self.transmitted_rsu + self.transmitted_uav > 0.0
                else 0.0
            ),
            "quality_per_chunk": (
                float(self.quality / delivery) if delivery > 0.0 else 0.0
            ),
            "quality_degradation_per_chunk": (
                float(self.degradation / delivery) if delivery > 0.0 else 0.0
            ),
            "stall": float(self.stall),
            "scheduled_stall_rate": (
                float(self.scheduled_stall / self.scheduled_playback)
                if self.scheduled_playback > 0.0 else 0.0
            ),
            "unscheduled_stall_rate": (
                float(self.unscheduled_stall / self.unscheduled_playback)
                if self.unscheduled_playback > 0.0 else 0.0
            ),
            "consumed_soc": float(self.consumed_soc),
            "charged_soc": float(self.charged_soc),
            "outage_slots": float(self.outage_slots),
            "charging_slots": float(self.charging_slots),
            "min_soc": (
                float(self.min_soc) if np.isfinite(self.min_soc) else 0.0
            ),
            "fast_active_action_dims_mean": float(self.fast_active_dims / slots),
            "fast_active_action_ratio_mean": float(self.fast_active_ratio / slots),
            "fast_service_rate": float(self.fast_service_rate / slots),
            "fast_mean_requested_chunks": float(
                self.fast_requested_chunks / slots
            ),
            "fast_action_saturation_ratio_mean": float(
                self.fast_action_saturation / slots
            ),
            "hired_uav_per_slot": float(self.hired_uav_slots / slots),
            "serving_hired_uav_ratio": (
                float(self.serving_uav_slots / self.hired_uav_slots)
                if self.hired_uav_slots > 0.0 else 0.0
            ),
            "idle_hired_uav_ratio": (
                float(self.idle_hired_uav_slots / self.hired_uav_slots)
                if self.hired_uav_slots > 0.0 else 0.0
            ),
            "charging_hired_uav_ratio": (
                float(self.charging_uav_slots / self.hired_uav_slots)
                if self.hired_uav_slots > 0.0 else 0.0
            ),
            "hired_uav_per_round": float(self.hired_uav_rounds / rounds),
            "rsu_links_per_round": float(self.rsu_links_rounds / rounds),
            "uav_links_per_round": float(self.uav_links_rounds / rounds),
            "scheduled_users_per_round": float(
                self.scheduled_users_rounds / rounds
            ),
            "slow_active_action_dims_mean": float(
                self.slow_active_dims_rounds / rounds
            ),
            "slow_active_action_ratio_mean": float(
                self.slow_active_ratio_rounds / rounds
            ),
            "slow_projection_count": float(self.slow_projection_count),
        }
        for index, value in enumerate(self.fast_layer_ratios, start=1):
            result[f"fast_layer_{index}_ratio"] = float(value / slots)
        return result