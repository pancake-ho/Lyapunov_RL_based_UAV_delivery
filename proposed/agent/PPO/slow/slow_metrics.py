from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class SlowDPPMetrics:
    """Accumulate slow-DPP evaluation metrics without averaging ratios early."""

    slots: int = 0
    rounds: int = 0

    fast_reward: float = 0.0
    slow_reward: float = 0.0
    round_dpp_cost: float = 0.0
    hire_cost_raw: float = 0.0

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
    fast_layer_ratios: np.ndarray = field(
        default_factory=lambda: np.zeros(
            4,
            dtype=np.float64,
        )
    )

    hired_uav_rounds: float = 0.0
    idle_hired_uav_rounds: float = 0.0
    rsu_links_rounds: float = 0.0
    uav_links_rounds: float = 0.0
    scheduled_users_rounds: float = 0.0
    slow_active_dims_rounds: float = 0.0
    slow_active_ratio_rounds: float = 0.0
    candidate_requests_rounds: float = 0.0
    unique_candidates_rounds: float = 0.0
    coordinate_sweeps_rounds: float = 0.0

    predicted_dpp_cost: float = 0.0
    realized_dpp_cost: float = 0.0
    prediction_error: float = 0.0
    prediction_abs_error: float = 0.0

    def add_slot(
        self,
        info: Dict[str, Any],
        fast_selected: Dict[str, Any],
        num_layers: int,
    ) -> None:
        reward_components = dict(
            info.get("reward_components", {})
        )
        fast = dict(
            reward_components.get(
                "fast_reward_components",
                {},
            )
        )

        self.slots += 1
        self.fast_reward += float(
            reward_components.get("fast_reward", 0.0)
        )
        self.delivery += float(
            fast.get("sum_delivery", 0.0)
        )
        self.quality += float(
            fast.get("sum_quality", 0.0)
        )
        self.degradation += float(
            fast.get("sum_quality_degradation", 0.0)
        )
        self.consumed_soc += float(
            fast.get("sum_consumed_soc", 0.0)
        )
        self.charged_soc += float(
            fast.get("sum_charged_soc", 0.0)
        )

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

        stall = np.asarray(
            info.get("stall", []),
            dtype=np.float32,
        )
        playback = np.asarray(
            info.get("playback", []),
            dtype=np.float32,
        )
        connection_type = np.asarray(
            info.get(
                "prev_connection_state",
                {},
            ).get("connection_type", []),
            dtype=np.int32,
        )

        if stall.shape == playback.shape:
            self.stall += float(stall.sum())
        if (
            stall.shape
            == playback.shape
            == connection_type.shape
        ):
            scheduled = connection_type > 0
            unscheduled = ~scheduled
            self.scheduled_stall += float(
                stall[scheduled].sum()
            )
            self.scheduled_playback += float(
                playback[scheduled].sum()
            )
            self.unscheduled_stall += float(
                stall[unscheduled].sum()
            )
            self.unscheduled_playback += float(
                playback[unscheduled].sum()
            )

        next_soc = np.asarray(
            info.get("next_E", []),
            dtype=np.float32,
        )
        if next_soc.size > 0:
            self.min_soc = min(
                self.min_soc,
                float(next_soc.min()),
            )
        self.outage_slots += float(
            np.asarray(
                info.get("outage", []),
                dtype=np.float32,
            ).sum()
        )
        self.charging_slots += float(
            np.asarray(
                info.get("charging_state", []),
                dtype=np.float32,
            ).sum()
        )

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
            fast_selected.get(
                "action_saturation_ratio",
                0.0,
            )
        )

        if self.fast_layer_ratios.shape != (num_layers,):
            self.fast_layer_ratios = np.zeros(
                num_layers,
                dtype=np.float64,
            )
        for layer in range(1, num_layers + 1):
            self.fast_layer_ratios[layer - 1] += float(
                fast_selected.get(
                    f"layer_{layer}_ratio",
                    0.0,
                )
            )

    def add_round(
        self,
        slow_reward: float,
        slow_components: Dict[str, Any],
        slow_selected: Dict[str, Any],
        predicted_cost: float,
        realized_cost: float,
    ) -> None:
        self.rounds += 1
        self.slow_reward += float(slow_reward)
        self.round_dpp_cost += float(
            slow_components.get(
                "round_dpp_cost",
                realized_cost,
            )
        )
        self.hire_cost_raw += float(
            slow_components.get("hire_cost_raw", 0.0)
        )

        action = slow_selected["env_action"]
        hiring = np.asarray(
            action["uav_hiring"],
            dtype=np.int32,
        )
        uav = np.asarray(
            action["uav_scheduling"],
            dtype=np.int32,
        )
        rsu = np.asarray(
            action["rsu_scheduling"],
            dtype=np.int32,
        )
        self.hired_uav_rounds += float(hiring.sum())
        self.idle_hired_uav_rounds += float(
            np.sum(
                (hiring == 1)
                & (uav.sum(axis=1) == 0)
            )
        )
        self.rsu_links_rounds += float(rsu.sum())
        self.uav_links_rounds += float(uav.sum())

        action_info = dict(
            slow_selected.get("action_info", {})
        )
        self.scheduled_users_rounds += float(
            action_info.get("num_scheduled_users", 0.0)
        )
        self.slow_active_dims_rounds += float(
            action_info.get("active_action_dims", 0.0)
        )
        self.slow_active_ratio_rounds += float(
            action_info.get("active_action_ratio", 0.0)
        )
        self.candidate_requests_rounds += float(
            action_info.get("candidate_requests", 0.0)
        )
        self.unique_candidates_rounds += float(
            action_info.get("unique_candidates", 0.0)
        )
        self.coordinate_sweeps_rounds += float(
            action_info.get(
                "coordinate_sweeps_completed",
                0.0,
            )
        )

        error = float(realized_cost - predicted_cost)
        self.predicted_dpp_cost += float(predicted_cost)
        self.realized_dpp_cost += float(realized_cost)
        self.prediction_error += error
        self.prediction_abs_error += abs(error)

    def merge(self, other: "SlowDPPMetrics") -> None:
        scalar_fields = (
            "slots",
            "rounds",
            "fast_reward",
            "slow_reward",
            "round_dpp_cost",
            "hire_cost_raw",
            "delivery",
            "transmitted_rsu",
            "transmitted_uav",
            "quality",
            "degradation",
            "stall",
            "scheduled_stall",
            "scheduled_playback",
            "unscheduled_stall",
            "unscheduled_playback",
            "consumed_soc",
            "charged_soc",
            "outage_slots",
            "charging_slots",
            "fast_active_dims",
            "fast_active_ratio",
            "fast_service_rate",
            "fast_requested_chunks",
            "fast_action_saturation",
            "hired_uav_rounds",
            "idle_hired_uav_rounds",
            "rsu_links_rounds",
            "uav_links_rounds",
            "scheduled_users_rounds",
            "slow_active_dims_rounds",
            "slow_active_ratio_rounds",
            "candidate_requests_rounds",
            "unique_candidates_rounds",
            "coordinate_sweeps_rounds",
            "predicted_dpp_cost",
            "realized_dpp_cost",
            "prediction_error",
            "prediction_abs_error",
        )
        for name in scalar_fields:
            setattr(
                self,
                name,
                getattr(self, name) + getattr(other, name),
            )

        self.min_soc = min(self.min_soc, other.min_soc)
        if (
            self.fast_layer_ratios.shape
            != other.fast_layer_ratios.shape
        ):
            raise ValueError(
                "Cannot merge metrics with different layer counts."
            )
        self.fast_layer_ratios += other.fast_layer_ratios

    def summary(self) -> Dict[str, float]:
        slots = max(int(self.slots), 1)
        rounds = max(int(self.rounds), 1)
        delivery = float(self.delivery)
        transmitted = float(
            self.transmitted_rsu + self.transmitted_uav
        )

        result = {
            "slots": float(self.slots),
            "rounds": float(self.rounds),
            "fast_reward": float(self.fast_reward),
            "fast_reward_per_slot": float(
                self.fast_reward / slots
            ),
            "slow_reward": float(self.slow_reward),
            "slow_reward_per_round": float(
                self.slow_reward / rounds
            ),
            "round_dpp_cost": float(self.round_dpp_cost),
            "round_dpp_cost_mean": float(
                self.round_dpp_cost / rounds
            ),
            "hire_cost_raw": float(self.hire_cost_raw),
            "hire_cost_raw_per_round": float(
                self.hire_cost_raw / rounds
            ),
            "delivery": delivery,
            "delivery_per_slot": float(
                delivery / slots
            ),
            "transmitted_rsu_per_slot": float(
                self.transmitted_rsu / slots
            ),
            "transmitted_uav_per_slot": float(
                self.transmitted_uav / slots
            ),
            "uav_transmission_share": (
                float(self.transmitted_uav / transmitted)
                if transmitted > 0.0
                else 0.0
            ),
            "quality_per_chunk": (
                float(self.quality / delivery)
                if delivery > 0.0
                else 0.0
            ),
            "quality_degradation_per_chunk": (
                float(self.degradation / delivery)
                if delivery > 0.0
                else 0.0
            ),
            "stall": float(self.stall),
            "scheduled_stall_rate": (
                float(
                    self.scheduled_stall
                    / self.scheduled_playback
                )
                if self.scheduled_playback > 0.0
                else 0.0
            ),
            "unscheduled_stall_rate": (
                float(
                    self.unscheduled_stall
                    / self.unscheduled_playback
                )
                if self.unscheduled_playback > 0.0
                else 0.0
            ),
            "consumed_soc": float(self.consumed_soc),
            "charged_soc": float(self.charged_soc),
            "outage_slots": float(self.outage_slots),
            "charging_slots": float(self.charging_slots),
            "min_soc": (
                float(self.min_soc)
                if np.isfinite(self.min_soc)
                else 0.0
            ),
            "fast_active_action_dims_mean": float(
                self.fast_active_dims / slots
            ),
            "fast_active_action_ratio_mean": float(
                self.fast_active_ratio / slots
            ),
            "fast_service_rate": float(
                self.fast_service_rate / slots
            ),
            "fast_mean_requested_chunks": float(
                self.fast_requested_chunks / slots
            ),
            "fast_action_saturation_ratio_mean": float(
                self.fast_action_saturation / slots
            ),
            "hired_uav_per_round": float(
                self.hired_uav_rounds / rounds
            ),
            "idle_hired_uav_per_round": float(
                self.idle_hired_uav_rounds / rounds
            ),
            "rsu_links_per_round": float(
                self.rsu_links_rounds / rounds
            ),
            "uav_links_per_round": float(
                self.uav_links_rounds / rounds
            ),
            "scheduled_users_per_round": float(
                self.scheduled_users_rounds / rounds
            ),
            "slow_active_action_dims_mean": float(
                self.slow_active_dims_rounds / rounds
            ),
            "slow_active_action_ratio_mean": float(
                self.slow_active_ratio_rounds / rounds
            ),
            "candidate_requests_per_round": float(
                self.candidate_requests_rounds / rounds
            ),
            "unique_candidates_per_round": float(
                self.unique_candidates_rounds / rounds
            ),
            "coordinate_sweeps_per_round": float(
                self.coordinate_sweeps_rounds / rounds
            ),
            "predicted_dpp_cost_mean": float(
                self.predicted_dpp_cost / rounds
            ),
            "realized_dpp_cost_mean": float(
                self.realized_dpp_cost / rounds
            ),
            "prediction_error_mean": float(
                self.prediction_error / rounds
            ),
            "prediction_mae": float(
                self.prediction_abs_error / rounds
            ),
        }

        for index, value in enumerate(
            self.fast_layer_ratios,
            start=1,
        ):
            result[f"fast_layer_{index}_ratio"] = float(
                value / slots
            )

        return result
