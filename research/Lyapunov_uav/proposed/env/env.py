from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np

try:
    from proposed.config import EnvConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import EnvConfig

from .action_types import EnvAction, SlowAction, FastAction
from .validators import parse_slow_action, parse_fast_action
from .channel import RSUChannelModel, UAVChannelModel
from .delivery.rsu_delivery import compute_rsu_delivery
from .delivery.uav_delivery import compute_uav_delivery
from .battery import UAVBattery
from .battery.battery_types import BatteryStepInfo, UAVBatteryMode
from .battery.energy_model import compute_energy_summary
from .battery.queue_model import check_outage, update_soc_virtual_q


class Env:
    """
    Modified Joint Lyapunov Optimization Scenario용 전체 환경 클래스

    Slow-timescale (round level):
        1) RSU scheduling y_mn(r)
        2) UAV hiring mu_m(r)
        3) UAV scheduling phi_un(r)

    Fast-timescale (slot level):
        1) RSU/UAV chunk, layer delivery
        2) UAV power allocation p_un(t)
        3) UAV charging/service mode I_u(t)
    """
    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)
        
        # 시스템 설정
        # user, uav, rsu 수 정의
        self.num_rsu = int(self.cfg.num_rsu)
        self.num_user = int(self.cfg.num_user)
        self.num_uav = int(self.cfg.num_uav)
        self.slow_T = int(self.cfg.slow_T)

        # 채널 객체
        self.rsu_channel = RSUChannelModel(self.cfg.rsu_channel)
        self.uav_channel = UAVChannelModel(self.cfg.uav_channel)

        # 배터리 객체
        self.batteries = [
            UAVBattery(
                config=self.cfg.battery,
                bandwidth=float(self.cfg.uav_channel.bandwidth),
                consume_hover_when_idle=True,
            )
            for _ in range(self.num_uav)
        ]

        # runtime state
        self.t = 0
        self.episode = 0
        self.round_idx = 0
        self.round_slot = 0

        # user video queue
        self.queue = np.zeros(self.num_user, dtype=np.float32)

        # coverage-region FSMC mobility state
        # region index: 0 = leftmost region, num_rsu - 1 = rightmost region
        # users move only to the left; when a user exits region 0, the same user slot
        # is wrapped to region num_rsu - 1 and treated as a newly entering user.
        self.user_region = np.zeros(self.num_user, dtype=np.int32)

        # distance states used by channel/delivery modules
        self.rsu_user_distance = self._default_rsu_user_distance()
        self.uav_user_distance = self._default_uav_user_distance()

        # slow-timescale decisions
        # rsu/uav의 스케줄링 및 uav 고용
        self.rsu_scheduling = np.zeros((self.num_rsu, self.num_user), dtype=np.int32)
        self.uav_hiring = np.zeros(self.num_uav, dtype=np.int32)
        self.uav_scheduling = np.zeros((self.num_uav, self.num_user), dtype=np.int32)

        # user/content 정보
        self.requested_content = np.zeros(self.num_user, dtype=np.int32)
        self.uav_cached_content = np.zeros(self.num_uav, dtype=np.int32)

        # UAV 상태
        self.outage = np.zeros(self.num_uav, dtype=np.int32)
        self.charging_state = np.zeros(self.num_uav, dtype=np.int32)
        self.charge_counters = np.zeros(self.num_uav, dtype=np.int32)

        # round energy
        self.round_start_E = np.zeros(self.num_uav, dtype=np.float32)

        # round-level reward accumulator
        self.round_fast_reward_sum = 0.0
        self.round_quality_sum = 0.0
        self.round_delivery_sum = 0.0
        self.round_stall_sum = 0.0
        self.round_battery_consume_sum = 0.0
        self.round_battery_charge_sum = 0.0

    # ------------------------------------------------------------------
    # default distance
    # ------------------------------------------------------------------
    def _default_rsu_user_distance(self) -> np.ndarray:
        return np.full(
            (self.num_rsu, self.num_user),
            max(float(self.cfg.rsu_channel.distance), float(self.cfg.rsu_channel.min_distance)),
            dtype=np.float32,
        )

    def _default_uav_user_distance(self) -> np.ndarray:
        return np.full(
            (self.num_uav, self.num_user),
            max(float(self.cfg.uav_channel.distance), float(self.cfg.uav_channel.min_distance)),
            dtype=np.float32,
        )
    
    def _sample_distance_matrix(
        self,
        channel_cfg,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """
        fallback slot-level link distance state 샘플링.

        현재 기본 mobility는 coverage-region FSMC를 사용한다.
        다만 cfg.use_region_fsmc_mobility=False로 둘 경우 기존 uniform/fixed
        distance sampling을 사용할 수 있도록 fallback을 유지한다.
        """
        sampling = str(getattr(channel_cfg, "distance_sampling", "fixed")).lower().strip()

        if sampling == "uniform":
            low = float(getattr(channel_cfg, "distance_min", channel_cfg.distance))
            high = float(getattr(channel_cfg, "distance_max", channel_cfg.distance))
            min_distance = float(getattr(channel_cfg, "min_distance", 1.0))

            sampled = self.rng.uniform(low=low, high=high, size=shape)
            sampled = np.maximum(sampled, min_distance)
            return sampled.astype(np.float32)

        return np.full(
            shape,
            max(float(channel_cfg.distance), float(channel_cfg.min_distance)),
            dtype=np.float32,
        )

    def _use_region_fsmc_mobility(self) -> bool:
        """
        coverage-region FSMC mobility 사용 여부.

        현재 formulation 기준 user mobility는 slot마다 확률 p로 다음 coverage
        region으로 넘어가는 FSMC로 정의하는 것이 맞으므로, 별도 설정이 없으면
        True를 기본값으로 둔다.
        """
        return bool(getattr(self.cfg, "use_region_fsmc_mobility", True))

    def _region_transition_prob(self) -> np.ndarray:
        """
        각 user가 한 slot 동안 왼쪽 coverage region으로 이동할 확률 p_n 반환.

        EnvConfig에 다음 이름 중 하나가 있으면 사용한다.
            - region_transition_prob
            - mobility_transition_prob
            - user_move_prob
            - p_region
            - p_move

        scalar이면 모든 user에게 동일하게 적용하고, 길이 num_user의 vector이면
        user별 transition probability로 적용한다.
        """
        value = None
        for name in (
            "region_transition_prob",
            "mobility_transition_prob",
            "user_move_prob",
            "p_region",
            "p_move",
        ):
            if hasattr(self.cfg, name):
                value = getattr(self.cfg, name)
                break

        if value is None:
            value = 0.05

        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full(self.num_user, float(arr), dtype=np.float32)
        elif arr.shape != (self.num_user,):
            raise ValueError(
                f"region transition probability must be scalar or shape "
                f"({self.num_user},), got {arr.shape}"
            )

        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def _region_spacing_m(self) -> float:
        """
        인접 coverage-region 대표 좌표 간 거리.

        현재 formulation은 연속 좌표를 직접 추적하지 않고 coverage region의
        대표 ground coordinate를 사용한다. 별도 config가 없으면 channel distance
        range의 상한값을 region 간격으로 사용한다.
        """
        default_spacing = float(getattr(self.cfg.rsu_channel, "distance_max", 25.0))
        spacing = float(getattr(self.cfg, "region_spacing_m", default_spacing))
        return max(spacing, float(getattr(self.cfg.rsu_channel, "min_distance", 1.0)))

    def _same_region_distance_m(self) -> float:
        """
        user와 RSU/UAV가 같은 coverage region에 있을 때 사용할 horizontal distance.

        대표 좌표만 사용하면 같은 region에서 거리가 0이 될 수 있으므로,
        channel singularity를 피하기 위해 distance_min을 기본값으로 둔다.
        """
        default_distance = float(getattr(self.cfg.rsu_channel, "distance_min", 5.0))
        distance = float(getattr(self.cfg, "same_region_distance_m", default_distance))
        min_distance = float(getattr(self.cfg.rsu_channel, "min_distance", 1.0))
        return max(distance, min_distance)

    def _sample_initial_user_regions(self) -> np.ndarray:
        """
        episode reset 시 user의 초기 coverage region을 샘플링.

        N0는 하나의 RSU coverage region 내 최대 user 수로 해석한다.
        num_user <= num_rsu * N0이면 region별 capacity를 넘지 않도록 샘플링하고,
        그렇지 않으면 fallback으로 uniform sampling을 사용한다.
        """
        cap_per_region = max(1, int(getattr(self.cfg, "N0", self.num_user)))
        candidate_regions = np.repeat(np.arange(self.num_rsu, dtype=np.int32), cap_per_region)

        if candidate_regions.size >= self.num_user:
            sampled = self.rng.choice(candidate_regions, size=self.num_user, replace=False)
        else:
            sampled = self.rng.integers(
                low=0,
                high=self.num_rsu,
                size=self.num_user,
                dtype=np.int32,
            )

        return sampled.astype(np.int32)

    def _sample_requested_content_for_users(self, user_indices: np.ndarray) -> np.ndarray:
        """
        주어진 user index들에 대해 신규 requested content를 Zipf 기반으로 샘플링.
        """
        user_indices = np.asarray(user_indices, dtype=np.int32)
        if user_indices.size == 0:
            return np.zeros(0, dtype=np.int32)

        video_ids = np.arange(1, self.cfg.num_video + 1, dtype=np.float64)
        probs = 1.0 / np.power(video_ids, float(self.cfg.zipf_alpha))
        probs = probs / probs.sum()
        sampled = self.rng.choice(self.cfg.num_video, size=user_indices.size, p=probs)
        return sampled.astype(np.int32)

    def _compute_distance_matrix_from_regions(self, channel_cfg, num_servers: int) -> np.ndarray:
        """
        coverage region 대표 좌표 기반 link distance matrix 계산.

        server index와 region index를 같은 coverage region index로 매핑한다.
        UAV는 trajectory optimization 없이 해당 region에서 hovering한다고 가정하므로,
        UAV-user horizontal distance도 동일한 region distance로 계산한다.
        UAV channel model 내부에서 altitude가 추가되어 3D LoS gain이 계산된다.
        """
        server_region = np.arange(num_servers, dtype=np.int32)
        user_region = self.user_region.astype(np.int32)

        region_gap = np.abs(server_region[:, None] - user_region[None, :]).astype(np.float32)
        distance = self._same_region_distance_m() + region_gap * self._region_spacing_m()

        min_distance = float(getattr(channel_cfg, "min_distance", 1.0))
        distance = np.maximum(distance, min_distance)

        return distance.astype(np.float32)

    def _active_user_sets_by_region(self) -> Dict[int, np.ndarray]:
        """
        현재 slot 기준 각 coverage region에 존재하는 active user set 반환.
        """
        return {
            int(m): np.flatnonzero(self.user_region == int(m)).astype(np.int32)
            for m in range(self.num_rsu)
        }

    def _region_transition_matrix(self) -> np.ndarray:
        """
        scalar p 기준 coverage-region FSMC transition matrix 반환.

        region index 0은 가장 왼쪽 region이다.
        user는 왼쪽으로만 이동하므로 i > 0에서는 i -> i-1로 이동하고,
        i = 0에서 이동이 발생하면 오른쪽 끝 region num_rsu-1로 재진입한다.
        """
        p_arr = self._region_transition_prob()
        p = float(np.mean(p_arr))

        P = np.zeros((self.num_rsu, self.num_rsu), dtype=np.float32)
        for region in range(self.num_rsu):
            P[region, region] = 1.0 - p
            next_region = region - 1 if region > 0 else self.num_rsu - 1
            P[region, next_region] = p

        return P

    def _apply_user_region_fsmc(self) -> Dict[str, Any]:
        """
        p 확률로 user가 왼쪽 coverage region으로 이동하는 FSMC transition 수행.

        - user_region[n] > 0에서 move가 발생하면 user_region[n] -= 1
        - user_region[n] == 0에서 move가 발생하면 해당 user slot은 시스템을 이탈하고,
          오른쪽 끝 region num_rsu-1에서 새로운 user로 재진입한 것으로 처리
        - 재진입 user는 queue를 init_queue로 초기화하고 requested_content를 새로 샘플링
        """
        prev_region = self.user_region.copy()
        p_move = self._region_transition_prob()
        move_mask = self.rng.random(self.num_user) < p_move

        wrapped_mask = move_mask & (prev_region <= 0)

        next_region = prev_region.copy()
        next_region[move_mask] -= 1
        next_region[wrapped_mask] = self.num_rsu - 1
        next_region = np.clip(next_region, 0, self.num_rsu - 1).astype(np.int32)

        self.user_region = next_region

        wrapped_users = np.flatnonzero(wrapped_mask).astype(np.int32)
        if wrapped_users.size > 0:
            self.queue[wrapped_users] = float(self.cfg.init_queue)
            self.requested_content[wrapped_users] = self._sample_requested_content_for_users(
                wrapped_users
            )

        active_user_sets = self._active_user_sets_by_region()

        return {
            "mobility_model": "coverage_region_fsmc",
            "transition_prob": p_move.copy(),
            "prev_user_region": prev_region.copy(),
            "move_mask": move_mask.astype(np.int32),
            "wrapped_mask": wrapped_mask.astype(np.int32),
            "wrapped_users": wrapped_users.copy(),
            "next_user_region": self.user_region.copy(),
            "active_user_sets_by_region": active_user_sets,
            "region_transition_matrix": self._region_transition_matrix(),
        }

    def _refresh_link_distances(self) -> None:
        """
        다음 slot에서 관측/사용할 RSU-user, UAV-user distance state 갱신.

        기본값은 coverage-region FSMC mobility에서 계산된 user_region 기반 거리이다.
        cfg.use_region_fsmc_mobility=False이면 기존 uniform/fixed random distance
        sampling으로 fallback한다.
        """
        if self._use_region_fsmc_mobility():
            self.rsu_user_distance = self._compute_distance_matrix_from_regions(
                self.cfg.rsu_channel,
                self.num_rsu,
            )
            self.uav_user_distance = self._compute_distance_matrix_from_regions(
                self.cfg.uav_channel,
                self.num_uav,
            )
            return

        self.rsu_user_distance = self._sample_distance_matrix(
            self.cfg.rsu_channel,
            (self.num_rsu, self.num_user),
        )
        self.uav_user_distance = self._sample_distance_matrix(
            self.cfg.uav_channel,
            (self.num_uav, self.num_user),
        )

    def reset(self) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        에피소드 초기화 수행 함수로
        큐, 배터리, 위치 등 State 변수 초기화
        """
        # time slot 및 round 초기화 / 동시에 에피소드는 1 증가
        self.t = 0
        self.episode += 1
        self.round_idx = 0
        self.round_slot = 0

        # 사용자 큐 초기화
        self.queue = np.full(self.num_user, float(self.cfg.init_queue), dtype=np.float32)

        # slow-timescale decision 초기화
        self.rsu_scheduling = np.zeros(
            (self.num_rsu, self.num_user), dtype=np.int32
        )
        self.uav_hiring = np.zeros(self.num_uav, dtype=np.int32)
        self.uav_scheduling = np.zeros(
            (self.num_uav, self.num_user), dtype=np.int32
        )

        # content 초기화
        self.requested_content = self._sample_requested_content()
        self.uav_cached_content = self._sample_uav_cached_content()

        # coverage-region FSMC mobility 초기화
        self.user_region = self._sample_initial_user_regions()

        # slot 0에서 policy가 관측할 link distance state
        self._refresh_link_distances()

        # battery 초기화
        for battery in self.batteries:
            battery.reset_episode()
            battery.start_round(round_horizon=self.slow_T)

        self.outage = np.zeros(self.num_uav, dtype=np.int32)
        self.charging_state = np.zeros(self.num_uav, dtype=np.int32)
        self.charge_counters = np.zeros(self.num_uav, dtype=np.float32)

        self.round_start_E = self.E.copy()
        self._reset_round_reward_accumulators()

        obs = self.get_fast_obs()
        info: Dict[str, Any] = {
            "episode": int(self.episode),
            "time": int(self.t),
            "round_idx": int(self.round_idx),
            "round_slot": int(self.round_slot),
            "reset": True,
            "obs_type": "fast_obs",
        }

        return obs, info
    
    def _reset_round_reward_accumulators(self):
        """
        slow-timescale reward 계산을 위한 round-level accumulator 초기화
        (slot별 fast reward component을 누적)
        """
        self.round_fast_reward_sum = 0.0
        self.round_quality_sum = 0.0
        self.round_delivery_sum = 0.0
        self.round_stall_sum = 0.0
        self.round_battery_consume_sum = 0.0
        self.round_battery_charge_sum = 0.0

    def _theta_z(self) -> np.ndarray:
        """
        user-side perturbed Lyapunov Target 반환
        """
        theta_z = getattr(self.cfg, "theta_z", None)
        if theta_z is None:
            raise ValueError("EnvConfig에서 theta_z의 값을 먼저 확정하세요.")
        else:
            theta_arr = np.asarray(theta_z, dtype=np.float32)

        if theta_arr.shape != (self.num_user, ):
            raise ValueError(
                f"theta_z는 ({self.num_user}, )의 shape을 가져야 합니다, 현재는 {theta_arr.shape} shape입니다."
            )
        
        return np.clip(theta_arr, 0.0, float(self.cfg.max_queue)).astype(np.float32)
    
    def _hire_cost(self) -> np.ndarray:
        """
        UAV 1대당 hiring cost 반환
        """
        value = getattr(self.cfg, "uav_hiring_cost", None)

        arr = np.asarray(value, dtype=np.float32)

        if arr.ndim == 0:
            return np.full(self.num_uav, float(arr), dtype=np.float32)

        if arr.shape != (self.num_uav,):
            raise ValueError(
                f"uav_hiring_cost must have shape ({self.num_uav},), got {arr.shape}"
            )

        return arr.astype(np.float32)
    
    @property
    def E(self) -> np.ndarray:
        """
        UAV Actual SoC Queue
        """
        return np.array([b.soc for b in self.batteries], dtype=np.float32)
    
    @property
    def Y(self) -> np.ndarray:
        """
        UAV Virtual Soc Queue
        """
        return np.array([b.virtual_q for b in self.batteries], dtype=np.float32)
    
    @property
    def Z(self) -> np.ndarray:
        """
        User virtual video queue
        """
        return np.clip(
            float(self.cfg.max_queue) - self.queue,
            0.0,
            float(self.cfg.max_queue),
        ).astype(np.float32)
    
    def _sample_requested_content(self) -> np.ndarray:
        """
        사용자가 요청하는 content 샘플링하는 함수로,
        Zipf 기반으로 [0, num_video-1] 중 하나를 선택함.
        """
        video_ids = np.arange(1, self.cfg.num_video+1, dtype=np.float64)
        probs = 1.0 / np.power(video_ids, float(self.cfg.zipf_alpha))
        probs = probs / probs.sum()
        sampled = self.rng.choice(self.cfg.num_video, size=self.num_user, p=probs)
        return sampled.astype(np.int32)
    
    def _sample_uav_cached_content(self) -> np.ndarray:
        """
        UAV cache content 초기화하는 함수로,
        현재는 각 UAV가 하나의 content만 cache 가능하다고 보고 균등 샘플링 적용
        """
        sampled = self.rng.integers(
            low=0,
            high=self.cfg.num_video,
            size=self.num_uav,
            dtype=np.int32,
        )
        return sampled.astype(np.int32)
    
    def _start_new_round(self) -> None:
        """
        라운드 초기화 함수
        """
        self.round_idx = self.t // self.slow_T
        self.round_slot = 0
        self.round_start_E = self.E.copy()

        for battery in self.batteries:
            battery.start_round(round_horizon=self.slow_T)

        self._reset_round_reward_accumulators()

    def _rule_based_uav_charge(self) -> np.ndarray:
        """
        fast action에 uav_charge가 명시되지 않은 경우 사용할 rule-based charging hook.

        현재 1단계에서는 SoC가 e_min 이하이고 고용된 UAV만 charging 후보로 둔다.
        """
        if not bool(self.cfg.battery.allow_charge):
            return np.zeros(self.num_uav, dtype=np.int32)
        if not bool(self.cfg.battery.enable_charging):
            return np.zeros(self.num_uav, dtype=np.int32)

        low_soc = (self.E <= float(self.cfg.battery.e_min)).astype(np.int32)
        return (low_soc * self.uav_hiring).astype(np.int32)

    def _clear_charging_service_actions(self, fast_act: FastAction) -> None:
        for uav_idx in range(self.num_uav):
            if int(fast_act.uav_charge[uav_idx]) != 1:
                continue
            fast_act.uav_chunks[uav_idx, :] = 0
            fast_act.uav_layers[uav_idx, :] = 0
            fast_act.uav_power[uav_idx, :] = 0.0

    def _compute_fast_reward(
        self,
        prev_Q: np.ndarray,
        next_Q: np.ndarray,
        prev_Z: np.ndarray,
        next_Z: np.ndarray,
        prev_E: np.ndarray,
        next_E: np.ndarray,
        prev_Y: np.ndarray,
        next_Y: np.ndarray,
        delivered_total_per_user: np.ndarray,
        quality_total_per_user: np.ndarray,
        playback: np.ndarray,
        uav_hiring: np.ndarray,
        charging_state: np.ndarray,
        battery_step_info: list[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Fast-timescale perturbed DPP reward

        Z_n(t)는 theta_z 근처로 유지하고, battery virtual queue는 service
        feasibility pressure로만 반영한다. 실제 PPO 연결은 별도 adapter에서
        수행한다.
        """
        theta_z = self._theta_z()

        prev_Q_arr = np.asarray(prev_Q, dtype=np.float32)
        next_Q_arr = np.asarray(next_Q, dtype=np.float32)
        prev_Z_arr = np.asarray(prev_Z, dtype=np.float32)
        next_Z_arr = np.asarray(next_Z, dtype=np.float32)
        prev_E_arr = np.asarray(prev_E, dtype=np.float32)
        next_E_arr = np.asarray(next_E, dtype=np.float32)
        prev_Y_arr = np.asarray(prev_Y, dtype=np.float32)
        next_Y_arr = np.asarray(next_Y, dtype=np.float32)

        delivered_arr = np.asarray(delivered_total_per_user, dtype=np.float32)
        quality_arr = np.asarray(quality_total_per_user, dtype=np.float32)

        consumed_soc_arr = np.array(
            [
                float(info.get("consumed_soc", 0.0))
                for info in battery_step_info
            ],
            dtype=np.float32,
        )
        charged_soc_arr = np.array(
            [
                float(info.get("charged_soc", 0.0))
                for info in battery_step_info
            ],
            dtype=np.float32,
        )

        playback_arr = np.asarray(playback, dtype=np.float32)

        # ------------------------------------------------------------------
        # Scaled fast-timescale DPP reward
        # ------------------------------------------------------------------
        # formulation의 slot-level DPP cost:
        #   alpha_Z * sum_n (Z_n(t)-theta_n^Z) * (b_n(t)-D_n(t))
        # + alpha_B * sum_u B_u(t) * (e_u(t)-e_u^c(t))
        # - V * sum_n q_n(t)
        #
        # PPO는 reward를 maximize하므로, 위 cost의 음수를 reward로 사용:
        #   alpha_Z * sum_n (Z_n(t)-theta_n^Z) * (D_n(t)-b_n(t))
        # - alpha_B * sum_u B_u(t) * (e_u(t)-e_u^c(t))
        # + V * sum_n q_n(t)
        #
        # 여기서 consumed_soc_arr / charged_soc_arr는 이미 SoC 단위이다.
        V = float(getattr(self.cfg, "V", 1.0))
        alpha_z = float(
            getattr(
                self.cfg,
                "alpha_z",
                getattr(self.cfg, "user_queue_scale", 1.0),
            )
        )
        alpha_b = float(
            getattr(
                self.cfg,
                "alpha_b",
                getattr(self.cfg, "battery_queue_scale", 1.0),
            )
        )

        reward_preset = str(getattr(self.cfg, "preset_name", "scaled_fast_dpp"))

        if hasattr(self.cfg, "reward_coefficients") and callable(self.cfg.reward_coefficients):
            reward_coefficients = self.cfg.reward_coefficients()
        else:
            reward_coefficients = {
                "alpha_z": alpha_z,
                "alpha_b": alpha_b,
                "V": V,
            }

        video_delivery_term = alpha_z * float(np.sum((prev_Z_arr - theta_z) * delivered_arr))
        video_playback_term = -alpha_z * float(np.sum((prev_Z_arr - theta_z) * playback_arr))
        video_drift_reward_term = video_delivery_term + video_playback_term

        battery_consume_term = -alpha_b * float(np.sum(prev_Y_arr * consumed_soc_arr))
        battery_charge_term = alpha_b * float(np.sum(prev_Y_arr * charged_soc_arr))
        battery_drift_reward_term = battery_consume_term + battery_charge_term

        quality_term = V * float(np.sum(quality_arr))

        fast_reward = (
            video_drift_reward_term
            + battery_drift_reward_term
            + quality_term
        )

        dpp_cost = -float(fast_reward)

        components: Dict[str, Any] = {
            "reward_preset": reward_preset,
            "reward_coefficients": reward_coefficients,
            "theta_z": theta_z.copy(),

            "prev_Q": prev_Q_arr.copy(),
            "next_Q": next_Q_arr.copy(),
            "prev_Z": prev_Z_arr.copy(),
            "next_Z": next_Z_arr.copy(),
            "prev_E": prev_E_arr.copy(),
            "next_E": next_E_arr.copy(),
            "prev_Y": prev_Y_arr.copy(),
            "next_Y": next_Y_arr.copy(),

            "delivered_total_per_user": delivered_arr.copy(),
            "quality_total_per_user": quality_arr.copy(),
            "playback_per_user": playback_arr.copy(),

            "consumed_soc_per_uav": consumed_soc_arr.copy(),
            "charged_soc_per_uav": charged_soc_arr.copy(),

            "alpha_z": float(alpha_z),
            "alpha_b": float(alpha_b),
            "V": float(V),

            "video_delivery_term": float(video_delivery_term),
            "video_playback_term": float(video_playback_term),
            "video_drift_reward_term": float(video_drift_reward_term),
            "battery_consume_term": float(battery_consume_term),
            "battery_charge_term": float(battery_charge_term),
            "battery_drift_reward_term": float(battery_drift_reward_term),
            "quality_term": float(quality_term),

            "slot_dpp_cost": float(dpp_cost),
            "fast_reward": float(fast_reward),

            "num_hired_uav": int(np.asarray(uav_hiring, dtype=np.int32).sum()),
            "num_charging_uav": int(np.asarray(charging_state, dtype=np.int32).sum()),
            "num_outage_uav": int(np.asarray(self.outage, dtype=np.int32).sum()),

            "sum_delivery": float(np.sum(delivered_arr)),
            "sum_quality": float(np.sum(quality_arr)),
            "sum_playback": float(np.sum(playback_arr)),
            "sum_consumed_soc": float(np.sum(consumed_soc_arr)),
            "sum_charged_soc": float(np.sum(charged_soc_arr)),
        }
        return float(fast_reward), components
    
    def _compute_slow_reward(
        self,
        is_round_boundary: bool,
        fast_reward: float,
        reward_components: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Slow-timescale reward

        - round가 끝난 뒤 round_info를 모아서 계산해야 한다.
        """
        self.round_fast_reward_sum += float(fast_reward)
        self.round_quality_sum += float(reward_components.get("sum_quality", 0.0))
        self.round_delivery_sum += float(reward_components.get("sum_delivery", 0.0))
        self.round_battery_consume_sum += float(reward_components.get("battery_consume_term", 0.0))
        self.round_battery_charge_sum += float(reward_components.get("battery_charge_term", 0.0))

        hire_cost_per_uav = self._hire_cost()
        hire_cost_raw = float(np.sum(np.asarray(self.uav_hiring, dtype=np.float32) * hire_cost_per_uav))
        hire_cost = -hire_cost_raw

        if not is_round_boundary:
            components = {
                "is_round_boundary": False,
                "slow_reward": 0.0,
                "round_fast_reward_sum_so_far": float(self.round_fast_reward_sum),
                "round_quality_sum_so_far": float(self.round_quality_sum),
                "round_delivery_sum_so_far": float(self.round_delivery_sum),
                "round_battery_consume_sum_so_far": float(self.round_battery_consume_sum),
                "round_battery_charge_sum_so_far": float(self.round_battery_charge_sum),
                "hire_cost_raw": float(hire_cost_raw),
                "hire_cost": float(hire_cost),
            }
            return 0.0, components
        
        slow_reward = float(hire_cost) + float(self.round_fast_reward_sum)
        components = {
            "is_round_boundary": True,
            "slow_reward": float(slow_reward),

            "round_fast_reward_sum": float(self.round_fast_reward_sum),
            "round_quality_sum": float(self.round_quality_sum),
            "round_delivery_sum": float(self.round_delivery_sum),
            "round_battery_consume_sum": float(
                self.round_battery_consume_sum
            ),
            "round_battery_charge_sum": float(
                self.round_battery_charge_sum
            ),

            "uav_hiring": self.uav_hiring.copy(),
            "hire_cost_per_uav": hire_cost_per_uav.copy(),
            "hire_cost_raw": float(hire_cost_raw),
            "hire_cost": float(hire_cost),
        }

        return float(slow_reward), components
    
    def _compute_reward(
        self,
        prev_Q: np.ndarray,
        next_Q: np.ndarray,
        prev_Z: np.ndarray,
        next_Z: np.ndarray,
        prev_E: np.ndarray,
        next_E: np.ndarray,
        prev_Y: np.ndarray,
        next_Y: np.ndarray,
        delivered_total_per_user: np.ndarray,
        quality_total_per_user: np.ndarray,
        playback: np.ndarray,
        uav_hiring: np.ndarray,
        charging_state: np.ndarray,
        battery_step_info: list[Dict[str, Any]],
        is_round_boundary: bool,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        env.step()에서 호출하는 통합 함수

        현재 반환하는 reward는 fast PPO 학습에 바로 쓰기 위해 fast_reward로 설정.
        slow_reward는 info["slow_reward_components"]에 넣고, HRL high-level trainer가 별도로 읽어 쓰는 구조를 선택함.
        """
        fast_reward, fast_components = self._compute_fast_reward(
            prev_Q=prev_Q,
            next_Q=next_Q,
            prev_Z=prev_Z,
            next_Z=next_Z,
            prev_E=prev_E,
            next_E=next_E,
            prev_Y=prev_Y,
            next_Y=next_Y,
            delivered_total_per_user=delivered_total_per_user,
            quality_total_per_user=quality_total_per_user,
            playback=playback,
            uav_hiring=uav_hiring,
            charging_state=charging_state,
            battery_step_info=battery_step_info,
        )

        slow_reward, slow_components = self._compute_slow_reward(
            is_round_boundary=is_round_boundary,
            fast_reward=fast_reward,
            reward_components=fast_components,
        )

        components: Dict[str, Any] = {
            "fast_reward": float(fast_reward),
            "slow_reward": float(slow_reward),
            "fast_reward_components": fast_components,
            "slow_reward_components": slow_components,
        }

        return float(fast_reward), components

    def apply_slow_action(self, action: EnvAction) -> SlowAction:
        """
        slow-timescale decision을 갱신하는 함수
        """
        slow_act = parse_slow_action(action, self.cfg)

        self.rsu_scheduling = slow_act.rsu_scheduling.copy()
        self.uav_hiring = slow_act.uav_hiring.copy()
        self.uav_scheduling = slow_act.uav_scheduling.copy()

        self._start_new_round()
        return slow_act

    def _build_effective_fast_action(self, fast_act: FastAction) -> FastAction:
        """
        slow-timescale decision은 round 동안 고정됨.
            - fast action은 chunk/layer 수, allocated power만 제어
            - residual user는 slow policy가 UAV candidate로 해석한 user로 해석
        """
        residual_users = (self.uav_scheduling.sum(axis=0) > 0).astype(np.int32)

        # 유효한 action 정의
        effective_uav_chunks = fast_act.uav_chunks.copy()
        effective_uav_layers = fast_act.uav_layers.copy()
        effective_uav_power = fast_act.uav_power.copy()

        effective_rsu_chunks = fast_act.rsu_chunks.copy()
        effective_rsu_layers = fast_act.rsu_layers.copy()

        # round-level hiring이 꺼져 있는 UAV는 fast action 제거
        inactive_uav_mask = (self.uav_hiring <= 0)
        effective_uav_chunks[inactive_uav_mask, :] = 0
        effective_uav_layers[inactive_uav_mask, :] = 0
        effective_uav_power[inactive_uav_mask, :] = 0.0

        # round-level uav scheduling이 없는 링크는 fast action 제거
        effective_uav_chunks = effective_uav_chunks * self.uav_scheduling
        effective_uav_layers = effective_uav_layers * self.uav_scheduling
        effective_uav_power = effective_uav_power * self.uav_scheduling.astype(np.float32)

        # round-level rsu scheduling 없으면 fast rsu action 제거
        effective_rsu_chunks = effective_rsu_chunks * self.rsu_scheduling
        effective_rsu_layers = effective_rsu_layers * self.rsu_scheduling

        # residual user가 아니면 UAV service 제거
        effective_uav_chunks = effective_uav_chunks * residual_users[None, :]
        effective_uav_layers = effective_uav_layers * residual_users[None, :]
        effective_uav_power = effective_uav_power * residual_users[None, :].astype(np.float32)

        return FastAction(
            rsu_chunks=effective_rsu_chunks,
            rsu_layers=effective_rsu_layers,
            uav_chunks=effective_uav_chunks,
            uav_layers=effective_uav_layers,
            uav_power=effective_uav_power,
            uav_charge=fast_act.uav_charge.copy(),
            playback=fast_act.playback.copy(),
            rsu_user_distance=fast_act.rsu_user_distance.copy(),
            uav_user_distance=fast_act.uav_user_distance.copy(),
            residual_users=residual_users.astype(np.int32),
            user_virtual_queue=self.Z.copy(),
            requested_content=self.requested_content.copy(),
            uav_cached_content=self.uav_cached_content.copy(),
        )
    
    def _apply_battery_transition(
        self,
        uav_idx: int,
        mu_active: bool,
        mode: UAVBatteryMode,
        links,
    ) -> Dict[str, Any]:
        """
        energy_model 및 queue_model 기준으로 동일 의미 transition 수행하는 함수
        """
        battery = self.batteries[uav_idx]

        soc_before = float(battery.soc)
        virtual_before = float(battery.virtual_q)

        energy_info = compute_energy_summary(
            config=self.cfg.battery,
            mode=mode,
            mu_active=bool(mu_active),
            links=links,
            consume_hover_when_idle=battery.consume_hover_when_idle,
        )
        
        consumed_soc, charged_soc, next_soc, next_virtual_q = update_soc_virtual_q(
            config=self.cfg.battery,
            soc=battery.soc,
            consumed_energy=float(energy_info["total_energy"]),
            charged_energy=float(energy_info["charge_energy"]),
        )

        battery.soc = float(next_soc)
        battery.virtual_q = float(next_virtual_q)
        battery.round_remaining_slots = max(0, int(battery.round_remaining_slots) - 1)

        is_outage = bool(
            check_outage(
                soc=battery.soc,
                consumed_soc=float(consumed_soc),
                soc_before=float(soc_before),
            )
        )
        self.outage[uav_idx] = int(is_outage)

        step_info = BatteryStepInfo(
            hover_energy=float(energy_info["hover_energy"]),
            comm_energy=float(energy_info["comm_energy"]),
            total_consumed=float(energy_info["total_energy"]),
            charged_energy=float(energy_info["charge_energy"]),
            consumed_soc=float(consumed_soc),
            charged_soc=float(charged_soc),
            soc_before=float(soc_before),
            soc_after=float(battery.soc),
            virtual_before=float(virtual_before),
            virtual_after=float(battery.virtual_q),
            outage=bool(is_outage),
        )

        return asdict(step_info)
    
    def get_slow_obs(self) -> Dict[str, np.ndarray]:
        """
        slow-timescale 상태값을 반환하는 함수
        """
        return {
            "Z": self.Z.copy(),
            "Y": self.Y.copy(),
            "rsu_user_distance": self.rsu_user_distance.copy(),
            "uav_user_distance": self.uav_user_distance.copy(),
            "user_region": self.user_region.copy(),
        }

    def get_fast_obs(self) -> Dict[str, np.ndarray]:
        """
        fast-timescale 상태값을 반환하는 함수.

        현재 시나리오 기준 fast policy는 slot t마다 다음을 관찰한다.
            - user video virtual queue Z(t)
            - UAV battery virtual queue B(t)
            - user 위치/거리
            - UAV-user scheduling decision phi(r)

        실제 fast action은 chunk/layer delivery와 UAV power allocation이며,
        UAV charging은 현재 구현에서는 rule-based service feasibility hook으로 처리한다.
        """
        return {
            "Z": self.Z.copy(),
            "Y": self.Y.copy(),
            "uav_scheduling": self.uav_scheduling.copy(),
            "rsu_user_distance": self.rsu_user_distance.copy(),
            "uav_user_distance": self.uav_user_distance.copy(),
            "user_region": self.user_region.copy(),
        }

    def step(self, action: EnvAction) -> tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        환경의 1-slot 진행 함수로, Fast-timescale 진행을 담당.
        """
        fast_act = parse_fast_action(action, self.cfg)

        # distance는 action이 아니라 현재 slot state이므로,
        # action parser가 만든/default distance를 무시하고 env state를 강제로 사용한다.
        fast_act.rsu_user_distance = self.rsu_user_distance.copy()
        fast_act.uav_user_distance = self.uav_user_distance.copy()

        fast_act.uav_charge = self._rule_based_uav_charge()
        self._clear_charging_service_actions(fast_act)

        fast_act_eff = self._build_effective_fast_action(fast_act)

        slow_act = SlowAction(
            rsu_scheduling=self.rsu_scheduling.copy(),
            uav_hiring=self.uav_hiring.copy(),
            uav_scheduling=self.uav_scheduling.copy(),
        )

        prev_t = int(self.t)
        prev_round_idx = int(self.round_idx)
        prev_round_slot = int(self.round_slot)

        prev_E = self.E.copy()
        prev_Y = self.Y.copy()
        prev_Q = self.queue.copy()
        prev_Z = self.Z.copy()

        # RSU delivery
        rsu_result = compute_rsu_delivery(
            cfg=self.cfg,
            slow_act=slow_act,
            fast_act=fast_act_eff,
            rsu_channel=self.rsu_channel,
            rng=self.rng,
        )

        # UAV delivery
        battery_soc_before_uav = self.E.copy()
        uav_result = compute_uav_delivery(
            cfg=self.cfg,
            slow_act=slow_act,
            fast_act=fast_act_eff,
            battery_parsed=battery_soc_before_uav,
            uav_channel=self.uav_channel,
            rng=self.rng,
        )

        # Battery transition per UAV
        battery_step_info: list[Dict[str, Any]] = []

        for u in range(self.num_uav):
            mu_active = bool(self.uav_hiring[u])
            charge_flag = bool(fast_act_eff.uav_charge[u])

            if not mu_active:
                mode = UAVBatteryMode.IDLE
                links = []
            elif charge_flag:
                mode = UAVBatteryMode.CHARGE
                links = []
            elif len(uav_result.links_uav[u]) > 0:
                mode = UAVBatteryMode.SERVE
                links = uav_result.links_uav[u]
            elif self.outage[u] == 1:
                mode = UAVBatteryMode.OUTAGE
                links = []
            else:
                mode = UAVBatteryMode.IDLE
                links = []
            
            self.charging_state[u] = int(mode == UAVBatteryMode.CHARGE)
            self.charge_counters[u] += int(mode == UAVBatteryMode.CHARGE)

            battery_info_u = self._apply_battery_transition(
                uav_idx=u,
                mu_active=mu_active,
                mode=mode,
                links=links,
            )
            battery_info_u["mode"] = str(mode.value)
            battery_step_info.append(battery_info_u)
        
        # delivery 총계산
        delivered_rsu_per_user = rsu_result.delivered_per_user.astype(np.float32)
        delivered_uav_per_user = uav_result.delivered_per_user.astype(np.float32)
        delivered_total_per_user = delivered_rsu_per_user + delivered_uav_per_user

        quality_rsu_per_user = rsu_result.quality_per_user.astype(np.float32)
        quality_uav_per_user = uav_result.quality_per_user.astype(np.float32)
        quality_total_per_user = quality_rsu_per_user + quality_uav_per_user

        playback = fast_act_eff.playback.astype(np.float32)
        consumed = np.minimum(self.queue, playback)
        stall = np.maximum(playback - self.queue, 0.0)

        self.queue = np.clip(
            self.queue - consumed + delivered_total_per_user,
            0.0,
            float(self.cfg.max_queue),
        ).astype(np.float32)
        post_delivery_Q = self.queue.copy()
        post_delivery_Z = self.Z.copy()

        # time update
        self.t += 1
        self.round_slot += 1

        is_round_boundary = bool(self.round_slot >= self.slow_T)

        if is_round_boundary:
            next_round_idx = int(self.t // self.slow_T)
            next_round_slot = 0
        else:
            next_round_idx = int(self.t // self.slow_T)
            next_round_slot = int(self.round_slot)

        next_t = int(self.t)

        terminated = False
        truncated = False

        reward, reward_components = self._compute_reward(
            prev_Q=prev_Q,
            next_Q=self.queue,
            prev_Z=prev_Z,
            next_Z=self.Z,
            prev_E=prev_E,
            next_E=self.E,
            prev_Y=prev_Y,
            next_Y=self.Y,
            delivered_total_per_user=delivered_total_per_user,
            quality_total_per_user=quality_total_per_user,
            playback=playback,
            uav_hiring=self.uav_hiring,
            charging_state=self.charging_state,
            battery_step_info=battery_step_info,
            is_round_boundary=is_round_boundary,
        )

        info: Dict[str, Any] = {
            # 1) transition meta
            "prev_time": prev_t,
            "next_time": next_t,
            "prev_round_slot": prev_round_slot,
            "next_round_slot": next_round_slot,
            "active_round_idx": prev_round_idx,
            "next_round_idx": next_round_idx,
            "is_round_boundary": is_round_boundary,
            "terminated": terminated,
            "truncated": truncated,

            # 2) slow-timescale active context
            "uav_hiring": self.uav_hiring.copy(),
            "rsu_scheduling": self.rsu_scheduling.copy(),
            "uav_scheduling": self.uav_scheduling.copy(),
            "requested_content": self.requested_content.copy(),
            "uav_cached_content": self.uav_cached_content.copy(),
            "user_region": self.user_region.copy(),
            "active_user_sets_by_region": self._active_user_sets_by_region(),
            "residual_users": fast_act_eff.residual_users.copy(),
            "uav_charge_effective": fast_act_eff.uav_charge.copy(),

            # 3) queue transition
            "prev_Q": prev_Q.copy(),
            "post_delivery_Q": post_delivery_Q.copy(),
            "post_delivery_Z": post_delivery_Z.copy(),
            "next_Q": self.queue.copy(),
            "prev_Z": prev_Z.copy(),
            "next_Z": self.Z.copy(),
            "playback": playback.copy(),
            "consumed": consumed.copy(),
            "stall": stall.copy(),

            # 4) battery transition
            "prev_E": prev_E.copy(),
            "next_E": self.E.copy(),
            "prev_Y": prev_Y.copy(),
            "next_Y": self.Y.copy(),
            "outage": self.outage.copy(),
            "charging_state": self.charging_state.copy(),
            "battery_step_info": battery_step_info,

            # 5) slot execution result
            "delivered_rsu_per_user": delivered_rsu_per_user.copy(),
            "delivered_uav_per_user": delivered_uav_per_user.copy(),
            "delivered_total_per_user": delivered_total_per_user.copy(),
            "quality_rsu_per_user": quality_rsu_per_user.copy(),
            "quality_uav_per_user": quality_uav_per_user.copy(),
            "quality_total_per_user": quality_total_per_user.copy(),
            "dpp_terms": reward_components,
            "reward_components": reward_components,

            "rsu_result": {
                "requested_mask": rsu_result.requested_mask.copy(),
                "capped_mask": rsu_result.capped_mask.copy(),
                "active_mask": rsu_result.active_mask.copy(),
                "delivered_chunks": rsu_result.delivered_chunks.copy(),
                "delivered_bits": rsu_result.delivered_bits.copy(),
                "delivered_quality": rsu_result.delivered_quality.copy(),
                "raw_channel_gain": rsu_result.raw_channel_gain.copy(),
                "link_capacity_bps": rsu_result.link_capacity_bps.copy(),
            },
            "uav_result": {
                "requested_mask": uav_result.requested_mask.copy(),
                "capped_mask": uav_result.capped_mask.copy(),
                "active_mask": uav_result.active_mask.copy(),
                "delivered_chunks": uav_result.delivered_chunks.copy(),
                "delivered_bits": uav_result.delivered_bits.copy(),
                "delivered_quality": uav_result.delivered_quality.copy(),
                "raw_channel_gain": uav_result.raw_channel_gain.copy(),
                "link_capacity_bps": uav_result.link_capacity_bps.copy(),
                "tx_power": uav_result.tx_power.copy(),
                "charge_mask": uav_result.charge_mask.copy(),
            },
        }

        # --------------------------------------------------------------
        # 실제 env round state 반영
        # --------------------------------------------------------------
        if is_round_boundary:
            self._start_new_round()
        else:
            self.round_idx = int(next_round_idx)
            self.round_slot = int(next_round_slot)

        # 다음 slot으로 넘어가기 전 coverage-region FSMC mobility를 적용한다.
        # reward 계산은 현재 slot의 delivery/battery transition 기준으로 수행하고,
        # mobility는 다음 observation state를 만드는 exogenous transition으로 처리한다.
        if self._use_region_fsmc_mobility():
            mobility_info = self._apply_user_region_fsmc()
        else:
            mobility_info = {
                "mobility_model": "distance_resampling",
                "prev_user_region": self.user_region.copy(),
                "next_user_region": self.user_region.copy(),
            }

        # 다음 slot observation에 들어갈 link distance state를 갱신한다.
        self._refresh_link_distances()

        info["mobility"] = mobility_info
        info["next_user_region"] = self.user_region.copy()
        info["next_active_user_sets_by_region"] = self._active_user_sets_by_region()
        info["next_rsu_user_distance"] = self.rsu_user_distance.copy()
        info["next_uav_user_distance"] = self.uav_user_distance.copy()
        info["next_Q"] = self.queue.copy()
        info["next_Z"] = self.Z.copy()
        
        obs = self.get_fast_obs()

        return obs, float(reward), terminated, truncated, info
