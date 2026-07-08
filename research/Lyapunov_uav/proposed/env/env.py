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
    Two-Timescale RSU-UAV-user video delivery environment.

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

        # region index
        # index: 0 = leftmost, num_rsu - 1 = rightmost
        # user는 왼쪽으로만 이동한다고 가정
        self.user_region = np.zeros(self.num_user, dtype=np.int32)

        # default distance states
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
    
    def _hire_cost(self) -> np.ndarray:
        """
        UAV 1대 고용에 따라 발생하는 비용을 array shape으로 반환해 주는 함수
        """
        value = getattr(self.cfg, "uav_hiring_cost", None)
        if value is None:
            return AttributeError("config.py 내 EnvConfig class에 uav_hiring_cost 변수가 필요합니다.")
        
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return np.full(self.num_uav, float(arr), dtype=np.float32)
        if arr.shape != (self.num_uav,):
            raise ValueError(
                f"uav_hiring_cost 변수는 ({self.num_uav},) shape을 가져야 합니다. 현재: {arr.shape}"
            )
        return arr.astype(np.float32)
    
    def _sample_requested_content(self) -> np.ndarray:
        """
        사용자가 요청하는 content를 Zipf distribution 기반으로 샘플링해주는 함수
        """
        video_ids = np.arange(1, self.cfg.num_video + 1, dtype=np.float64)
        probs = 1.0 / np.power(video_ids, float(self.cfg.zipf_alpha))
        probs = probs / probs.sum()
        sampled = self.rng.choice(self.cfg.num_video, size=self.num_user, p=probs)
        return sampled.astype(np.int32)
    
    def _sample_uav_cached_content(self) -> np.ndarray:
        """
        각 UAV가 캐싱한 content를 샘플링해주는 함수.
        현재는 caching content 선택 자체는 최적화 대상에서 제외함.
        """
        sampled = self.rng.integers(
            low=0,
            high=self.cfg.num_video,
            size=self.num_uav,
            dtype=np.int32,
        )
        return sampled.astype(np.int32)

    def _sample_user_requested_content(self) -> int:
        """
        새로 진입하는 user의 requested content를 하나 샘플링해주는 함수.
        """
        video_ids = np.arange(1, self.cfg.num_video + 1, dtype=np.float64)
        probs = 1.0 / np.power(video_ids, float(self.cfg.zipf_alpha))
        probs = probs / probs.sum()
        return int(self.rng.choice(self.cfg.num_video, p=probs))
    
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
    
    def _reset_user_regions(self) -> None:
        """
        episode 시작 시 user region을 초기화하는 함수
        """
        self.user_region = self.rng.integers(
            low=0,
            high=self.num_rsu,
            size=self.num_user,
            dtype=np.int32,
        )
    
    def _sample_local_distance(self, channel_cfg, size: tuple[int, ...]) -> np.ndarray:
        """
        동일한 coverage region 내 local distance를 샘플링하는 함수
        """
        sampling = str(getattr(channel_cfg, "distance_sampling", "uniform")).lower().strip()

        if sampling == "uniform":
            low = float(getattr(channel_cfg, "distance_min", channel_cfg.distance))
            high = float(getattr(channel_cfg, "distance_max", channel_cfg.distance))
            min_distance = float(getattr(channel_cfg, "min_distance", 1.0))
            sampled = self.rng.uniform(low=low, high=high, size=size)
            sampled = np.maximum(sampled, min_distance)
            return sampled.astype(np.float32)

        return np.full(
            size,
            max(float(channel_cfg.distance), float(channel_cfg.min_distance)),
            dtype=np.float32,
        )
    
    def _refresh_link_distances(self) -> None:
        """
        user_region을 기준으로 RSU-user, UAV-user 거리 상태를 생성하는 함수.
        같은 region 안에서 local distance ~ Uniform(distance_min, distance_max)
        """
        self.rsu_user_distance = self._sample_local_distance(
            self.cfg.rsu_channel,
            (self.num_rsu, self.num_user),
        )
        self.uav_user_distance = self._sample_local_distance(
            self.cfg.uav_channel,
            (self.num_uav, self.num_user),
        )

    def _update_user_region(self) -> Dict[str, np.ndarray]:
        """
        FSMC에 따른 user 위치 영역을 업데이트하는 함수.
        다음과 같은 동작 수행.
        
            1) 매 slot마다 각 user는 move_prob 확률로 왼쪽 region으로 이동한다.
            2) region 0에서 user 이동이 발생하면 해당 user가 오른쪽 끝 region으로 재진입.
            3) 새로 진입한 user는 queue와 requested content를 새로 초기화.
        """
        prev_region = self.user_region.copy()

        move_prob = float(self.cfg.move_prob)
        move_mask = (self.rng.random(self.num_user) < move_prob)

        entered_mask = np.zeros(self.num_user, dtype=np.int32)

        for n in range(self.num_user):
            if not bool(move_mask[n]):
                continue

            if int(self.user_region[n]) > 0:
                self.user_region[n] -= 1
            else:
                print(f"user {n} | region 이탈로 재진입 발생")
                self.user_region[n] = self.num_rsu - 1
                self.queue[n] = float(self.cfg.init_queue)
                self.requested_content[n] = self._sample_user_requested_content()
                entered_mask[n] = 1
        
        return {
            "prev_user_region": prev_region.astype(np.int32),
            "next_user_region": self.user_region.copy().astype(np.int32),
            "move_mask": move_mask.astype(np.int32),
            "entered_mask": entered_mask.astype(np.int32),
        }
    
    def _region_mask_rsu(self) -> np.ndarray:
        """
        RSU m이 user n을 같은 coverage region에서만 서비스하도록 하는 함수
        """
        rsu_idx = np.arange(self.num_rsu, dtype=np.int32)[:, None]
        user_region = self.user_region[None, :]
        return (rsu_idx == user_region).astype(np.int32)

    def _region_mask_uav(self) -> np.ndarray:
        """
        UAV u가 해당 coverage region user만 서비스하도록 하는 함수
        현재 num_uav == num_rsu이므로 uav index를 region index로 사용한다.
        """
        uav_idx = np.arange(self.num_uav, dtype=np.int32)[:, None]
        user_region = self.user_region[None, :]
        return (uav_idx == user_region).astype(np.int32)
    
    def _get_effective_rsu_connection_matrix(self) -> np.ndarray:
        """
        현재 region constraint가 반영된 RSU-user connection matrix.
        """
        return (self.rsu_scheduling * self._region_mask_rsu()).astype(np.int32)

    def _get_effective_uav_connection_matrix(self) -> np.ndarray:
        """
        현재 region constraint 및 UAV hiring이 반영된 UAV-user connection matrix.
        """
        return (
            self.uav_scheduling
            * self.uav_hiring[:, None]
            * self._region_mask_uav()
        ).astype(np.int32)

    def _get_user_node_connection_state(self) -> Dict[str, np.ndarray]:
        """
        fast-timescale policy에 제공할 user별 node connection state를 만든다.

        반환값:
            rsu_connection:
                shape = (M, N)
                rsu_connection[m, n] = 1이면 user n이 RSU m과 연결됨.

            uav_connection:
                shape = (U, N)
                uav_connection[u, n] = 1이면 user n이 UAV u와 연결됨.

            connected_rsu:
                shape = (N,)
                user n이 연결된 RSU index. 없으면 -1.

            connected_uav:
                shape = (N,)
                user n이 연결된 UAV index. 없으면 -1.

            connection_type:
                shape = (N,)
                0 = no connection
                1 = RSU only
                2 = UAV only
                3 = both RSU and UAV candidate connection

        주의:
            현재 slow action validator가 user별 단일 연결을 강제하지 않을 수 있으므로,
            matrix 형태를 함께 제공한다. connected_rsu/connected_uav는 첫 번째 연결 index만 제공한다.
        """
        rsu_connection = self._get_effective_rsu_connection_matrix()
        uav_connection = self._get_effective_uav_connection_matrix()

        connected_rsu = np.full(self.num_user, -1, dtype=np.int32)
        connected_uav = np.full(self.num_user, -1, dtype=np.int32)
        connection_type = np.zeros(self.num_user, dtype=np.int32)

        for n in range(self.num_user):
            rsu_candidates = np.flatnonzero(rsu_connection[:, n] > 0)
            uav_candidates = np.flatnonzero(uav_connection[:, n] > 0)

            has_rsu = len(rsu_candidates) > 0
            has_uav = len(uav_candidates) > 0

            if has_rsu:
                connected_rsu[n] = int(rsu_candidates[0])
            if has_uav:
                connected_uav[n] = int(uav_candidates[0])

            if has_rsu and has_uav:
                connection_type[n] = 3
            elif has_rsu:
                connection_type[n] = 1
            elif has_uav:
                connection_type[n] = 2
            else:
                connection_type[n] = 0

        return {
            "rsu_connection": rsu_connection.astype(np.int32),
            "uav_connection": uav_connection.astype(np.int32),
            "connected_rsu": connected_rsu.astype(np.int32),
            "connected_uav": connected_uav.astype(np.int32),
            "connection_type": connection_type.astype(np.int32),
        }

    def reset(self) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        에피소드 초기화 수행.
        """
        # time slot 및 round 초기화 / 동시에 에피소드는 1 증가
        self.t = 0
        self.episode += 1
        self.round_idx = 0
        self.round_slot = 0

        # 사용자 큐 초기화
        self.queue = np.full(self.num_user, float(self.cfg.init_queue), dtype=np.float32)

        self._reset_user_regions()

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
            "user_region": self.user_region.copy(),
            "requested_content": self.requested_content.copy(),
            "uav_cached_content": self.uav_cached_content.copy(),
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
    
    def _start_new_round(self) -> None:
        """
        round boundary에서 round-level state 초기화 수행.
        """
        self.round_idx = self.t // self.slow_T
        self.round_slot = 0
        self.round_start_E = self.E.copy()

        for battery in self.batteries:
            battery.start_round(round_horizon=self.slow_T)

        self._reset_round_reward_accumulators()

    def _rule_based_uav_charge(self) -> np.ndarray:
        """
        UAV 충전은 규칙 기반 동작으로 처리.
        """
        if not bool(self.cfg.battery.allow_charge):
            return np.zeros(self.num_uav, dtype=np.int32)
        if not bool(self.cfg.battery.enable_charging):
            return np.zeros(self.num_uav, dtype=np.int32)

        low_soc = (self.E <= float(self.cfg.battery.e_min)).astype(np.int32)
        return (low_soc * self.uav_hiring).astype(np.int32)

    def _clear_charging_service_actions(self, fast_act: FastAction) -> None:
        """
        충전중인 UAV의 서비스를 차단하는 함수
        """
        for uav_idx in range(self.num_uav):
            if int(fast_act.uav_charge[uav_idx]) != 1:
                continue

            fast_act.uav_chunks[uav_idx, :] = 0
            fast_act.uav_layers[uav_idx, :] = 0
            fast_act.uav_power[uav_idx, :] = 0.0

    def apply_slow_action(self, action: EnvAction) -> SlowAction:
        """
        round-level slow-timescale decision을 갱신하는 함수.
        """
        if int(self.round_slot) != 0: # 안전장치
            raise RuntimeError(
                "apply_slow_action() 함수는 round boundary에서만 호출해야 합니다. "
                f"현재 round_slot = {self.round_slot}"
            )

        slow_act = parse_slow_action(action, self.cfg)

        rsu_region_mask = self._region_mask_rsu()
        uav_region_mask = self._region_mask_uav()

        self.rsu_scheduling = (slow_act.rsu_scheduling * rsu_region_mask).astype(np.int32)
        self.uav_hiring = slow_act.uav_hiring.copy().astype(np.int32)
        self.uav_scheduling = (
            slow_act.uav_scheduling
            * self.uav_hiring[:, None]
            * uav_region_mask
        ).astype(np.int32)

        self._start_new_round()

        return SlowAction(
            rsu_scheduling=self.rsu_scheduling.copy(),
            uav_hiring=self.uav_hiring.copy(),
            uav_scheduling=self.uav_scheduling.copy(),
        )
    
    def _build_effective_fast_action(self, fast_act: FastAction) -> FastAction:
        """
        slow-timescale decision과 현재 region constraint를 반영하여
        실제 실행 가능한 fast action으로 projection하는 함수.
        """
        rsu_connection = self._get_effective_rsu_connection_matrix()
        uav_connection = self._get_effective_uav_connection_matrix()

        residual_users = (uav_connection.sum(axis=0) > 0).astype(np.int32)

        effective_rsu_chunks = fast_act.rsu_chunks.copy()
        effective_rsu_layers = fast_act.rsu_layers.copy()

        effective_uav_chunks = fast_act.uav_chunks.copy()
        effective_uav_layers = fast_act.uav_layers.copy()
        effective_uav_power = fast_act.uav_power.copy()

        # RSU scheduling + region constraint
        effective_rsu_chunks = effective_rsu_chunks * rsu_connection
        effective_rsu_layers = effective_rsu_layers * rsu_connection

        # UAV hiring off이면 service 제거
        inactive_uav_mask = self.uav_hiring <= 0
        effective_uav_chunks[inactive_uav_mask, :] = 0
        effective_uav_layers[inactive_uav_mask, :] = 0
        effective_uav_power[inactive_uav_mask, :] = 0.0

        # UAV scheduling + residual + region constraint
        effective_uav_chunks = (
            effective_uav_chunks
            * uav_connection
            * residual_users[None, :]
        )
        effective_uav_layers = (
            effective_uav_layers
            * uav_connection
            * residual_users[None, :]
        )
        effective_uav_power = (
            effective_uav_power
            * uav_connection.astype(np.float32)
            * residual_users[None, :].astype(np.float32)
        )

        return FastAction(
            rsu_chunks=effective_rsu_chunks.astype(np.int32),
            rsu_layers=effective_rsu_layers.astype(np.int32),
            uav_chunks=effective_uav_chunks.astype(np.int32),
            uav_layers=effective_uav_layers.astype(np.int32),
            uav_power=effective_uav_power.astype(np.float32),
            uav_charge=fast_act.uav_charge.copy().astype(np.int32),
            playback=fast_act.playback.copy().astype(np.float32),
            rsu_user_distance=self.rsu_user_distance.copy(),
            uav_user_distance=self.uav_user_distance.copy(),
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
        UAV battery transition을 수행하는 함수
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
        uav_hiring: np.ndarray,
        charging_state: np.ndarray,
        battery_step_info: list[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Scaled Fast-timescale DPP reward 계산을 수행하는 함수.

        다음과 같은 산식을 가짐:

            R_L(t)
            = alpha_Z * sum_n Z_n(t) d_n(t)
              - alpha_B * sum_u B_u(t) e_u(t)
              + alpha_B * sum_u B_u(t) e_c(t)
              + V * sum_n q_n(t)
        """
        prev_Q_arr = np.asarray(prev_Q, dtype=np.float32)
        next_Q_arr = np.asarray(next_Q, dtype=np.float32)
        prev_Z_arr = np.asarray(prev_Z, dtype=np.float32)
        next_Z_arr = np.asarray(next_Z, dtype=np.float32)
        prev_E_arr = np.asarray(prev_E, dtype=np.float32)
        next_E_arr = np.asarray(next_E, dtype=np.float32)
        prev_B_arr = np.asarray(prev_Y, dtype=np.float32)
        next_B_arr = np.asarray(next_Y, dtype=np.float32)

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

        alpha_Z = float(getattr(self.cfg, "alpha_Z", 1.0))
        alpha_B = float(getattr(self.cfg, "alpha_B", 30.0))
        V = float(getattr(self.cfg, "V", 1.0))

        raw_video_delivery_term = float(np.sum(prev_Z_arr * delivered_arr))
        raw_battery_consume_term = -float(np.sum(prev_B_arr * consumed_soc_arr))
        raw_battery_charge_term = float(np.sum(prev_B_arr * charged_soc_arr))
        raw_quality_term = float(np.sum(quality_arr))

        video_delivery_term = alpha_Z * raw_video_delivery_term
        battery_consume_term = alpha_B * raw_battery_consume_term
        battery_charge_term = alpha_B * raw_battery_charge_term
        quality_term = V * raw_quality_term

        fast_reward = (
            video_delivery_term
            + battery_consume_term
            + battery_charge_term
            + quality_term
        )

        components: Dict[str, Any] = {
            "reward_coefficients": (
                self.cfg.reward_coefficients()
                if hasattr(self.cfg, "reward_coefficients")
                else {
                    "alpha_Z": alpha_Z,
                    "alpha_B": alpha_B,
                    "V": V,
                }
            ),

            "prev_Q": prev_Q_arr.copy(),
            "next_Q": next_Q_arr.copy(),
            "prev_Z": prev_Z_arr.copy(),
            "next_Z": next_Z_arr.copy(),

            "prev_E": prev_E_arr.copy(),
            "next_E": next_E_arr.copy(),
            "prev_B": prev_B_arr.copy(),
            "next_B": next_B_arr.copy(),

            "delivered_total_per_user": delivered_arr.copy(),
            "quality_total_per_user": quality_arr.copy(),

            "consumed_soc_per_uav": consumed_soc_arr.copy(),
            "charged_soc_per_uav": charged_soc_arr.copy(),

            "raw_video_delivery_term": raw_video_delivery_term,
            "raw_battery_consume_term": raw_battery_consume_term,
            "raw_battery_charge_term": raw_battery_charge_term,
            "raw_quality_term": raw_quality_term,

            "video_delivery_term": video_delivery_term,
            "battery_consume_term": battery_consume_term,
            "battery_charge_term": battery_charge_term,
            "quality_term": quality_term,
            "fast_reward": float(fast_reward),

            "num_hired_uav": int(np.asarray(uav_hiring, dtype=np.int32).sum()),
            "num_charging_uav": int(np.asarray(charging_state, dtype=np.int32).sum()),
            "num_outage_uav": int(np.asarray(self.outage, dtype=np.int32).sum()),

            "sum_delivery": float(np.sum(delivered_arr)),
            "sum_quality": float(np.sum(quality_arr)),
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
        Slow-timescale DPP reward 계산을 수행하는 함수.

        다음과 같은 산식을 가짐:

            R_H(r)
            = - W_hire * sum_u mu_u(r) D_u^hire
              + sum_{t in T_r} R_L(t)
        """
        self.round_fast_reward_sum += float(fast_reward)
        self.round_quality_sum += float(reward_components.get("sum_quality", 0.0))
        self.round_delivery_sum += float(reward_components.get("sum_delivery", 0.0))
        self.round_battery_consume_sum += float(reward_components.get("battery_consume_term", 0.0))
        self.round_battery_charge_sum += float(reward_components.get("battery_charge_term", 0.0))

        hire_cost_per_uav = self._hire_cost()
        hire_cost_raw = float(np.sum(np.asarray(self.uav_hiring, dtype=np.float32) * hire_cost_per_uav))
        hire_weight = float(getattr(self.cfg, "hire_weight", 1.0))
        hire_cost = -hire_weight * hire_cost_raw

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
                "hire_weight": float(hire_weight),
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
            "hire_weight": float(hire_weight),
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
    
    def get_slow_obs(self) -> Dict[str, np.ndarray]:
        """
        slow-timescale 상태값을 반환하는 함수.
        """
        return {
            "Z": self.Z.copy(),
            "B": self.Y.copy(),
            "user_region": self.user_region.copy(),
        }

    def get_fast_obs(self) -> Dict[str, np.ndarray]:
        """
        fast-timescale 상태값을 반환하는 함수.
        """
        connection_state = self._get_user_node_connection_state()

        return {
            "Z": self.Z.copy(),
            "B": self.Y.copy(),
            "user_region": self.user_region.copy(),

            # user가 어떤 node와 연결돼 있는지 나타내는 fast policy state
            "connection_type": connection_state["connection_type"].copy(),
            "connected_rsu": connection_state["connected_rsu"].copy(),
            "connected_uav": connection_state["connected_uav"].copy(),
            "rsu_connection": connection_state["rsu_connection"].copy(),
            "uav_connection": connection_state["uav_connection"].copy(),
        }

    def step(self, action: EnvAction) -> tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        환경의 1-slot 진행 함수로, Fast-timescale 진행을 담당.
        """
        fast_act = parse_fast_action(action, self.cfg)

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

        prev_user_region = self.user_region.copy()
        prev_connection_state = self._get_user_node_connection_state()

        prev_E = self.E.copy()
        prev_B = self.Y.copy()
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

        # Queue update
        playback = fast_act_eff.playback.astype(np.float32)
        consumed = np.minimum(self.queue, playback)
        stall = np.maximum(playback - self.queue, 0.0)

        self.queue = np.clip(
            self.queue - consumed + delivered_total_per_user,
            0.0,
            float(self.cfg.max_queue),
        ).astype(np.float32)

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
        truncated = bool(next_t >= int(self.cfg.episode_slots))

        reward, reward_components = self._compute_reward(
            prev_Q=prev_Q,
            next_Q=self.queue,
            prev_Z=prev_Z,
            next_Z=self.Z,
            prev_E=prev_E,
            next_E=self.E,
            prev_Y=prev_B,
            next_Y=self.Y,
            delivered_total_per_user=delivered_total_per_user,
            quality_total_per_user=quality_total_per_user,
            uav_hiring=self.uav_hiring,
            charging_state=self.charging_state,
            battery_step_info=battery_step_info,
            is_round_boundary=is_round_boundary,
        )

        # round state update
        if is_round_boundary:
            self._start_new_round()
        else:
            self.round_idx = int(next_round_idx)
            self.round_slot = int(next_round_slot)

        region_info = self._update_user_region()
        self._refresh_link_distances()

        next_connection_state = self._get_user_node_connection_state()

        info: Dict[str, Any] = {
            # transition meta
            "prev_time": prev_t,
            "next_time": next_t,
            "prev_round_slot": prev_round_slot,
            "next_round_slot": int(self.round_slot),
            "active_round_idx": prev_round_idx,
            "next_round_idx": int(self.round_idx),
            "is_round_boundary": is_round_boundary,
            "terminated": terminated,
            "truncated": truncated,

            # slow context
            "uav_hiring": self.uav_hiring.copy(),
            "rsu_scheduling": self.rsu_scheduling.copy(),
            "uav_scheduling": self.uav_scheduling.copy(),
            "requested_content": self.requested_content.copy(),
            "uav_cached_content": self.uav_cached_content.copy(),
            "residual_users": fast_act_eff.residual_users.copy(),
            "uav_charge_effective": fast_act_eff.uav_charge.copy(),

            # connection state
            "prev_connection_state": prev_connection_state,
            "next_connection_state": next_connection_state,

            # mobility
            "prev_user_region": prev_user_region.copy(),
            "next_user_region": self.user_region.copy(),
            "region_info": region_info,

            # queue transition
            "prev_Q": prev_Q.copy(),
            "next_Q": self.queue.copy(),
            "prev_Z": prev_Z.copy(),
            "next_Z": self.Z.copy(),
            "playback": playback.copy(),
            "consumed": consumed.copy(),
            "stall": stall.copy(),

            # battery transition
            "prev_E": prev_E.copy(),
            "next_E": self.E.copy(),
            "prev_B": prev_B.copy(),
            "next_B": self.Y.copy(),
            "prev_Y": prev_B.copy(),
            "next_Y": self.Y.copy(),
            "outage": self.outage.copy(),
            "charging_state": self.charging_state.copy(),
            "battery_step_info": battery_step_info,

            # slot result
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

        obs = self.get_fast_obs()

        return obs, float(reward), terminated, truncated, info