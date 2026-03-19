from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np

from channel import ChannelModel
from config import EnvConfig

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
        self.num_rsu = self.cfg.num_rsu
        self.num_user = self.cfg.num_user
        self.num_uav = self.cfg.num_uav
        self.slow_T = self.cfg.slow_T

        # 비디오 및 캐싱 시스템 설정
        self.num_video = self.cfg.num_video
        self.rsu_caching = self.cfg.rsu_caching
        self.layer = self.cfg.layer
        self.chunk = self.cfg.chunk
        self.rsu_capacity = self.cfg.rsu_capacity

        # 사용자 이동 패턴 설정
        self.spawn_base = self.cfg.spawn_base
        self.spawn_amp = self.cfg.spawn_amp
        self.spawn_period = self.cfg.spawn_period
        self.depart_base = self.cfg.depart_base
        self.depart_amp = self.cfg.depart_amp
        self.depart_period = self.cfg.depart_period

        # 채널 설정
        self.rsu_channel = self.cfg.rsu_channel
        self.uav_channel = self.cfg.uav_channel

        # 배터리 모델
        self.battery_cfg = self.cfg.battery
        self.E_max = self.battery_cfg.e_max
        self.E_min = self.battery_cfg.e_min
        self.batteries = [UAVBattery(self.battery_cfg) for _ in range(self.num_uav)]

        # runtime 상태 관리
        self.t = 0 
        self.episode = 0
        self.round_idx = 0
        self.round_slot = 0
        self.queue = np.zeros(self.num_user, dtype=np.float32)
        self.mu = np.zeros(self.num_uav, dtype=np.int32)
        self.outage = np.zeros(self.num_uav, dtype=np.int32)
        self.charging_state = np.zeros(self.num_uav, dtype=np.int32)
        self.tx_power = np.zeros(self.num_uav, dtype=np.float32)
        self.charge_counters = np.zeros(self.num_uav, dtype=np.int32)
        self.scheduled_users = np.zeros((self.num_uav, self.num_user), dtype=np.int32)
        self.round_start_E = np.zeros(self.num_uav, dtype=np.float32)

        self.reset()
    
    @property
    def E(self) -> np.ndarray:
        """
        UAV들의 Actual SoC Queue 반환 함수
        """
        return np.array([b.soc for b in self.batteries], dtype=np.float32)
    
    @property
    def Y(self) -> np.ndarray:
        """
        UAV들의 Virtual Soc Queue 반환 함수
        """
        return np.array([b.virtual_q for b in self.batteries], dtype=np.float32)
    
    def _start_new_round(self):
        """
        라운드 초기화 함수
        """
        self.round_idx = self.t // self.slow_T
        self.round_slot = 0
        self.round_start_E = self.E.copy()
        for b in self.batteries:
            b.start_round(round_horizon=self.slow_T, reset_virtual_queue=True)
        
    def reset(self) -> Dict[str, np.ndarray]:
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
        self.queue = np.full(self.num_user, self.cfg.initial_queue, dtype=np.float32)

        # 배터리 큐 초기화
        for b in self.batteries:
            b.reset_episode()
            b.start_round(self.slow_T, reset_virtual_queue=True)

        # UAV 고용 상태 초기화
        # 초기 상태이므로, 미고용 상태를 가정함
        self.mu = np.zeros(self.num_uav, dtype=np.int32)
        self.outage = np.zeros(self.num_uav, dtype=np.int32)
        self.charging_state = np.zeros(self.num_uav, dtype=np.int32)
        self.tx_power = np.zeros(self.num_uav, dtype=np.float32)

        # 충전 상태 초기화
        # 아무 uav 도 충전 안하고 있다고 가정
        self.charge_counters = np.zeros(self.num_uav, dtype=np.int32)
        self.scheduled_users = np.zeros((self.num_uav, self.num_user), dtype=np.int32)
        self.round_start_E = self.E.copy()

        return self._get_state()

    def _get_state(self) -> Dict[str, np.ndarray]:
        """
        현 상태값을 반환하는 함수
        """
        return {
            "Q": self.queue.copy(),
            "Y": self.Y.copy(),
            "E": self.E.copy(),
            "mu": self.mu.copy(),
            "round_slot": np.array([self.round_slot], dtype=np.int32),
            "outage": self.outage.copy(),
        }
    
# ========================================검증 함수========================================================
    def _test_binary_matrix(
        self,
        value: np.ndarray,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=np.int32)
        if arr.shape != shape:
            raise ValueError(f"Expected shape {shape}, but got {arr.shape}.")
        return (arr > 0).astype(np.int32)

    def _test_binary_vector(self, value: np.ndarray, size: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.int32)
        if arr.shape != (size,):
            raise ValueError(f"Expected shape ({size},), but got {arr.shape}.")
        return (arr > 0).astype(np.int32)

    def _test_float_vector(self, value: np.ndarray, size: int, clip_min: float = 0.0) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (size,):
            raise ValueError(f"Expected shape ({size},), but got {arr.shape}.")
        return np.maximum(arr, clip_min)

# ========================================계산 함수========================================================
    def _chunk_size_bits(self, layer_idx: int) -> float:
        clipped_layer = int(np.clip(layer_idx, 1, self.cfg.layer))
        return self.cfg.base_chunk_size_bits * float(clipped_layer)

    def _quality_weight(self, layer_idx: int) -> float:
        clipped_layer = int(np.clip(layer_idx, 1, self.cfg.layer))
        if clipped_layer <= len(self.cfg.quality_weights):
            return float(self.cfg.quality_weights[clipped_layer - 1])
        return float(clipped_layer)

    def _effective_distances(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        distances = action.get(
            "uav_user_distance",
            np.full((self.num_uav, self.num_user), self.cfg.uav_channel.distance, dtype=np.float32),
        )
        distances = np.asarray(distances, dtype=np.float32)
        if distances.shape != (self.num_uav, self.num_user):
            raise ValueError(
                f"uav_user_distance must have shape {(self.num_uav, self.num_user)}, got {distances.shape}."
            )
        return np.maximum(distances, self.cfg.uav_channel.min_distance)
    
    def _compute_uav_delivery(
        self,
        served_users: np.ndarray,
        requested_chunks: np.ndarray,
        layers: np.ndarray,
        tx_power: np.ndarray,
        distances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        UAV Delivery 동작 구현 함수
        """
        actual_delivery = np.zeros((self.num_uav, self.num_user), dtype=np.int32)
        user_rate = np.zeros((self.num_uav, self.num_user), dtype=np.float32)
        remaining_request = requested_chunks.astype(np.int32).copy()

        for u in range(self.num_uav):
            active_users = np.where(served_users[u] == 1)[0]
            if active_users.size == 0:
                print("UAV에게 청크를 전송받는 user가 없습니다.")
                continue

            for n in active_users:
                if remaining_request[n] <= 0:
                    print("user들의 청크 요청이 마무리되었습니다.")
                    continue

                cap_bps = self.uav_channel.capacity(
                    tx_power=float(tx_power[u]),
                    distance=float(distances[u, n]),
                    rng=self.rng,
                )
                user_rate[u, n] = float(cap_bps)

                per_user_bits = cap_bps * self.battery_cfg.slot_duration / active_users.size
                chunk_bits = self._chunk_size_bits(int(layers[n]))
                max_chunks = int(np.floor(per_user_bits / chunk_bits))
                delivered = int(min(remaining_request[n], max_chunks))
                actual_delivery[u, n] = max(0, delivered)
                remaining_request[n] -= actual_delivery[u, n]
        
        return actual_delivery, user_rate

    def step(self, action: Dict[str, np.ndarray]):
        """
        환경의 1 time step 진행 함수 (Fast Timescale 기준)

        - Slow Timescale: UAV 고용 및 스케줄링 수행
        - Fast Timescale: 비디오 청크 및 레이어 전송 수행
        """
        # Slow Timescale
        # 매 T 주기마다, UAV 고용 여부를 갱신함
        # 또한, User scheduling을 갱신함
        new_mu = action.get("hiring", np.zeros(self.num_uav, dtype=np.int32))
        scheduled_users = action.get(
            "scheduled_users",
            np.zeros((self.num_uav, self.num_user), dtype=np.int32)
        )
        new_mu = self._test_binary_vector(new_mu, self.num_uav)
        scheduled_users = self._test_binary_matrix(scheduled_users, (self.num_uav, self.num_user))

        if self.t % self.slow_T == 0:
            self.mu = new_mu.copy()
            self.scheduled_users = scheduled_users.copy()
            self._start_new_round()

        # Fast Timescale
        # 매 time-step 마다, l_n (chunk) 및 k_n (layer) 를 전송함
        # 또한, 전송 전력및 충전, served user 수도 선택함
        served_users = action.get("served_users", np.zeros((self.num_uav, self.num_user), dtype=np.int32))
        layers = action.get("layer", np.ones(self.num_user, dtype=np.int32))
        tx_power = action.get("tx_power", np.zeros(self.num_uav, dtype=np.float32))
        charge = action.get("charge", np.zeros(self.num_uav, dtype=np.int32))
        playback = action.get(
            "playback",
            np.full(self.num_user, self.cfg.playback_rate, dtype=np.float32),
        )

        # chunk 테스트
        served_users = self._test_binary_matrix(served_users, (self.num_uav, self.num_user))
        requested_chunks = np.asarray(requested_chunks, dtype=np.int32)
        if requested_chunks.shape != (self.num_user,):
            raise ValueError(f"chunk must have shape ({self.num_user},), got {requested_chunks.shape}.")
        
        # layer 테스트
        layers = np.asarray(layers, dtype=np.int32)
        if layers.shape != (self.num_user,):
            raise ValueError(f"layer must have shape ({self.num_user},), got {layers.shape}.")
        layers = np.clip(layers, 1, self.cfg.layer)

        # tx_power 및 charge, playback 테스트
        tx_power = self._test_float_vector(tx_power, self.num_uav, clip_min=0.0)
        tx_power = np.minimum(tx_power, self.battery_cfg.max_tx_power)

        charge = self._test_binary_vector(charge, self.num_uav)
        playback = np.asarray(playback, dtype=np.float32)
        if playback.shape != (self.num_user,):
            raise ValueError(f"playback mush have shape ({self.num_user},), got {playback.shape}.")
        playback = np.maximum(playback, 0.0)
        distances = self._effective_distances(action)

        # 배터리 Queue
        # E(t+1) = max(E_t - e_u, 0) + e_c
        prev_E = self.E.copy()
        prev_Y = self.Y.copy()
        queue_before = self.queue.copy()

        for u in range(self.num_uav):
            if self.mu[u] == 0:
                served_users[u, :] = 0
            
            if charge[u] == 1 and not self.battery_cfg.enable_charging:
                charge[u] = 0

            if charge[u] == 1 and not self.battery_cfg.allow_charge:
                served_users[u, :] = 0

            if self.outage[u] == 1:
                served_users[u, :] = 0

            served_users[u, :] = served_users[u, :] * self.scheduled_users[u, :]
            
            if served_users[u].sum() > self.cfg.uav_user_cap:
                active_idx = np.where(served_users[u] == 1)[0]
                served_users[u, :] = 0
                served_users[u, active_idx[:self.cfg.uav_user_cap]] = 1

        actual_delivery, user_rate = self._compute_uav_delivery(
            served_users=served_users,
            requested_chunks=requested_chunks,
            layers=layers,
            tx_power=tx_power,
            distances=distances,
        )

        # 배터리 step
        battery_infos: List[Dict[str, Any]] = []
        for u in range(self.num_uav):
            active_users = int(served_users[u].sum())
            serving_flag = (active_users > 0)
            delivered_chunks_u = int(actual_delivery[u].sum())

            info = self.batteries[u].step(
                tx_power=float(tx_power[u]),
                delivered_chunks=delivered_chunks_u,
                is_serving=serving_flag,
                do_charge=bool(charge[u]),
            )
            self.outage[u] = int(info.outage)
            self.charging_state[u] = int(charge[u])
            self.tx_power[u] = tx_power[u]
            self.charge_counters[u] += int(charge[u])
            battery_infos.append(asdict(info))
        
        delivered_per_user = actual_delivery.sum(axis=0).astype(np.float32)
        delivered_quality = np.array(
            [delivered_per_user[n] * self._quality_weight(int(layers[n])) for n in range(self.num_user)],
            dtype=np.float32,
        )

        consumed = np.minimum(self.queue, playback)
        stall = np.maximum(playback - self.queue, 0.0)
        self.queue = np.clip(self.queue - consumed + delivered_per_user, 0.0, self.cfg.max_queue)

        reward = float(
            delivered_quality.sum()
            - self.cfg.stall_penalty * stall.sum()
            - self.cfg.battery_virtual_penalty * self.Y.sum()
            - self.cfg.outage_penalty * self.outage.sum()
        )

        self.t += 1
        self.round_slot = self.t % self.slow_T
        done = False

        info: Dict[str, Any] = {
            "prev_E": prev_E,
            "next_E": self.E.copy(),
            "prev_Y": prev_Y,
            "next_Y": self.Y.copy(),
            "mu": self.mu.copy(),
            "scheduled_users": self.scheduled_users.copy(),
            "effective_served_users": served_users.copy(),
            "actual_delivery": actual_delivery.copy(),
            "delivered_per_user": delivered_per_user.copy(),
            "delivered_quality": delivered_quality.copy(),
            "uav_user_rate_bps": user_rate.copy(),
            "queue_before": queue_before,
            "queue_after": self.queue.copy(),
            "playback": playback.copy(),
            "stall": stall.copy(),
            "tx_power": self.tx_power.copy(),
            "charge": charge.copy(),
            "outage": self.outage.copy(),
            "battery_step_info": battery_infos,
            "round_slot": self.round_slot,
            "round_idx": self.round_idx,
        }

        return self._get_state(), reward, done, info