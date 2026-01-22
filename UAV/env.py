#--------------------------------------------------------------------------------------------------------------------------------
# 1027. (코드 의도)
# state, action 을 다시 정리해서 Markov Process 구축해보자 / step by step 으로 action, state, reward 가 변하는 과정이 계속해서 보여져야 함 - 구현
# 랜덤하게 시뮬레이션 돌렸을 때 state transition, action space 는 잘 바뀌어가고있는 지 시각적으로 확인해야 함 - 구현
#--------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------
# 1103. (코드 수정 1)
# 차량의 숫자를 결정하기보다, "차량의 최대 수(총 대수)"만 결정을 해서 유동적으로 판단 - 구현
# 차량은 왔다갔다 하고, 이동 확률(FSMC) 도 범위마다 달라야 할 것 같음 (비디오 길이도 제한을 두자) - 구현
# 어떠한 확률로 user 가 새로 생성되게끔 (user 가 새로 들어올 수 있게 / 나가고) - 구현
# Channel Cap 고려해보자 (거리가 고정이긴 하지만) / 이거 안넘는거 검증할 수 있게 (time step 마다 채널 랜덤 생성) -> 고정된 청크로 하자 - 구현
# MBS -> Channel Cap 도 고려할 수 있어야 함 - 구현
#--------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------
# 1110. (코드 수정 2)
# finite 한 청크를 다 받은 이후에는 인기도에 맞춰 새로운 청크를 요청할 수 있게 해야 함 - 구현
# 사용자가 몇 번쨰 청크까지 수신했는 지 -> state - 구현
# ex) 마지막 청크까지 수신했으면 다른 청크를 요청할 수도 있음 (제약) - 구현
# Lyapunov Optimization 책 chapter 4 까지 공부 하자 (같이 공부)
# 삼각함수로 확률 모델링 (나가고, 들어오는)
#--------------------------------------------------------------------------------------------------------------------------------


import os
import numpy as np
import json
import matplotlib.pyplot as plt
import logging
import math
from typing import Optional, Dict, Any, Tuple
np.random.seed(42)

directory = './UAV/result_plot/'

# 유틸함수
def setup_logger(name: str = "UAV.env", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = None
    return logger

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

class ChannelModel:
    """ 
    무선 채널(Rayleigh Fading) 시뮬레이션 클래스
    channel cap 할당 (rsu, uav)
    거리는 고정 - 20 ?
    """
    def __init__(self, distance: float=20, bandwidth: float=1.0, snr_db: float=30.0,
                 inr_db: float=5.0, pathloss_b: float=2.0, shadow_sigma_db: float=4.0):
        self.distance = float(distance)
        self.Bandwidth = float(bandwidth)
        
        self.SNR_lin = 10 ** (float(snr_db) / 10)
        self.INR_lin = 10 ** (float(inr_db) / 10)
        self.b = float(pathloss_b)
        self.shadow_sigma = float(shadow_sigma_db)

    def _sample_fading(self):
        return np.random.rayleigh(1.0)
    
    def _sample_shadowing_lin(self):
        s_db = np.random.normal(0.0, self.shadow_sigma)
        return 10 ** (s_db / 10.0)
    
    def _sample_channel_capacity(self):
        u = self._sample_fading()
        S_lin = self._sample_shadowing_lin()

        h2 = (S_lin / (self.distance ** self.b)) * (u ** 2)
        sinr = (self.SNR_lin * h2) / (1.0 + self.INR_lin)
        cap = self.Bandwidth * math.log2(1.0 + sinr)
        return float(max(0.0, cap))

    def calculate_capacity(self):
        self.ChannelCapacity_list = []
        for i in range(10):
            self.fast_fading_gain = abs(np.random.normal(0, 1))
            self.shadowing_effect = abs(np.random.normal(0, self.shadow_sigma * self.shadow_sigma))


class Env:
    def __init__(self, loglevel: int = logging.DEBUG):
        # 일단은 임의로 변수 정의
        self.num_rsu = 10 # 동시에, 최대 사용자 수
        self.num_user = 10

        # 제한 걸기
        self.uav_total = 5 # 총 UAV 대수 (확인해야함)
        self.N0 = 3 # RSU 가 전송할 수 있는 최대 USER

        # 콘텐츠 및 캐시
        self.num_video = 3 # 전체 비디오 파일 수
        self.RSU_caching = 2 # RSU 가 캐시하는 비디오 수
        self.UAV_caching = 2    
        
        # 비디오
        self.layer = 5
        self.chunk = 5
        self.C = np.full(self.num_rsu, 3, dtype=int) # RSU 별 슬롯 당 전달 가능한 총 청크 수 (3으로 가정)
        
        # 사용자 이탈 / 진입 확률 제어
        # 삼각함수로 모델링
        # self.spawn_prob = 0.10 # 신규 유입 확률
        # self.depart_prob = 0.05 # 이탈 확률

        self.spawn_base = 0.10 # 평균 신규 유입 확률
        self.depart_base = 0.05 # 평균 이탈 확률

        # 진폭
        self.spawn_amp = 0.05 # 유입 확률은 0.05 ~ 0.15
        self.depart_amp = 0.02 # 이탈 확률은 0.03 ~ 0.07

        # 주기
        self.spawn_period = 200.0
        self.depart_period = 300.0

        self.active_mask = np.ones(self.num_user, dtype=np.int32)

        # 청크 당 인기도 정의
        # 또한, 모든 청크 수신 시 Zipf(alpha) 분포로 신규 비디오 샘플링
        self.zipf_alpha = 1.1
        self.chunks_per_video = np.full(self.num_video, int(self.chunk), dtype=np.int32)
        self.received_chunks = np.zeros(self.num_user, dtype=np.int32)

        # MBS 가 슬롯 당 처리할 수 있는 최대 청크
        self.C_MBS = 2

        # 사용자가 몇 번째 청크까지 수신했는지?
        self.trans_id = 0

        # MBS 딜레이
        self.mbs_delay = 2 # MBS 걸릴 때 딜레이 (2 time-slot, 즉 t=1 에서 MBS 이용하면 t=3 에 청크 도착)
        # self.pending_mbs = [ [] for _ in range(self.num_user)] # 사용자별 도착 예약 타임스태프 리스트
        
        # MBS 이용에 Cap 을 고려함에 따라, 도착해야 할 청크가 못 올 수 있음
        # 이하는 그러한 상황을 고려해 코드를 수정한 내용
        self.mbs_in_delay = [ [] for _ in range(self.num_user) ] # 아직 딜레이 중
        self.mbs_ready = np.zeros(self.num_user, dtype=np.int32) # 딜레이는 끝났지만, 아직 도착하지 않은 청크 수

        # 에피소드 및 스텝
        self.max_steps = 1000
        self.episode: int = 5

        # FSMC 이동 확률
        self.move_prob = np.clip(np.linspace(0.05, 0.25, self.num_rsu), 0, 1) # Region 에 따라 FSMC 다르게 설정

        # 거리? (임의 가정)
        # 통신 반경 1 = 50m 가정 시 RSU 는 30m, UAV 는 20m 떨어져 있다고 가정(일단은)
        self.fixed_distance_rsu = 0.6
        self.fixed_distance_uav = 0.4

        # 거리에 따른 채널
        # 이는 청크의 레이어 및 개수에 영향을 준다
        self.channel_rsu = ChannelModel(distance=self.fixed_distance_rsu, bandwidth=1.0,
                                        snr_db=30.0, inr_db=5.0, pathloss_b=2.0, shadow_sigma_db=4.0)
        self.channel_uav = ChannelModel(distance=self.fixed_distance_uav, bandwidth=1.0,
                                        snr_db=30.0, inr_db=5.0, pathloss_b=2.0, shadow_sigma_db=4.0)

        # 보상 가중치 부분
        self.cs1 = -5.0
        self.cs2 = -1.0
        self.cq = -0.1
        self.cu = -0.2

        # 초기 상태 샘플링
        self.user_requests = np.random.randint(0, self.num_video, size=self.num_user)
        self.rsu_cache = []
        for _ in range(self.num_rsu):
            if self.num_video > 0:
                cached_videos = np.random.choice(self.num_video, size=min(self.RSU_caching, self.num_video), replace=False)
            else:
                cached_videos = []
            self.rsu_cache.append(set(cached_videos))

        # 사용자의 초기 위치는 0 부터 m - 1 까지 랜덤으로 가정 (m: RSU 개수) - 수정 필요
        self.user_region = np.random.randint(0, self.num_rsu, size=self.num_user)

        # 버퍼
        self.init_q = 5
        self.queue = np.full(self.num_user, self.init_q, dtype=np.int32)
        self.needed_chunks = np.full(self.num_user, self.chunk, dtype=np.int32)

        # 이름 수정 필요
        self.U_current = np.zeros(self.num_rsu, dtype=np.int32) # RSU 별 사용 중인 UAV 의 수
        self.u_current = np.zeros(self.num_user, dtype=np.int32) # 사용자별 UAV 전송 여부 상태

        # 트레이스
        self.logger = setup_logger(level=loglevel)
        self.trace_enabled = True
        self.trace = []
        self.save_dir = directory
        ensure_dir(self.save_dir)

        # 로그 정리용/
        self.finished_events = []

        # Lyapunov Optimization 도입에 따른 추가 수정 사항

        self.queue_bound = 100.0 # Virtual Queue 를 위한, 상한 설정
        self.V = 10.0 # DPP 가중치
        self.phi = 0.2 # quality fluctuation 가중치
        self.playback_b = 1 # slot 당 소모되는 청크 수
        
        self.quality_levels = np.linspace(30.0, 42.0, self.layer).astype(np.float32) # quality measure
        self.quality_max = float(np.max(self.quality_levels))

        # Virtual Queue
        # init_q = 5 로 설정했으므로, Z(0) = Q_bound - 5
        self.Z = np.clip(self.queue_bound - self.queue, 0, self.queue_bound).astype(np.int32)
        self.last_l = np.ones(self.num_user, dtype=np.int32) # 품질 변동 계산을 위함

    def reset(self):
        # 시간 축 및 에피소드 초기화    
        self.t = 0
        self.episode += 1

        # 사용자 큐 및 필요 청크 초기화 (init 함수에서 사용한 수치 재활용)
        self.queue[:] = self.init_q
        self.needed_chunks[:] = self.chunk

        # 청크 수신 진행도 및 청크 받고 있는 지 여부
        self.received_chunks[:] = 0
        self.active_mask[:] = 1 

        # MBS 상태 초기화
        self.mbs_in_delay = [ [] for _ in range(self.num_user) ]
        self.mbs_ready[:] = 0

        # UAV 및 RSU 상태 초기화
        self.U_current[:] = 0
        self.u_current[:] = 0

        # 트레이스 초기화
        self.trace.clear()
        self.trans_id = 0

        # 로그 정리용
        self.finished_events = []

        # Virtual Queue 초기화
        self.Z[:] = np.clip(self.queue_bound - self.queue, 0.0, self.queue_bound).astype(np.int32)
        self.last_l[:] = 1
        

    def _get_state(self) -> Dict[str, np.ndarray]:
        x_matrix = np.zeros((self.num_rsu, self.num_user), dtype=np.int32)
        for n in range(self.num_user):
            if n < self.num_user and self.active_mask[n]:
                m = int(self.user_region[n])
                x_matrix[m, n] = 1
        
        return {
            # 'Q': self.queue.copy(),
            'Z': self.Z.copy(),
            'x': x_matrix.copy(),
            'U': self.U_current.copy(),
            # 이하는 새로 추가한 State 들
            'N': self.needed_chunks.copy(), # 청크 얼마나 더 수신해야 하는 지
        }
    
    def _action_summary(self, y: np.ndarray, k: np.ndarray, l: np.ndarray, u: np.ndarray) -> Dict[str, Any]:
        return {
            "y_pairs": int(np.sum(y)),
            "k_min": int(np.min(k)) if k.size else 0,
            "k_max": int(np.max(k)) if k.size else 0,
            "l_sum": int(np.sum(l)),
            "u_used": int(np.sum(u))
        }

    def _collect_mbs_arrivals(self) -> np.ndarray:
        """
        Case 2 - MBS 를 이용하는 경우, 현재 t 에 도착하는 MBS 예약 청크를 버퍼에 반영
        (기존) 캐시 미스시 도착 예약 하고, 도착 시 제한 없이 큐에 반영
        (수정사항) 슬롯 t 에서 실제 도착 반영량에 상한 설정 (너무 많이는 못 받게)
        
        - mbs_in_delay: 아직 딜레이 중인, 도착 예정 시간들 리스트
        - mbs_ready: 딜레이는 끝, 하지만 아직 큐에 반영이 안된 청크 수
        - C_MBS: 슬롯 당, 실제로 반영할 수 있는 최대 ㅊ청크 Cap
        """
        arrived_mbs = np.zeros(self.num_user, dtype=np.int32)

        # 딜레이 끝난 청크들을 ready 큐로 옮김
        for n in range(self.num_user):
            lst = self.mbs_in_delay[n] # [도착시각, 도착시각, ...]
            k = 0
            while k < len(lst) and lst[k] <= self.t:
                self.mbs_ready[n] += 1
                k += 1
            if k > 0:
                # 앞의 k 는 ready 큐로 옮겼기 때문에, 리스트에서 제거
                self.mbs_in_delay[n] = lst[k:]
        
        # 이번 슬롯에 Cap 한도 내에서 도착 처리
        budget = int(self.C_MBS)
        for n in range(self.num_user):
            if budget <= 0:
                break
            if self.mbs_ready[n] <= 0 or self.needed_chunks[n] <= 0:
                continue

            deliverable = min(self.mbs_ready[n], budget, int(self.needed_chunks[n]))
            if deliverable > 0:
                arrived_mbs[n] = deliverable
                self.queue[n] += deliverable
                self.needed_chunks[n] -= deliverable
                self.received_chunks[n] += deliverable
                self.mbs_ready[n] -= deliverable
                budget -= deliverable

                if self.needed_chunks[n] <= 0:
                    self.on_finish_video(n, source="MBS")

        return arrived_mbs


    def spawn_prob(self, t: int) -> float:
        """
        time-step t 에서의 신규 유입 확률
        """
        theta = 2.0 * math.pi * (float(t) / float(self.spawn_period))
        p = self.spawn_base + self.spawn_amp * math.sin(theta)

        return float(np.clip(p, 0.0, 1.0)) # 확률이기 때문에 clipping
    

    def depart_prob(self, t: int) -> float:
        """
        time-step t 에서의 이탈 확률
        """
        theta = 2.0 * math.pi * (float(t) / float(self.depart_period))
        p = self.depart_base + self.depart_amp * math.sin(theta)
        
        return float(np.clip(p, 0.0, 1.0))


    def spawn_depart_users(self):
        """ 사용자 생성 및 이탈 함수 """
        departed_info = [] # (user, region, video, queue, needed_chunk)
        spawned_info = [] # (user, region, video, queue, needed_chunk)

        depart_p = self.depart_prob(self.t) # 이탈 확률
        spawn_p = self.spawn_prob(self.t) # 생성 확률

        # 이탈
        for n in range(self.num_user):
            if self.active_mask[n] and np.random.rand() < depart_p:
                region = int(self.user_region[n])
                video = int(self.user_requests[n])
                queue_before = int(self.queue[n])
                needed_chunk_before = int(self.needed_chunks[n])

                departed_info.append((n, region, video, queue_before, needed_chunk_before))

                self.active_mask[n] = 0
                self.queue[n] = 0
                self.Z[n] = 0
                self.last_l[n] = 1
                self.needed_chunks[n] = 0
                self.received_chunks[n] = 0
                self.mbs_in_delay[n] = []
                self.mbs_ready[n] = 0

        # 생성
        for n in range(self.num_user):
            if not self.active_mask[n] and np.random.rand() < spawn_p:
                self.active_mask[n] = 1
                self.user_region[n] = np.random.randint(0, self.num_rsu)

                vid = self.sample_popular_video()
                self.user_requests[n] = vid
                self.needed_chunks[n] = int(self.chunks_per_video[vid])
                self.queue[n] = self.init_q
                self.Z[n] = int(np.clip(self.queue_bound - self.queue[n], 0, self.queue_bound))
                self.mbs_in_delay[n] = []
                self.mbs_ready[n] = 0

                spawned_info.append((n, int(self.user_region[n]), int(vid), int(self.needed_chunks[n])))
                
        return departed_info, spawned_info
        
        
    def fsmc_step(self):
        """ FSMC 에 맞춰 사용자 이동 """
        for n in range(self.num_user):
            if not self.active_mask[n]:
                continue
            m = int(self.user_region[n])
            if m < self.num_rsu - 1 and np.random.rand() < self.move_prob[m]:
                self.user_region[n] = m + 1

    def sample_popular_video(self):
        """ 
        인기도에 기반하여, 새로운 청크를 받는 함수 (zipf 이용)
        zipf 이론: "인기 순위가 k 인 content 의 요청 확률은 (1/k) ^ alpha 에 비례한다" (요청 확률 모델링할 때 많이 이용)
                  alpha 가 클 수록 상위 content 에 인기도가 몰리는 구조 (현 구조에서는 1.1 으로, 어느 정도 상위 content 에 인기가 몰린다 가정)
        """
        ranks = np.arange(1, self.num_video + 1)
        p = 1.0 / np.power(ranks, self.zipf_alpha)
        p /= p.sum()
        return int(np.random.choice(self.num_video, p=p))
    
    def on_finish_video(self, n: int, source: str):
        """ 
        모든 청크 수신 시 새로운 비디오를 받아야 함
        이 부분에서, 위에서 논의한 zipf (인기도) 에 따라 새 청크를 선택할 수 있게 함
        """
        # 이전에 요청받은 비디오 정보 (단순 디버그용)
        old_vid = int(self.user_requests[n])
        old_chunks = None
        if 0 <= old_vid < int(self.num_video):
            old_chunks = int(self.chunks_per_video[old_vid])

        # 인기도를 기반으로 새 비디오를 선택
        new_vid = self.sample_popular_video()
        self.user_requests[n] = new_vid
        self.received_chunks[n] = 0
        self.needed_chunks[n] = int(self.chunks_per_video[new_vid])

        self.finished_events.append({
            "user": int(n),
            "source": source, # 누가 마지막 청크를 채웠는 지 (UAV / RSU / MBS)
            "old_vid": old_vid,
            "old_chunks": old_chunks,
            "new_vid": int(new_vid),
            "new_needed_chunks": int(self.needed_chunks[n])            
        })

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool]:
        """
        'y' 는 RSU 스케줄링 결정 Term 으로 (M x N) array -> y[m][n] = 1 이면 RSU m 이 사용자 n 으로 전송
        'ㅣ' 은 전송 청크 수 Term 으로 (M x N) array
        'k' 는 전송 layer 수 Term 으로 (M x N) array
        'u' 는 UAV 스케줄링 결정 Term 으로 (N, ) array -> u[n] = 1 이면 사용자 n 이 UAV 에게서 전송 받음
        """
        y = action['y'].astype(int)
        l = action['l'].astype(int)
        k = action['k'].astype(int)
        u = action['u'].astype(int)

        # 한 time-slot 에서 벌어지는 비디오 이벤트 모을 버퍼 느낌
        self.finished_events = []

        Z_prev = self.Z.copy()
        last_l_prev = self.last_l.copy()

        action_summary = self._action_summary(y, k, l, u)
        self.logger.debug(
            f"[t={self.t}] action | y_pairs={action_summary['y_pairs']}"
            f"| u_used={action_summary['u_used']} "
            f"| k_range=({action_summary['k_min']},{action_summary['k_max']}) | l_sum={action_summary['l_sum']}"
        )

        # RSU 전송 용량 확인
        for m in range(self.num_rsu):
            if np.sum(y[m]) > self.N0:
                raise ValueError(f"ERROR | RSU {m} 은 최대 {self.N0} 명의 사용자만을 스케줄링합니다.")
        
        # UAV 사용 가능 대수 확인
        if np.sum(u) > self.uav_total:
            raise ValueError(f"ERROR | UAV 는 최대 {self.uav_total} 개만을 사용합니다.")
        
        # RSU 별 UAV 사용량 확인
        self.u_current = u.astype(int).copy() # UAV 현재 사용 상태
        self.U_current[:] = 0
        for n in range(self.num_user):
            if self.u_current[n] == 1:
                m = int(self.user_region[n])
                self.U_current[m] += 1
        
        # 보상 계산 - 변수 초기화
        active = (self.active_mask == 1)

        stall_count = int(np.sum((self.queue == 0) & active)) # 강한 재생 중단 패널티
        small_buffer_mask = (self.queue > 0) & (self.queue <= 5) & active

        root_vals = np.sqrt(self.queue)
        small_buffer_sum = int(np.sum(root_vals * small_buffer_mask)) # 약한 재생 중단 패널티

        quality_degradation = 0 # 품질 저하 패널티
        uav_cost = int(np.sum(self.u_current)) # UAV 패널티
        
        # MBS 처리
        arrived_mbs = self._collect_mbs_arrivals()
        if np.any(arrived_mbs > 0):
            for n, c in enumerate(arrived_mbs):
                if c > 0:
                    self.logger.info(f"[t={self.t}] MBS Arrival | User {n} + {int(c)} chunk")

        # 전송 처리
        delivered_users = np.zeros(self.num_user, dtype=np.int32) # 이번 time-slot 에 도착한 청크의 수
        delivered_layers = np.zeros(self.num_user, dtype=np.int32) # 각 사용자에게 전달된 레이어 수

        queue_before = self.queue.copy() # 디버그용
        # 제약 추가
        # 한 User 는 오직 하나의 RSU 만 스케줄링됨
        if np.any(np.sum(y, axis=0) > 1):
            raise ValueError("ERROR | 한 사용자를 여러 RSU 가 동시에 스케줄링했습니다.")

        for m in range(self.num_rsu):
            for n in range(self.num_user):
                if y[m, n] == 1: # RSU 가 사용자를 대상으로 스케줄링 했을 때

                    if self.user_region[n] != m: # 사용자가 RSU m 의 영역에 없다면 패스
                        self.logger.debug(f"RSU {m} -> Pass (user {n} not in region)")
                        continue

                    if self.needed_chunks[n] <= 0: # 사용자가 더 받아야 할 청크가 없다면 패스
                        self.logger.debug(f"RSU {m} -> Pass (user {n} no needed chunks)")
                        continue
                    
                    vid = int(self.user_requests[n])
                    cache_hit = (vid in self.rsu_cache[m])
                    layers = int(np.clip(k[m, n], 1, self.layer))
                    chunks_req = int(max(0, l[m, n]))
                    if chunks_req == 0:
                        continue

                    if self.u_current[n] == 1: # UAV 가 사용자에게 전송한다면 (Case3)
                        if not cache_hit:
                            self.logger.debug(f"UAV SKIP | RSU {m} has no cache for user {n}")
                            continue
                        
                        # channel cap 을 고려하여 layer 수와 청크 수를 결정 (UAV)
                        C_UAV = self.channel_uav._sample_channel_capacity()
                        max_by_channel = int(C_UAV // max(layers, 1)) # M * K <= C 로부터 M <= (C / K)
                        deliverable = max(0, min(chunks_req, int(self.needed_chunks[n]), max_by_channel)) # 비디오 청크 수에 clipping 제한

                        if deliverable > 0:
                            delivered_users[n] += deliverable
                            delivered_layers[n] += deliverable * layers
                            self.queue[n] += deliverable # 큐에 저장
                            self.needed_chunks[n] -= deliverable
                            quality_degradation += deliverable * (self.layer - layers) # quality degradation

                            self.received_chunks[n] += deliverable # 현재 청크 받은 진행도
                            if self.needed_chunks[n] <= 0:
                                self.on_finish_video(n, source="UAV") # 다 받았을 때
                            self.logger.info(f"[UAV 이용] [t={self.t}] RSU {m} | UAV -> user {n} 청크{deliverable} 개(l), 레이어 수 {layers} 개(k) 전송 성공 | l*k = {deliverable * layers}, C={C_UAV:.2f}")
                        continue
                        
                    if cache_hit: # RSU 가 직접 전송한다면 (Case 1)

                        # channel cap 을 고려하여 layer 수와 청크 수를 결정 (RSU)
                        C_RSU = self.channel_rsu._sample_channel_capacity()
                        max_by_channel = int(C_RSU // max(layers, 1)) # M * K <= C 로부터 M <= (C / K)
                        deliverable = max(0, min(chunks_req, int(self.needed_chunks[n]), max_by_channel)) # 비디오 청크 수에 clipping 제한

                        if deliverable > 0:
                            delivered_users[n] += deliverable
                            delivered_layers[n] += deliverable * layers
                            self.queue[n] += deliverable # 큐에 저장
                            self.needed_chunks[n] -= deliverable
                            quality_degradation += deliverable * (self.layer - layers)

                            self.received_chunks[n] += deliverable # 현재 청크 받은 진행도
                            if self.needed_chunks[n] <= 0:
                                self.on_finish_video(n, source="RSU") # 다 받았을 때
                            self.logger.info(f"[RSU 이용] [t={self.t}] RSU {m} -> user {n} 청크 {deliverable} 개(l), 레이어 수 {layers} 개(k) 전송 성공 | l*k = {deliverable * layers}, C={C_RSU:.2f}")
                        continue

                    if chunks_req > 0: # RSU 이용 / Cache miss 로 인한 MBS 사용 (Case 2)
                        for _ in range(chunks_req):
                            # 딜레이가 끝나야 ready queue 로 감
                            self.mbs_in_delay[n].append(self.t + self.mbs_delay)

                        self.logger.info(f"[MBS 이용] [t={self.t}] user {n} 캐시 미스 | MBS scheduled {chunks_req} chunks, first arrival t >= {self.t + self.mbs_delay}")
                        

                    # 일단 제외해보고..
                    # else: # RSU 가 사용자에게 청크 전송한다면
                    #     video_id = self.user_requests[n]
                    #     cache_hit = video_id in self.rsu_cache[m] # RSU 가 캐싱하고 있는 지 확인
                    #     if cache_hit: 
                    #         layers = max(1, min(int(k[m, n]), self.layer))
                    #         delivered_users[n] = 1
                    #         delivered_layers[n] = layers
                    #         if layers < self.layer:
                    #             quality_degradation += (self.layer - layers)
                    #         self.logger.info(
                    #             f"RSU {m} -> user {n} 전송 성공 (Cache hit, Video{video_id}) | layers={layers}"
                    #         )
                    #     else: # 캐시 미스 -> RSU 이용
                    #         delivered_users[n] = 1
                    #         delivered_layers[n] = 1 # 지연 발생 내재
                    #         quality_degradation += (self.layer - 1)
                    #         self.logger.info(
                    #             f"RSU {m} -> user {n} Cache miss, Video {video_id}"
                    #         )
                        
                    # if delivered_users[n] == 1: # 청크 전송 완료 처리
                    #     self.needed_chunks[n] -= 1
                
        # # 전달된 청크를 사용자 큐에 추가
        # for n in range(self.num_user):
        #     if delivered_users[n] == 1:
        #         self.queue[n] += 1
        
        # 로그 출력
        if self.finished_events:
            for ev in self.finished_events:
                user = ev["user"]
                src = ev["source"]
                old_vid = ev["old_vid"]
                old_chunks = ev["old_chunks"]
                new_vid = ev["new_vid"]
                new_chunks = ev["new_needed_chunks"]

                if old_chunks is not None:
                    self.logger.info(
                        f"[t={self.t}] [FINISH] user {user} 모든 청크 수신 완료 "
                        f"(source={src}) | 이전 video = {old_vid} (chunks = {old_chunks}) -> "
                        f"인기도 기반 새 video = {new_vid} 시작 (needed_chunks = {new_chunks})"
                    )
            
            self.finished_events = []


        M_eff = (arrived_mbs + delivered_users).astype(np.int32) 

        # 이번 슬롯에 실제로 도착한 청크의 품질 구성
        l_now = last_l_prev.copy()
        for n in range(self.num_user):
            if not self.active_mask[n]:
                continue

            if delivered_users[n] > 0:
                m = int(self.user_region[n])
                l_now[n] = int(k[m, n]) # layer
            
            elif arrived_mbs[n] > 0:
                l_now[n] = 2 # MBS 에 의해 받은 품질은 일단 임의로 가정
        
        # DPP 계산
        l_idx = np.clip(l_now - 1, 0, len(self.quality_levels) - 1)
        P_l = self.quality_levels[l_idx]
        quality_term = (self.quality_max - P_l) * M_eff

        fluct_term = self.phi * np.abs(l_now - last_l_prev) * (M_eff > 0).astype(np.int32)
        drift_term = Z_prev * (self.playback_b - M_eff)

        dpp = (drift_term + self.V * (quality_term + fluct_term)) * self.active_mask.astype(dpp.dtype)

        # Virtual Queue 업데이트
        Z_next = np.minimum(Z_prev + self.playback_b, self.queue_bound) - M_eff
        Z_next = np.clip(Z_next, 0, self.queue_bound).astype(np.int32)
        
        Z_next = Z_next * self.active_mask.astype(np.int32)
        self.Z[:] = Z_next

        # last_l 업데이트
        arrived_mask = (M_eff > 0) & (self.active_mask == 1) # 받은게있어야
        self.last_l[arrived_mask] = l_now[arrived_mask]

        # 비디오 재생 시 큐 감소
        for n in range(self.num_user):
            if self.queue[n] > 0:
                self.queue[n] -= 1
        
        # 사용자 이동 (FSMC)
        self.t += 1
        self.fsmc_step()
        departed_info, spawned_info = self.spawn_depart_users()

        # 에피소드 종료 판단 부분
        done = False
        done_reason: Optional[str] = None
        if self.t >= self.max_steps:
            done = True
            done_reason = "max_steps"
        if np.all(self.needed_chunks <= 0):
            done = True
            done_reason = done_reason or "all_chunks_delivered"

        log_t = self.t - 1

        # 1) 큐 변화 로그
        self.logger.info(
            f"[t={log_t}] User's Queue {queue_before.tolist()} -> {self.queue.tolist()}"
        )
        if done_reason:
            self.logger.info(f"[t={log_t}] done: {done_reason}")

        # 2) USER-OUT / USER-IN / USER-FLOW 로그
        for (u, region, vid, q, need) in departed_info:
            self.logger.info(
                f"[t={log_t}] [USER-OUT] user {u} | region={region} "
                f"| video={vid} | Queue={q} | Needed_chunks={need}"
            )
        for (u, region, vid, need) in spawned_info:
            self.logger.info(
                f"[t={log_t}] [USER-IN] user {u} | region={region} "
                f"| new_video={vid} | Init_Queue={self.init_q} | Needed_chunks={need}"
            )

        if departed_info or spawned_info:
            active_cnt = int(np.sum(self.active_mask))
            self.logger.info(
                f"[t={log_t}] [USER-FLOW] USER_OUT={len(departed_info)}, "
                f"USER_IN={len(spawned_info)}, active={active_cnt}"
            )
        
        # 보상 계산할 부분
        # 보상 = playback stall (strong) + playback stall (weak) + quality degradation + UAV cost
        # dpp 추가
        Reward_s1 = stall_count
        Reward_s2 = small_buffer_sum
        Reward_q = quality_degradation
        Reward_u = uav_cost
        Reward_dpp = -float(np.sum(dpp))
        reward = Reward_dpp + (self.cs1 * Reward_s1) + (self.cs2 * Reward_s2) + (self.cq * Reward_q) + (self.cu * Reward_u)
    


        # 트레이스 기록
        if self.trace_enabled:
            self.trace.append({
                "t": int(self.t),
                "state": {"Q": queue_before.tolist(),
                          "U": self.U_current.tolist(),
                          "X": self.user_region.tolist()},
                "action": self._action_summary(y, k, l, u),
                "delivered_users": delivered_users.tolist(),
                "delivered_layers": delivered_layers.tolist(),
                "reward_terms": {
                    "stall_strong": int(Reward_s1),
                    "stall_weak": int(Reward_s2),
                    "quality_deg": int(Reward_q),
                    "uav_cost": int(Reward_u)
                },
                "reward": float(reward),
                "next_queue": self.queue.tolist(),
                "needed_chunks": self.needed_chunks.tolist()
            })

        next_state = self._get_state()
        return next_state, reward, done

    # 시각화
    def export_trace_json(self, path: Optional[str] = None) -> str:
        if path is None:
            path = os.path.join(self.save_dir, f"trace_ep{self.episode}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.trace, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Trace JSON saved: {path}")
        return path


    def plot_trace(self, save_dir: Optional[str] = None) -> None:
        if not self.trace:
            self.logger.warning("Trace is empty. Run some steps first.")
            return
        if save_dir is None:
            save_dir = self.save_dir
        ensure_dir(save_dir)

        T = [d["t"] for d in self.trace]
        total_Q = [int(np.sum(d["next_queue"])) for d in self.trace]
        stall_strong = [d["reward_terms"]["stall_strong"] for d in self.trace]
        stall_weak = [d["reward_terms"]["stall_weak"] for d in self.trace]
        q_deg = [d["reward_terms"]["quality_deg"] for d in self.trace]
        u_cost = [d["reward_terms"]["uav_cost"] for d in self.trace]
        R = [d["reward"] for d in self.trace]

        def _one_plot(xs, ys, title, fname):
            plt.figure()
            plt.plot(xs, ys)
            plt.title(title)
            plt.xlabel("t")
            plt.ylabel(title)
            path = os.path.join(save_dir, fname)
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            self.logger.info(f"Saved plot: {path}")

        _one_plot(T, total_Q, "Total Buffer (sum Q)", f"ep{self.episode}_sumQ.png")
        _one_plot(T, stall_strong, "Strong Stall Count", f"ep{self.episode}_stall_strong.png")
        _one_plot(T, stall_weak, "Weak Stall (small buffer sum)", f"ep{self.episode}_stall_weak.png")
        _one_plot(T, q_deg, "Quality Degradation", f"ep{self.episode}_qdeg.png")
        _one_plot(T, u_cost, "UAV Cost", f"ep{self.episode}_uav_cost.png")
        _one_plot(T, R, "Reward", f"ep{self.episode}_reward.png")

    def render_text(self, last_k: int = 5) -> None:
        """최근 last_k 스텝 트레이스 요약을 텍스트로 출력"""
        k = min(last_k, len(self.trace))
        for d in self.trace[-k:]:
            self.logger.info(
                f"[t={d['t'] - 1}] Queue_Sum={int(np.sum(d['next_queue']))} | "
                f"RSU 스케줄링 사용자 수={d['action']['y_pairs']}, UAV 사용자 수={d['action']['u_used']} | "
                f"stall(S/W)={d['reward_terms']['stall_strong']}/{d['reward_terms']['stall_weak']} | "
                f"Q Deg 합={d['reward_terms']['quality_deg']} | R={d['reward']:.3f}"
            )

    def sample_random_action(self) -> Dict[str, np.ndarray]:
        """
        제약을 만족하는 랜덤 액션 생성:
        - y[m,n]∈{0,1}: RSU m이 사용자 n을 이 슬롯에 서비스
        - k[m,n]∈{1..layer}: (전송 레이어 수) – step()에서 layers=int(clip(k,1,layer))로 사용
        - l[m,n]∈{0,1,2,...}: (전송 청크 수)
        - u[n]∈{0,1}: UAV가 사용자 n을 서비스
        제약:
        - RSU m은 최대 N0명의 사용자만 선택
        - RSU별 Σ_n l[m,n] ≤ C[m]
        - 사용자 n은 활성(active_mask[n]==1) & needed_chunks[n]>0 & 지역 일치(user_region[n]==m)
        - UAV는 총 uav_total 이하, 후보는 (해당 RSU 캐시 히트 & 활성 & needed>0 & y[m,n]==1)에서 선택
        - 사용자별 l[m,n] ≤ needed_chunks[n] (잔여 필요 청크 초과 방지)

        반환: {"y": (M,N), "l": (M,N), "k": (M,N), "u": (N,)}
        """
        import numpy as np

        M = int(self.num_rsu)
        N = int(self.num_user)
        layer_max = int(self.layer)

        y = np.zeros((M, N), dtype=int)
        k = np.ones((M, N), dtype=int)           # 기본 1로 두고, 실제 선택된 페어에 랜덤 재할당
        l = np.zeros((M, N), dtype=int)
        u = np.zeros(N, dtype=int)

        rng = np.random.default_rng()

        # ---------- (1) RSU별 사용자 선택: 지역 일치 & 활성 & 필요 청크 ----------
        for m in range(M):
            # 후보: 이 RSU 영역에 있고, 활성이며, 아직 받아야 할 청크가 남아있는 사용자
            candidates = [n for n in range(N)
                        if (self.active_mask[n] == 1)
                        and (int(self.user_region[n]) == m)
                        and (int(self.needed_chunks[n]) > 0)]
            if not candidates:
                continue
            rng.shuffle(candidates)
            chosen = candidates[: int(self.N0)]  # RSU m의 동시 서비스 사용자 수 제한
            for n in chosen:
                y[m, n] = 1

        # ---------- (2) k, l 할당: RSU별 청크 예산 C[m] 내에서 분배 ----------
        # k는 선택된 (m,n)에 대해서만 1..layer_max에서 균등 샘플
        # l는 각 RSU의 cap=C[m]을 랜덤 라운드로 사용자들에게 1씩 분배하되
        #    각 사용자 잔여 필요 청크(needed_chunks[n])를 넘지 않게 클립
        for m in range(M):
            users_m = np.where(y[m] == 1)[0]
            if users_m.size == 0:
                continue

            # 우선 k를 무작위로 부여(1..layer_max)
            k[m, users_m] = rng.integers(low=1, high=layer_max + 1, size=users_m.size)

            # RSU m의 청크 총예산
            cap = int(self.C[m])
            if cap <= 0:
                continue

            # 사용자별 최대 가능 l 상한(= 남은 필요 청크)
            # (동일 슬롯에서 타 RSU가 같은 사용자에 l을 주는 경우는 지역상 불가하므로 단순화 가능)
            max_l_per_user = {int(n): int(self.needed_chunks[n]) for n in users_m}

            # 라운드-로빈에 약간의 랜덤성을 더해 분배
            user_list = users_m.tolist()
            rng.shuffle(user_list)

            # cap 소진 또는 모든 사용자의 상한이 0이 되면 종료
            while cap > 0 and user_list:
                progressed = False
                for n in list(user_list):
                    # 이 사용자에 대해 현재까지 분배한 양
                    cur_l = int(l[m, n])
                    # 해당 사용자에 허용되는 최대치
                    max_allow = max(0, max_l_per_user[n] - cur_l)
                    if max_allow <= 0:
                        # 더 줄 수 없으면 후보에서 제거
                        user_list.remove(n)
                        continue
                    # 이번 라운드에 1청크 할당
                    l[m, n] = cur_l + 1
                    cap -= 1
                    progressed = True
                    if cap <= 0:
                        break
                if not progressed:
                    # 더 이상 분배 불가
                    break

        # ---------- (3) UAV 선택: 캐시 히트 사용자 + RSU가 스케줄한 사용자 중에서 ----------
        # (step()에서 UAV 경로는 RSU 캐시 히트가 아니면 스킵되므로, 후보를 캐시 히트로 한정)
        u_candidates = set()
        for m in range(M):
            users_m = np.where(y[m] == 1)[0]
            if users_m.size == 0:
                continue
            for n in users_m:
                if (self.active_mask[n] != 1) or (self.needed_chunks[n] <= 0):
                    continue
                vid = int(self.user_requests[n])
                if vid in self.rsu_cache[m]:  # 캐시 히트 사용자만 UAV 후보
                    u_candidates.add(int(n))

        u_candidates = list(u_candidates)
        if u_candidates:
            rng.shuffle(u_candidates)
            pick = u_candidates[: int(self.uav_total)]
            # 선택 사용자 중 일부만 실제 UAV 사용하도록 약간 랜덤성 부여(과소/과다 방지)
            for n in pick:
                # 50% 확률로 UAV 사용 (원하면 확률 조정)
                if rng.random() < 0.5:
                    u[n] = 1

        return {"y": y, "l": l, "k": k, "u": u}


    def run_random(self, steps: int = 100, save: bool = True) -> None:
        """랜덤 정책으로 steps 스텝 실행하고 결과 저장"""
        self.reset()
        for _ in range(steps):
            a = self.sample_random_action()
            _, _, done = self.step(a)
            if done:
                break
        if save:
            self.export_trace_json()
            self.plot_trace()

# MWE
if __name__ == "__main__":
    env = Env(loglevel=logging.INFO)
    env.run_random(steps=10, save=True)
    env.render_text(last_k=10)