import math
import random
from dataclasses import dataclass
from typing import Callable, List, Dict

# 데이터 구조 정의
@dataclass
class QueueSamplePath:
    arrivals: List[int]
    services: List[int]
    backlog: List[int]

class SingleServerQueue:
    """ 단일 서버 큐 구현 """
    def __init__(self, arrival_fn: Callable[[], int], service_fn: Callable[[], int], q0: int=0, seed: int=0):
        """
        arrval_fn: 한 슬롯의 A(t) (도착할 패킷 수) 를 리턴하는 함수
        service_fn: 한 슬롯의 μ(t) (서비스 가능한 패킷 수) 를 리턴하는 함수
        q0: 초기 큐 길이
        """
        self.arrival_fn = arrival_fn
        self.service_fn = service_fn
        self.q = max(q0, 0)
        random.seed(seed)
    
    def step(self):
        """
        한 슬롯 진행
        return : (A(t), u(t), Q(t+1))
        """
        a = max(int(self.service_fn()), 0)
        mu = max(int(self.service_fn()), 0)

        served = min(self.q, mu)
        self.q = self.q - served + a
        
        return a, mu, self.q
    
    def simulate(self, T: int) -> QueueSamplePath:
        """
        T 슬롯 동안 시뮬레이션
        Q(0) 은 backlog[0] 으로 기록, 이후 Q(1), ..., Q(T) 기록
        """
        arrivals = []
        services = []
        backlog = [self.q]

        for _ in range(T):
            a, mu, q_next = self.step()
            arrivals.append(a)
            services.append(mu)
            backlog.append(q_next)
        
        return QueueSamplePath(arrivals=arrivals, services=services, backlog=backlog)

# 확률 과정들

def bernoulli_process(p: float):
    """
    매 슬롯마다 1이 올 확률 p, 0이 올 확률 1-p
    평균은 p
    """
    def _sample():
        return 1 if random.random() < p else 0
    
    return _sample

def poisson_process(lamda: float):
    """
    평균 lamda 인 Poisson 과정
    정수 패킷 수를 반환
    """
    def _sample():
        # 간단한 Poisson 샘플링
        L = math.exp(-lamda)
        k = 0
        p_val = 1.0
        while p_val > L:
            k += 1
            p_val *= random.random()
        return k - 1
    
    return _sample

# 통계량 계산

def summarize_sample_path(sp: QueueSamplePath):
    T = len(sp.arrivals) # 슬롯의 개수
    total_arrivals = sum(sp.arrivals)
    total_service = sum(sp.services)
    avg_arrival_rate = total_arrivals / T
    avg_service_rate = total_service / T

    # Q(t) 는 길이가 T+1
    avg_backlog = sum(sp.backlog) / len(sp.backlog)

    # rate stability 근사
    final_Q_over_T = sp.backlog(-1) / T

    return {
        "T": T,
        "avg_arrival_rate": avg_arrival_rate,
        "avg_service_rate": avg_service_rate,
        "avg_backlog": avg_backlog,
        "final_Q": sp.backlog[-1],
        "final_Q_over_T": final_Q_over_T
    }

def print_summary(name: str, stats: Dict[str, float]):
    print(f"\n=== {name} ===")
    print(f"T                       = {stats['T']}")
    print(f"평균 도착률 λ̂          = {stats['avg_arrival_rate']:.4f}")
    print(f"평균 서비스률 μ̂        = {stats['avg_service_rate']:.4f}")
    print(f"평균 큐 길이 E[Q] 근사 = {stats['avg_backlog']:.4f}")
    print(f"마지막 Q(T)            = {stats['final_Q']}")
    print(f"Q(T)/T (rate stability) = {stats['final_Q_over_T']:.6f}")

def main():
    T = 50000

    # Case 1
    