import math
from typing import Optional

import numpy as np
from config import ChannelConfig

class ChannelModel:
    """
    무선 채널 (Rayleigh Fading) 시뮬레이션 클래스
    RSU, UAV 에 대한 Channel Capacity 할당
    (가정) 거리는 20
    """
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.distance = float(config.distance)
        self.bandwidth = float(config.bandwidth)
        self.gamma_db = float(config.gamma_db)
        self.sigma_db = float(config.sigma_db)
        self.beta = float(config.beta)
        self.mu_db = float(config.mu_db)
        self.min_distance = float(config.min_distance)

    @staticmethod
    def db_to_linear(db_value: float) -> float:
        """
        단위를 db(데시벨)에서 linear로 변경해 주는 함수
        """
        return 10.0 ** (db_value / 10.0)
    
    def _rng(self, rng: Optional[np.random.Generator]) -> np.random.Generator:
        """
        난수 생성기 함수
        """
        return rng if rng is not None else np.random.default_rng
    
    def sample_shadowing_linear(self, rng: Optional[np.random.Generator] = None) -> float:
        generator = self._rng(rng)
        shadow_db = generator.normal(loc=self.mu_db, scale=self.sigma_db)
        return self.db_to_linear(float(shadow_db))
    
    def sample_small_scale_fading(self, rng: Optional[np.random.Generator] = None) -> float:
        generator = self._rng(rng)

        # Rayleigh Fading에서, amplitude는 mean 1의 exponential power gain으로 정의됨
        return float(generator.exponential(scale=1.0))
    
    def sample_channel_gain(self, distance: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> float:
        d = max(float(distance if distance is not None else self.distance), self.min_distance)
        shadowing = self.sample_shadowing_linear(rng=rng)
        fading = self.sample_small_scale_fading(rng=rng)
        pathloss = d ** self.beta
        return (shadowing * fading) / pathloss
    
    def capacity(
        self, 
        tx_power: float, 
        distance: Optional[float] = None, 
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        if tx_power <= 0.0:
            return 0.0
        
        reference_snr = self.db_to_linear(self.gamma_db)
        channel_gain = self.sample_channel_gain(distance=distance, rng=rng)
        snr = max(0.0, float(tx_power)) * reference_snr * channel_gain
        return self.bandwidth * math.log2(1.0 + snr)
    