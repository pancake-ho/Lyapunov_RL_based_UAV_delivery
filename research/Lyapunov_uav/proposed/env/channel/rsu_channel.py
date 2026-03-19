import math
from typing import Optional

import numpy as np
from config import ChannelConfig
from .base_channel import BaseChannelModel

class RSUChannelModel(BaseChannelModel):
    """
    RSU에 대한 무선 채널 (Rayleigh Fading) 시뮬레이션 클래스
    """
    def __init__(self, config: ChannelConfig, tx_power: float = 1.0):
        super().__init__(config)
        self.tx_power = float(tx_power)
    
    def compute_gain(
        self,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        return self.sample_channel_gain(distance=distance, rng=rng)
    
    def compute_snr(
        self,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        RSU의 Transmit SNR을 계산하는 함수로,
        RSU는 고정된 TX power를 사용한다고 가정한다.
        """
        gain = self.compute_gain(distance=distance, rng=rng)
        reference_snr = self.db_to_linear(self.gamma_db)
        return max(0.0, self.tx_power) * reference_snr * gain
    
    def capacity(
        self,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        snr = self.compute_snr(distance=distance, rng=rng)
        return self.bandwidth * math.log2(1.0 + snr)