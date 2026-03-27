import math
from typing import Optional

import numpy as np
from config import ChannelConfig
from .base_channel import BaseChannelModel


class RSUChannelModel(BaseChannelModel):
    """
    RSU-Vehicle 지상 link용 channel model 클래스로,
    BaseChannelModel class가 생성한 normalized channel power gain을 사용하여
    SNR 및 Shannon Capacity 계산을 담당.
    """
    def __init__(self, config: ChannelConfig, tx_power: float = 1.0):
        super().__init__(config)
        self.tx_power = float(tx_power) # 현재 tx_power는 임의로 가정함

        if self.tx_power < 0:
            raise ValueError(f"tx_power는 0 이상이어야 합니다, 현재 값은 {self.tx_power}입니다.")
    
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
        RSU link instantaneous SNR을 계산하는 함수
        """
        gain = self.compute_gain(distance=distance, rng=rng)
        return self.snr_from_gain(gain)
    
    def snr_from_gain(self, gain: float) -> float:
        """
        gain으로부터 SNR을 계산하는 함수
        """
        reference_snr = self.db_to_linear(self.gamma_db)
        return float(self.tx_power * reference_snr * max(0.0, float(gain)))
    
    def capacity(
        self,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Shannon Capacity[bps] 또는 config.bandwidth 단위에 따라 일관된 rate를 계산하는 함수
        """
        gain = self.compute_snr(distance=distance, rng=rng)
        return self.capacity_from_gain(gain)
    
    def capacity_from_gain(
        self,
        gain: float,
    ) -> float:
        """'
        gain으로부터 Capacity를 계산하는 함수
        """
        snr = self.snr_from_gain(gain)
        return float(self.bandwidth * math.log2(1.0 + snr))