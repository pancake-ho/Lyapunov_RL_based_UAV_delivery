from typing import Optional

import numpy as np
from config import ChannelConfig


class BaseChannelModel:
    """
    무선 채널 (Rayleigh Fading) 시뮬레이션 샘플링 공통부 클래스
    """
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.distance = float(config.distance)
        self.bandwidth = float(config.bandwidth)
        self.gamma_db = float(config.gamma_db)

        self.shadowing_sigma_db = float(config.sigma_db)
        self.shadowing_mu_db = float(config.mu_db)

        self.beta = float(config.beta)
        self.min_distance = float(config.min_distance)

    @staticmethod
    def db_to_linear(db_value: float) -> float:
        """
        단위를 db(데시벨)에서 linear로 변경해 주는 함수
        """
        return 10.0 ** (float(db_value) / 10.0)
    
    def _rng(self, rng: Optional[np.random.Generator]) -> np.random.Generator:
        """
        난수 생성기 함수
        """
        return rng if rng is not None else np.random.default_rng()
    
    def sample_shadowing_linear(self, rng: Optional[np.random.Generator] = None) -> float:
        generator = self._rng(rng)
        shadow_db = generator.normal(
            loc=self.shadowing_mu_db, 
            scale=self.shadowing_sigma_db,
        )
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
        return float((shadowing * fading) / pathloss)