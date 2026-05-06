from typing import Optional
import numpy as np

try:
    from proposed.config import ChannelConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import ChannelConfig


class BaseChannelModel:
    """
    공통 무선 채널 (Rayleigh Fading) gain 샘플링 베이스 클래스로
    normalized channel power gain 생성을 담당
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

        if self.beta <= 0:
            raise ValueError(f"beta는 0보다 커야 합니다, 현재 값은 {self.beta}입니다.")
        if self.min_distance <= 0:
            raise ValueError(f"min_distance는 0보다 커야 합니다, 현재 값은 {self.min_distance}입니다.")
        if self.distance < 0:
            raise ValueError(f"distance는 0 이상이어야 합니다, 현재 값은 {self.distance}입니다.")
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidt는 0보다 커야 합니다, 현재 값은 {self.bandwidth}입니다.")
        
        seed = getattr(config, "seed", None)
        self.rng = np.random.default_rng(seed)

        # normalized noise power used by delivery/battery link info.
        self.noise_power = 1.0 / max(self.db_to_linear(self.gamma_db), 1e-12)
        
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
        return rng if rng is not None else self.rng
    
    def sample_shadowing_linear(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        linear scale에서 Log-normal shadowing factor를 계산하는 함수
        """
        generator = self._rng(rng)
        shadow_db = generator.normal(
            loc=self.shadowing_mu_db, 
            scale=self.shadowing_sigma_db,
        )
        return float(self.db_to_linear(shadow_db))
    
    def sample_small_scale_fading(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        Rayleigh fading power gain을 계산하는 함수
        """
        generator = self._rng(rng)

        # Rayleigh fading power gain |h|^2 follows Exponential(mean=1)
        return float(generator.exponential(scale=1.0))
    
    def compute_pathloss(self, distance: Optional[float] = None) -> float:
        d = max(float(distance if distance is not None else self.distance), self.min_distance)
        return float(d ** self.beta)
    
    def sample_channel_gain(self, distance: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> float:
        shadowing = self.sample_shadowing_linear(rng=rng)
        fading = self.sample_small_scale_fading(rng=rng)
        pathloss = self.compute_pathloss(distance=distance)
        return float((shadowing * fading) / pathloss)
