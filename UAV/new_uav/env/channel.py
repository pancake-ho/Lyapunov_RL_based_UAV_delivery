from config import ChannelConfig

class ChannelModel:
    """
    무선 채널 (Rayleigh Fading) 시뮬레이션 클래스
    RSU, UAV 에 대한 Channel Capacity 할당
    (가정) 거리는 20
    """
    def __init__(self, config: ChannelConfig):
        self.config = config

        self.distance = self.config.distance
        self.bandwidth = self.config.bandwidth
        self.gamma_db = self.config.gamma_db
        self.sigma_db = self.config.sigma_db
        self.gamma_db = self.config.gamma_db
        self.mu_db = self.config.mu_db