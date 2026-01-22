class ChannelModel:
    """
    무선 채널 (Rayleigh Fading) 시뮬레이션 클래스
    RSU, UAV 에 대한 Channel Capacity 할당
    (가정) 거리는 20
    """
    def __init__(self, distance: float=20, bandwidth: float=1e6, gamma_db: int=35,
                 beta: int=2, sigma_db: int=4, mu_db: int=0):
        self.distance = distance
        self.bandwidth = bandwidth
        self.gamma = 10 ** (gamma_db / 10)
        self.sigma_db = sigma_db
        self.gamma_db = gamma_db
        self.mu_db = mu_db
