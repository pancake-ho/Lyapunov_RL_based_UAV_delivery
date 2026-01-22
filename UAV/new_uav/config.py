from dataclasses import dataclass

@dataclass
class EnvConfig:
    # 시스템 설정
    num_user: int = 10
    num_rsu: int = 10
    num_uav: int = 10
    N0: int = 3

    # 비디오 및 캐싱
    num_video: int = 3
    rsu_caching: int = 2
    layer: int = 5
    chunk: int = 5
    rsu_capacity: int = 3
    mbs_capacity: int = 3
    