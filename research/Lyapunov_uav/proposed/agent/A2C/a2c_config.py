from dataclasses import dataclass


@dataclass
class A2CConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    device: str = "cpu"

    # 검증
    def __post_init__(self) -> None:
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma는 (0, 1] 범위 내에 존재해야 합니다, got {self.gamma}")
        
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda는 [0, 1] 범위 내에 존재해야 합니다, got {self.gae_lambda}")
        
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate는 양수 값을 가져야 합니다, got {self.learning_rate}")
        
        if self.value_coef < 0.0:
            raise ValueError(f"value_coef는 0 이상의 값을 가져야 합니다, got {self.value_coef}")
        
        if self.entropy_coef < 0.0:
            raise ValueError(f"entropy_coef는 0 이상의 값을 가져야 합니다, got {self.entropy_coef}")
        
        if self.max_grad_norm <= 0.0:
            raise ValueError(f"max_grad_norm은 양수 값을 가져야 합니다, got {self.max_grad_norm}")
        
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim은 양수 값을 가져야 합니다, got {self.hidden_dim}")
        
        if not isinstance(self.device, str) or len(self.device.strip()) == 0:
            raise ValueError(f"device는 string type 값을 가져야 합니다, got {self.device}")