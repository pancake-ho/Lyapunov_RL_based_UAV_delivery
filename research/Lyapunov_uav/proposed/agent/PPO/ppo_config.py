from dataclasses import dataclass
import torch


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    batch_size: int = 256
    hidden_dim: int = 256
    rollout_steps: int = 1024
    total_updates: int = 1000
    save_interval: int = 50
    log_interval: int = 10
    seed: int = 2026
    device: str = "cuda" if torch.cuda.is_available() else "cpu"