import numpy as np
from config import PPOConfig
from network import ActorCritic
from typing import Tuple

import torch
import torch.optim as optim
from torch.distributions import Normal

EPS = 1e-6

class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: PPOconfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = ActorCritic(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
    
    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0 + EPS, 1.0 + EPS)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))
    
    @staticmethod
    def _squash_log_prob(dist: Normal, raw_action: torch.Tensor, squashed_action: torch.Tensor)
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        obs_torch = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, value = self.model.get_dist(obs_torch)
        return float(value.item())
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, _, values = self.model.get_dist(obs)
        raw_action = self._atanh(actions)
        log_prob = self.