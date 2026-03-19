import numpy as np
from config import PPOConfig
from network import ActorCritic
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from buffer import RolloutBuffer

EPS = 1e-6

class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: PPOconfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = ActorCritic(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
    
    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        """
        policy는 raw Gaussian에서 샘플링 후 tanh 해서 action을 만듦.
        즉, update때 저장된 action은 이미 squased action.
        따라서 다시 raw Gaussian으로 되돌리기 위해 atanh를 구현하는 함수임.
        """
        x = torch.clamp(x, -1.0 + EPS, 1.0 + EPS)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))
    
    @staticmethod
    def _squash_log_prob(dist: Normal, raw_action: torch.Tensor, squashed_action: torch.Tensor) -> torch.Tensor:
        """
        tanh transform의 Jacobian 보정을 구현하는 함수로,
        원래 raw Gaussian log_prob만 쓰기에는 실제 policy가 tanh 변환되었기 때문에 보정이 필요
        """
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - squashed_action.pow(2) + EPS)
        return log_prob.sum(dim=-1)
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        env와 상호작용할 때 호출하는 함수
        """
        obs_torch = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) # batch 차원 추가
        dist, mean, value = self.model.get_dist(obs_torch)
        raw_action = mean if deterministic else dist.rsample() # action 샘플링 (deterministic이면 mean)
        action = torch.tanh(raw_action) # [-1, 1] 범위의 action 생성
        log_prob = self._squash_log_prob(dist, raw_action, action)
        return (
            action.squeeze(0).detach().cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )
    
    def value(self, obs: np.ndarray) -> float:
        """
        buffer의 rollout 끝에서 bootstrap value 계산할 때 사용하는 함수
        """
        obs_torch = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, value = self.model.get_dict(obs_torch)
        return float(value.item()) 
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO 업데이트에서 사용하는 함수
        1. 입력
        - obs: 배치 상태
        - actions: 이미 저장된 squashed action

        2. 흐름
        - 현재 policy 분포 dist와 value 계산
        - raw action 만들고, 현재 policy 기준 log_prob 계산
        - 최종적으로, entropy 계산 및 value 반환
        """
        dist, _, values = self.model.get_dist(obs)
        raw_action = self._atanh(actions)
        log_prob = self._squash_log_prob(dist, raw_action, actions)
        entropy = dist.entropy().sum(dim=-1) # 이 entropy는 raw Gaussian의 entropy
        return log_prob, values, entropy
    
    def update(self, rollout: RolloutBuffer) -> Dict[str, float]:
        """
        PPO 학습 함수
        """
        # 1. 데이터 준비
        batch = rollout.get_tensors(self.device) # rollout 전체를 batch로 바꿈
        num_samples = batch["obs"].shape[0]
        batch_size = min(self.cfg.batch_size, num_samples) # 미니배치 학습 준비

        # 학습 모니터링용
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl_div": 0.0,
            "clip_frac": 0.0,
        }
        num_updates = 0

        # 여러 epochs 반복
        for _ in range(self.cfg.update_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                obs_batch = batch["obs"][batch_idx]
                act_batch = batch["actions"][batch_idx]
                old_log_prob_batch = batch["log_probs"][batch_idx]
                returns_batch = batch["returns"][batch_idx]
                adv_batch = batch["advantages"][batch_idx]

                new_log_prob, entropy, value = self.evaluate_actions(obs_batch, act_batch)
                ratio = torch.exp(new_log_prob - old_log_prob_batch)
                unclipped = ratio * adv_batch
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                
                policy_loss = -torch.mean(torch.min(unclipped, clipped))
                value_loss = F.mse_loss(value, returns_batch)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_prob - old_log_prob_batch
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.cfg.clip_ratio).float())
                
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.mean().item())
                stats["approx_kl_div"] += float(approx_kl_div.item())
                stats["clip_frac"] += float(clip_frac.item())
                num_updates += 1

        for k in stats:
            stats[k] /= max(num_updates, 1)
        return stats

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimzier.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )
    
    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])