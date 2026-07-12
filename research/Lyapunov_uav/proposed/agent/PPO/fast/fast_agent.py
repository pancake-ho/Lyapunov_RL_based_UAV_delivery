from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import EnvConfig

try:
    from agent.PPO.common import (
        ObsNormalizer,
        RolloutBuffer,
        explained_var,
        flatten_fast_obs,
        get_device,
        load_checkpoint,
        save_checkpoint,
        to_tensor,
    )
except ModuleNotFoundError:  # package 상대 import fallback
    from ..common import (
        ObsNormalizer,
        RolloutBuffer,
        explained_var,
        flatten_fast_obs,
        get_device,
        load_checkpoint,
        save_checkpoint,
        to_tensor,
    )

from .fast_action import FastActionCodec
from .fast_network import FastActorCritic


@dataclass
class FastPPOConfig:
    """
    Fast-timescale PPO hyperparameters.

    현재 시나리오 기준:
        수치 최적화 전 단계이므로, 기본값은 안정적인 training을 추구하는 쪽으로 설정함.
    """
    rollout_steps: int = 8192
    update_epochs: int = 4
    batch_size: int = 1024

    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-5
    max_grad_norm: float = 0.3

    clip_coef: float = 0.10
    value_coef: float = 0.05
    entropy_coef: float = 1e-3

    normalize_obs: bool = True
    normalize_adv: bool = True

    hidden_dims: Tuple[int, ...] = (256, 256)
    init_log_std: float = -1.0

    use_value_huber_loss: bool = True
    use_value_clip: bool = True
    value_clip_coef: float = 50_000.0

    fail_on_nan: bool = True
    target_kl: Optional[float] = 0.015

    device: str = "auto"


class FastPPOAgent:
    """
    Fast-timescale PPO Agent.

    slot-level action만 학습하는 agent로,
    slow decision은 obs에서 읽어서 FastActionCodec이 env action에 붙임.
    """
    def __init__(
        self,
        env_cfg: EnvConfig,
        obs_dim: int,
        ppo_cfg: Optional[FastPPOConfig] = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg or FastPPOConfig()

        self.device = get_device(self.ppo_cfg.device)
        self.codec = FastActionCodec(env_cfg)

        self.obs_dim = obs_dim
        self.action_dim = self.codec.action_dim

        self.model = FastActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.ppo_cfg.hidden_dims,
            init_log_std=self.ppo_cfg.init_log_std,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.ppo_cfg.lr,
            eps=1e-5,
        )

        self.buffer = RolloutBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=self.ppo_cfg.rollout_steps,
            device=self.device,
            gamma=self.ppo_cfg.gamma,
            gae_lambda=self.ppo_cfg.gae_lambda,
        )
        
        if self.ppo_cfg.normalize_obs:
            self.obs_normalizer: Optional[ObsNormalizer] = ObsNormalizer(obs_dim=self.obs_dim)
        else:
            self.obs_normalizer = None
        
    def _check_finite_array(self, name: str, arr: np.ndarray) -> None:
        if not bool(self.ppo_cfg.fail_on_nan):
            return

        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} contains NaN or Inf.")

    def _check_finite_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not bool(self.ppo_cfg.fail_on_nan):
            return

        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN or Inf.")

    def obs_to_vec(
        self,
        obs: Dict[str, Any],
        update_norm: bool = True,
    ) -> np.ndarray:
        """
        fast observation dict를 vector로 변환한다.

        반드시 flatten_fast_obs()를 사용하여 raw slow scheduling key가
        policy input에 섞이지 않도록 한다.
        """
        obs_vec = flatten_fast_obs(obs).astype(np.float32)

        if obs_vec.shape[0] != self.obs_dim:
            raise ValueError(
                f"obs dim mismatch: expected {self.obs_dim}, got {obs_vec.shape[0]}"
            )

        self._check_finite_array("obs_vec_before_norm", obs_vec)

        if self.obs_normalizer is not None:
            obs_vec = self.obs_normalizer.normalize(
                obs_vec,
                update=update_norm,
            )

        self._check_finite_array("obs_vec_after_norm", obs_vec)

        return obs_vec.astype(np.float32)
    
    @torch.no_grad()
    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        update_norm: bool = True,
    ) -> Dict[str, Any]:
        """
        실제 Fast Agent가 선택하는 action을 반환 및 env와 호환되게 설정.
        """
        obs_vec = self.obs_to_vec(
            obs,
            update_norm=update_norm,
        )

        action_mask = self.codec.build_action_mask(obs)

        obs_tensor = to_tensor(obs_vec, device=self.device).unsqueeze(0)
        mask_tensor = to_tensor(action_mask, device=self.device).unsqueeze(0)

        raw_action_tensor, log_prob_tensor, value_tensor = self.model.act(
            obs=obs_tensor,
            deterministic=deterministic,
            action_mask=mask_tensor,
        )

        raw_action = raw_action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        log_prob = float(log_prob_tensor.squeeze(0).detach().cpu().item())
        value = float(value_tensor.squeeze(0).detach().cpu().item())

        self._check_finite_array("raw_action", raw_action)

        env_action = self.codec.decode(raw_action, obs)

        active_action_dims = int(
            np.sum(action_mask)
        )

        active_action_ratio = float(
            np.mean(action_mask)
        )

        active_values = raw_action[
            action_mask > 0.0
        ]

        if active_values.size > 0:
            action_saturation_ratio = float(
                np.mean(
                    np.abs(active_values) >= 1.0
                )
            )
        else:
            action_saturation_ratio = 0.0

        return {
            "obs_vec": obs_vec,
            "raw_action": raw_action,
            "action_mask": action_mask,
            "env_action": env_action,
            "log_prob": log_prob,
            "value": value,
            "active_action_dims": (
                active_action_dims
            ),
            "active_action_ratio": (
                active_action_ratio
            ),
            "action_saturation_ratio": (
                action_saturation_ratio
            ),
        }
    
    def store_transition(
        self,
        obs_vec: np.ndarray,
        raw_action: np.ndarray,
        action_mask: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        RolloutBuffer에 transition 저장.
        """
        if self.ppo_cfg.fail_on_nan:
            if not np.isfinite(float(reward)):
                raise RuntimeError(
                    f"reward is NaN or Inf: {reward}"
                )
            if not np.isfinite(float(value)):
                raise RuntimeError(
                    f"value is NaN or Inf: {value}"
                )
            if not np.isfinite(float(log_prob)):
                raise RuntimeError(
                    "log_prob is NaN or Inf: "
                    f"{log_prob}"
                )

        self.buffer.add(
            obs=obs_vec,
            action=raw_action,
            action_mask=action_mask,
            reward=float(reward),
            done=bool(done),
            value=float(value),
            log_prob=float(log_prob),
        )

    @torch.no_grad()
    def estimate_value(
        self,
        obs: Dict[str, Any],
        update_norm: bool = False,
    ) -> float:
        """
        Input으로 주어지는 obs에 대하여 value 계산 후 detach/값 반환.
        """
        obs_vec = self.obs_to_vec(
            obs,
            update_norm=update_norm,
        )
        obs_tensor = to_tensor(obs_vec, device=self.device).unsqueeze(0)
        value = self.model.value(obs_tensor)

        return float(value.squeeze(0).detach().cpu().item())
    
    def finish_rollout(
        self,
        last_obs: Dict[str, Any],
        last_done: bool,
    ) -> None:
        """
        GAE 계산 및 last_done=False인 경우 value 추정까지 수행.
        """
        if last_done:
            last_value = 0.0
        else:
            last_value = self.estimate_value(
                last_obs,
                update_norm=False,
            )

        self.buffer.compute_returns_and_advs(
            last_value=last_value,
            last_done=last_done,
            normalize_adv=self.ppo_cfg.normalize_adv,
        )
    
    def _compute_value_loss(
        self,
        new_value: torch.Tensor,
        old_value: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        value clipping / huber option을 포함한 critic loss.
        """
        if self.ppo_cfg.use_value_clip:
            value_clipped = old_value + torch.clamp(
                new_value - old_value,
                -float(self.ppo_cfg.value_clip_coef),
                float(self.ppo_cfg.value_clip_coef),
            )

            if self.ppo_cfg.use_value_huber_loss:
                value_loss_unclipped = F.smooth_l1_loss(
                    new_value,
                    returns,
                    reduction="none",
                )
                value_loss_clipped = F.smooth_l1_loss(
                    value_clipped,
                    returns,
                    reduction="none",
                )
            else:
                value_loss_unclipped = (new_value - returns).pow(2)
                value_loss_clipped = (value_clipped - returns).pow(2)

            return torch.max(value_loss_unclipped, value_loss_clipped).mean()

        if self.ppo_cfg.use_value_huber_loss:
            return F.smooth_l1_loss(new_value, returns)

        return F.mse_loss(new_value, returns)
    
    def update(self) -> Dict[str, float]:
        """
        Buffer에서 experience를 꺼내어 Agent 업데이트 및 로그 확인용 정보 반환.
        """
        if len(self.buffer) == 0:
            raise RuntimeError("비어있는 buffer로는 PPO Agent를 update할 수 없습니다.")
        
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []
        approx_kl_values: list[float] = []
        clip_frac_values: list[float] = []

        early_stopped = False
        completed_minibatches = 0

        for _ in range(int(self.ppo_cfg.update_epochs)):
            for batch in self.buffer.iter_minibatches(
                batch_size=int(self.ppo_cfg.batch_size),
                shuffle=True,
                include_action_masks=True,
            ):
                new_log_prob, entropy, new_value = self.model.evaluate_actions(
                    obs=batch.obs,
                    actions=batch.actions,
                    action_mask=batch.action_masks,
                )

                self._check_finite_tensor("new_log_prob", new_log_prob)
                self._check_finite_tensor("entropy", entropy)
                self._check_finite_tensor("new_value", new_value)

                log_ratio = new_log_prob - batch.old_log_probs
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = (
                        torch.abs(ratio - 1.0) > float(self.ppo_cfg.clip_coef)
                    ).float().mean()

                #이미 KL이 target을 넘었다면
                # 현재 minibatch의 추가 update를 수행하지 않는다.
                if (
                    self.ppo_cfg.target_kl is not None
                    and float(approx_kl.item())
                    > float(self.ppo_cfg.target_kl)
                ):
                    approx_kl_values.append(
                        float(approx_kl.item())
                    )
                    clip_frac_values.append(
                        float(clip_frac.item())
                    )
                    early_stopped = True
                    break

                adv = batch.advantages

                policy_loss_1 = -adv * ratio
                policy_loss_2 = -adv * torch.clamp(
                    ratio,
                    1.0 - float(self.ppo_cfg.clip_coef),
                    1.0 + float(self.ppo_cfg.clip_coef),
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                value_loss = self._compute_value_loss(
                    new_value=new_value,
                    old_value=batch.old_values,
                    returns=batch.returns,
                )

                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + float(self.ppo_cfg.value_coef) * value_loss
                    - float(self.ppo_cfg.entropy_coef) * entropy_mean
                )

                self._check_finite_tensor("ppo_loss", loss)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.ppo_cfg.max_grad_norm),
                )
                self.optimizer.step()

                completed_minibatches += 1

                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropy_values.append(float(entropy_mean.detach().cpu()))
                approx_kl_values.append(float(approx_kl.detach().cpu()))
                clip_frac_values.append(float(clip_frac.detach().cpu()))
            
            if early_stopped:
                break

        obs, actions, old_log_probs, returns, advantages, old_values = self.buffer.get_tensors()
        action_masks = self.buffer.get_action_masks_tensor()

        with torch.no_grad():
            _, _, value_after = self.model.evaluate_actions(obs, actions, action_masks)

            explained_v = explained_var(
                y_pred=value_after.detach().cpu().numpy(),
                y_true=returns.detach().cpu().numpy(),
            )

        buffer_summary = self.buffer.summary()

        logs = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "approx_kl": float(np.mean(approx_kl_values)) if approx_kl_values else 0.0,
            "clipfrac": float(np.mean(clip_frac_values)) if clip_frac_values else 0.0,
            "explained_variance": float(explained_v),
            "buffer_reward_mean": float(buffer_summary["reward_mean"]),
            "buffer_reward_std": float(buffer_summary["reward_std"]),
            "active_action_dims_mean": float(
                action_masks
                .sum(dim=-1)
                .mean()
                .detach()
                .cpu()
                .item()
            ),
            "active_action_ratio_mean": float(
                action_masks
                .mean()
                .detach()
                .cpu()
                .item()
            ),
            "early_stopped": float(early_stopped),
            "completed_minibatches": float(
                completed_minibatches
            ),
        }

        self.buffer.reset()

        return logs

    def save(
        self,
        path: str | Path,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        model/optimizer 저장.
        """
        merged_extra: Dict[str, Any] = {
            "fast_ppo_config": asdict(self.ppo_cfg),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

        if self.obs_normalizer is not None:
            merged_extra["obs_normalizer"] = self.obs_normalizer.state_dict()

        if extra is not None:
            merged_extra.update(extra)

        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            extra=merged_extra,
        )

    def load(
        self,
        path: str | Path,
        strict: bool = True,
        load_optimizer: bool = False,
    ) -> Dict[str, Any]:
        checkpoint = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer if load_optimizer else None,
            device=self.device,
            strict=bool(strict),
        )

        extra = checkpoint.get("extra", {})
        obs_norm_state = extra.get("obs_normalizer", None)

        if obs_norm_state is not None and self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(obs_norm_state)

        return checkpoint