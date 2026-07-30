from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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
    """Fast-timescale PPO hyperparameters."""

    rollout_steps: int = 3600
    update_epochs: int = 4
    batch_size: int = 450

    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-5
    max_grad_norm: float = 0.5

    clip_coef: float = 0.15
    value_coef: float = 0.5

    categorical_entropy_coef: float = 5e-4
    power_entropy_coef: float = 1e-4

    normalize_obs: bool = True
    normalize_adv: bool = True

    hidden_dims: Tuple[int, ...] = (256, 256)
    init_log_std: float = -1.0

    use_value_huber_loss: bool = True
    use_value_clip: bool = True
    value_clip_coef: float = 0.5

    fail_on_nan: bool = True
    target_kl: Optional[float] = 0.02

    device: str = "auto"


class FastPPOAgent:
    """
    Fast-timescale mixed-action PPO agent.

    Joint DPP mode에서 중요한 불변식:
        1) Slow candidate forecast 중 model parameter를 갱신하지 않는다.
        2) Forecast observation은 normalizer 통계를 갱신하지 않는다.
        3) 실제 round에서도 normalizer 통계를 고정하고, round 종료 뒤
           실제 raw observation batch로 한 번 갱신한다.
        4) Forecast는 actor-only batched inference를 사용한다.
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

        self.obs_dim = int(obs_dim)
        self.action_dim = int(self.codec.action_dim)

        self.model = FastActorCritic(
            obs_dim=self.obs_dim,
            action_spec=self.codec.spec,
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

        self.obs_normalizer: Optional[ObsNormalizer]
        if self.ppo_cfg.normalize_obs:
            self.obs_normalizer = ObsNormalizer(obs_dim=self.obs_dim)
        else:
            self.obs_normalizer = None

    # ------------------------------------------------------------------
    # validation / observation handling
    # ------------------------------------------------------------------
    def _check_finite_array(self, name: str, arr: np.ndarray) -> None:
        if bool(self.ppo_cfg.fail_on_nan) and not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} contains NaN or Inf.")

    def _check_finite_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if bool(self.ppo_cfg.fail_on_nan) and not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN or Inf.")

    def raw_obs_to_vec(self, obs: Dict[str, Any]) -> np.ndarray:
        """Normalizer 적용 전 Fast observation vector."""
        obs_vec = flatten_fast_obs(obs).astype(np.float32)
        if obs_vec.shape != (self.obs_dim,):
            raise ValueError(
                f"obs dim mismatch: expected {(self.obs_dim,)}, "
                f"got {obs_vec.shape}"
            )
        self._check_finite_array("raw_obs_vec", obs_vec)
        return obs_vec

    def _normalize_raw_matrix(
        self,
        raw_obs_matrix: np.ndarray,
        *,
        update_norm: bool,
    ) -> np.ndarray:
        raw = np.asarray(raw_obs_matrix, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.ndim != 2 or raw.shape[1] != self.obs_dim:
            raise ValueError(
                "raw observation matrix shape mismatch: "
                f"expected=(*,{self.obs_dim}), got={raw.shape}"
            )
        self._check_finite_array("raw_obs_matrix", raw)

        if self.obs_normalizer is None:
            return raw.astype(np.float32, copy=False)

        if update_norm:
            # RunningMeanStd는 batch update를 지원한다. 여러 sample을 하나씩
            # 갱신하지 않고 동일한 통계 상태에서 한 번에 반영한다.
            self.obs_normalizer.rms.update(raw)

        mean = np.asarray(self.obs_normalizer.rms.mean, dtype=np.float64)
        std = np.asarray(self.obs_normalizer.rms.std, dtype=np.float64)
        normalized = (raw.astype(np.float64) - mean[None, :]) / (
            std[None, :] + float(self.obs_normalizer.eps)
        )
        normalized = np.clip(
            normalized,
            -float(self.obs_normalizer.clip),
            float(self.obs_normalizer.clip),
        ).astype(np.float32)
        self._check_finite_array("normalized_obs_matrix", normalized)
        return normalized

    def obs_to_vec(
        self,
        obs: Dict[str, Any],
        update_norm: bool = True,
    ) -> np.ndarray:
        raw = self.raw_obs_to_vec(obs)
        return self._normalize_raw_matrix(
            raw,
            update_norm=bool(update_norm),
        )[0]

    def update_obs_normalizer(
        self,
        raw_observations: Sequence[np.ndarray] | np.ndarray,
    ) -> None:
        """완료된 실제 round의 raw observations만 batch 반영."""
        if self.obs_normalizer is None:
            return
        raw = np.asarray(raw_observations, dtype=np.float32)
        if raw.size == 0:
            return
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.shape[1] != self.obs_dim:
            raise ValueError(
                f"normalizer update obs_dim mismatch: {raw.shape}"
            )
        self._check_finite_array("normalizer_update_batch", raw)
        self.obs_normalizer.rms.update(raw)

    # ------------------------------------------------------------------
    # action selection
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def select_env_actions_batch(
        self,
        observations: Sequence[Dict[str, Any]],
        *,
        deterministic: bool = True,
        update_norm: bool = False,
    ) -> Dict[str, Any]:
        """
        Slow candidate forecast용 actor-only GPU batch inference.

        critic/log_prob/entropy를 계산하지 않으며 GPU->CPU transfer도
        candidate batch당 한 번만 수행한다.
        """
        if len(observations) == 0:
            return {
                "raw_obs_matrix": np.zeros((0, self.obs_dim), dtype=np.float32),
                "obs_matrix": np.zeros((0, self.obs_dim), dtype=np.float32),
                "raw_action_matrix": np.zeros(
                    (0, self.action_dim), dtype=np.float32
                ),
                "action_mask_matrix": np.zeros(
                    (0, self.action_dim), dtype=np.float32
                ),
                "env_actions": [],
            }

        raw_matrix = np.stack(
            [self.raw_obs_to_vec(obs) for obs in observations],
            axis=0,
        ).astype(np.float32)
        obs_matrix = self._normalize_raw_matrix(
            raw_matrix,
            update_norm=bool(update_norm),
        )
        base_masks = np.stack(
            [self.codec.build_base_action_mask(obs) for obs in observations],
            axis=0,
        ).astype(np.float32)

        obs_tensor = to_tensor(obs_matrix, device=self.device)
        mask_tensor = to_tensor(base_masks, device=self.device)

        action_tensor, eff_mask_tensor = self.model.policy_action(
            obs=obs_tensor,
            deterministic=bool(deterministic),
            action_mask=mask_tensor,
        )

        raw_actions = action_tensor.detach().cpu().numpy().astype(np.float32)
        eff_masks = eff_mask_tensor.detach().cpu().numpy().astype(np.float32)
        self._check_finite_array("batch_policy_actions", raw_actions)
        self._check_finite_array("batch_effective_masks", eff_masks)

        env_actions = [
            self.codec.decode(raw_actions[i], observations[i])
            for i in range(len(observations))
        ]

        return {
            "raw_obs_matrix": raw_matrix,
            "obs_matrix": obs_matrix,
            "raw_action_matrix": raw_actions,
            "action_mask_matrix": eff_masks,
            "env_actions": env_actions,
        }

    @torch.no_grad()
    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        update_norm: bool = True,
    ) -> Dict[str, Any]:
        """실제 Fast rollout에서 PPO transition에 필요한 전체 값 반환."""
        raw_obs_vec = self.raw_obs_to_vec(obs)
        obs_vec = self._normalize_raw_matrix(
            raw_obs_vec,
            update_norm=bool(update_norm),
        )[0]
        base_mask = self.codec.build_base_action_mask(obs)

        obs_tensor = to_tensor(obs_vec, device=self.device).unsqueeze(0)
        mask_tensor = to_tensor(base_mask, device=self.device).unsqueeze(0)

        (
            action_tensor,
            log_prob_tensor,
            value_tensor,
            eff_mask_tensor,
        ) = self.model.act(
            obs_tensor,
            deterministic=bool(deterministic),
            action_mask=mask_tensor,
        )

        policy_action = (
            action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        )
        eff_mask = (
            eff_mask_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        )
        self._check_finite_array("policy_action", policy_action)
        self._check_finite_array("eff_action_mask", eff_mask)

        env_action = self.codec.decode(policy_action, obs)

        power_raw = policy_action[self.codec.spec.uav_power_slice]
        power_mask = eff_mask[self.codec.spec.uav_power_slice] > 0.0
        if np.any(power_mask):
            action_saturation_ratio = float(
                np.mean(np.abs(np.tanh(power_raw[power_mask])) >= 0.98)
            )
        else:
            action_saturation_ratio = 0.0

        action_stats = self.codec.action_statistics(policy_action, eff_mask)

        return {
            "raw_obs_vec": raw_obs_vec,
            "obs_vec": obs_vec,
            "raw_action": policy_action,
            "action_mask": eff_mask,
            "env_action": env_action,
            "log_prob": float(log_prob_tensor.squeeze(0).cpu().item()),
            "value": float(value_tensor.squeeze(0).cpu().item()),
            "active_action_dims": int(np.sum(eff_mask)),
            "active_action_ratio": float(np.mean(eff_mask)),
            "action_saturation_ratio": action_saturation_ratio,
            **action_stats,
        }

    # ------------------------------------------------------------------
    # PPO buffer / update
    # ------------------------------------------------------------------
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
        if self.ppo_cfg.fail_on_nan:
            for name, value in (
                ("reward", reward),
                ("value", value),
                ("log_prob", log_prob),
            ):
                if not np.isfinite(float(value)):
                    raise RuntimeError(f"{name} is NaN or Inf: {value}")

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
        obs_vec = self.obs_to_vec(obs, update_norm=update_norm)
        obs_tensor = to_tensor(obs_vec, device=self.device).unsqueeze(0)
        value = self.model.value(obs_tensor)
        return float(value.squeeze(0).detach().cpu().item())

    def finish_rollout(
        self,
        last_obs: Dict[str, Any],
        last_done: bool,
    ) -> None:
        last_value = 0.0 if last_done else self.estimate_value(
            last_obs,
            update_norm=False,
        )
        self.buffer.compute_returns_and_advs(
            last_value=last_value,
            last_done=bool(last_done),
            normalize_adv=self.ppo_cfg.normalize_adv,
        )

    def _compute_value_loss(
        self,
        new_value: torch.Tensor,
        old_value: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        if self.ppo_cfg.use_value_clip:
            value_clipped = old_value + torch.clamp(
                new_value - old_value,
                -float(self.ppo_cfg.value_clip_coef),
                float(self.ppo_cfg.value_clip_coef),
            )
            if self.ppo_cfg.use_value_huber_loss:
                loss_unclipped = F.smooth_l1_loss(
                    new_value, returns, reduction="none"
                )
                loss_clipped = F.smooth_l1_loss(
                    value_clipped, returns, reduction="none"
                )
            else:
                loss_unclipped = (new_value - returns).pow(2)
                loss_clipped = (value_clipped - returns).pow(2)
            return torch.max(loss_unclipped, loss_clipped).mean()

        if self.ppo_cfg.use_value_huber_loss:
            return F.smooth_l1_loss(new_value, returns)
        return F.mse_loss(new_value, returns)

    def update(self) -> Dict[str, float]:
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot update PPO with an empty buffer.")
        if not self.buffer.advantages_ready:
            raise RuntimeError("finish_rollout() must be called before update().")

        policy_losses: list[float] = []
        value_losses: list[float] = []
        categorical_entropies: list[float] = []
        power_entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        early_stopped = False
        completed_minibatches = 0

        self.model.train()
        for _ in range(int(self.ppo_cfg.update_epochs)):
            for batch in self.buffer.iter_minibatches(
                batch_size=int(self.ppo_cfg.batch_size),
                shuffle=True,
                include_action_masks=True,
            ):
                if batch.action_masks is None:
                    raise RuntimeError("PPO action masks are required.")

                (
                    new_log_prob,
                    categorical_entropy,
                    power_entropy,
                    new_value,
                ) = self.model.evaluate_actions(
                    obs=batch.obs,
                    actions=batch.actions,
                    action_mask=batch.action_masks,
                )

                for name, tensor in (
                    ("new_log_prob", new_log_prob),
                    ("categorical_entropy", categorical_entropy),
                    ("power_entropy", power_entropy),
                    ("new_value", new_value),
                ):
                    self._check_finite_tensor(name, tensor)

                log_ratio = new_log_prob - batch.old_log_probs
                ratio = torch.exp(log_ratio)
                self._check_finite_tensor("ppo_ratio", ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = (
                        torch.abs(ratio - 1.0)
                        > float(self.ppo_cfg.clip_coef)
                    ).float().mean()

                if (
                    self.ppo_cfg.target_kl is not None
                    and float(approx_kl.item())
                    > float(self.ppo_cfg.target_kl)
                ):
                    approx_kls.append(float(approx_kl.item()))
                    clip_fracs.append(float(clip_frac.item()))
                    early_stopped = True
                    break

                adv = batch.advantages
                policy_loss_unclipped = -adv * ratio
                policy_loss_clipped = -adv * torch.clamp(
                    ratio,
                    1.0 - float(self.ppo_cfg.clip_coef),
                    1.0 + float(self.ppo_cfg.clip_coef),
                )
                policy_loss = torch.maximum(
                    policy_loss_unclipped,
                    policy_loss_clipped,
                ).mean()

                value_loss = self._compute_value_loss(
                    new_value=new_value,
                    old_value=batch.old_values,
                    returns=batch.returns,
                )

                categorical_entropy_mean = categorical_entropy.mean()
                power_entropy_mean = power_entropy.mean()

                loss = (
                    policy_loss
                    + float(self.ppo_cfg.value_coef) * value_loss
                    - float(self.ppo_cfg.categorical_entropy_coef)
                    * categorical_entropy_mean
                    - float(self.ppo_cfg.power_entropy_coef)
                    * power_entropy_mean
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
                categorical_entropies.append(
                    float(categorical_entropy_mean.detach().cpu())
                )
                power_entropies.append(
                    float(power_entropy_mean.detach().cpu())
                )
                approx_kls.append(float(approx_kl.detach().cpu()))
                clip_fracs.append(float(clip_frac.detach().cpu()))

            if early_stopped:
                break

        obs, actions, _, returns, _, _ = self.buffer.get_tensors()
        action_masks = self.buffer.get_action_masks_tensor()
        with torch.no_grad():
            _, _, _, value_after = self.model.evaluate_actions(
                obs, actions, action_masks
            )
            self._check_finite_tensor("value_after", value_after)
            explained_v = explained_var(
                y_pred=value_after.detach().cpu().numpy(),
                y_true=returns.detach().cpu().numpy(),
            )

        buffer_summary = self.buffer.summary()
        cat_entropy_mean = (
            float(np.mean(categorical_entropies))
            if categorical_entropies
            else 0.0
        )
        power_entropy_mean = (
            float(np.mean(power_entropies)) if power_entropies else 0.0
        )
        entropy_bonus = (
            float(self.ppo_cfg.categorical_entropy_coef) * cat_entropy_mean
            + float(self.ppo_cfg.power_entropy_coef) * power_entropy_mean
        )

        logs = {
            "policy_loss": (
                float(np.mean(policy_losses)) if policy_losses else 0.0
            ),
            "value_loss": (
                float(np.mean(value_losses)) if value_losses else 0.0
            ),
            "categorical_entropy": cat_entropy_mean,
            "power_entropy": power_entropy_mean,
            "entropy_bonus": entropy_bonus,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": (
                float(np.mean(clip_fracs)) if clip_fracs else 0.0
            ),
            "explained_variance": float(explained_v),
            "buffer_reward_mean": float(buffer_summary["reward_mean"]),
            "buffer_reward_std": float(buffer_summary["reward_std"]),
            "active_action_dims_mean": float(
                action_masks.sum(dim=-1).mean().detach().cpu().item()
            ),
            "active_action_ratio_mean": float(
                action_masks.mean().detach().cpu().item()
            ),
            "early_stopped": float(early_stopped),
            "completed_minibatches": float(completed_minibatches),
        }

        self.buffer.reset()
        return logs

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def save(
        self,
        path: str | Path,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_extra: Dict[str, Any] = {
            "policy_type": "conditional_mixed_categorical_gaussian_v1",
            "fast_ppo_config": asdict(self.ppo_cfg),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
        if self.obs_normalizer is not None:
            merged_extra["obs_normalizer"] = (
                self.obs_normalizer.state_dict()
            )
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
        obs_norm_state = extra.get("obs_normalizer")
        if obs_norm_state is not None and self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(obs_norm_state)
        return checkpoint

    def load_legacy_transfer(self, path: str | Path) -> Dict[str, Any]:
        """기존 all-Gaussian checkpoint에서 critic/normalizer만 이전."""
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        source_state = checkpoint["model_state_dict"]
        target_state = self.model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if (
                key.startswith("critic_network.")
                and key in target_state
                and target_state[key].shape == value.shape
            )
        }
        if not compatible:
            raise RuntimeError("No compatible critic parameters found.")

        self.model.load_state_dict(compatible, strict=False)
        extra = checkpoint.get("extra", {})
        obs_norm_state = extra.get("obs_normalizer")
        if obs_norm_state is not None and self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(obs_norm_state)

        return {
            "transferred_keys": sorted(compatible.keys()),
            "num_transferred_tensors": len(compatible),
            "source_extra": extra,
        }
