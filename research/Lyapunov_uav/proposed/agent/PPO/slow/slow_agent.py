from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import EnvConfig

try:
    from agent.PPO.common import (
        ObsNormalizer,
        RolloutBuffer,
        explained_var,
        flatten_slow_obs,
        get_device,
        load_checkpoint,
        save_checkpoint,
        to_tensor,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ..common import (
        ObsNormalizer,
        RolloutBuffer,
        explained_var,
        flatten_slow_obs,
        get_device,
        load_checkpoint,
        save_checkpoint,
        to_tensor,
    )

from .slow_action import SlowActionCodec
from .slow_network import SlowActorCritic


@dataclass
class SlowPPOConfig:
    """
    Slow-timescale PPO hyperparameters.
    """
    # ------------------------------------------------------------
    # 1) rollout / PPO update
    # ------------------------------------------------------------
    rollout_rounds: int = 128
    update_epochs: int = 4
    batch_size: int = 32

    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ------------------------------------------------------------
    # 2) optimization
    # ------------------------------------------------------------
    lr: float = 3e-5
    max_grad_norm: float = 0.5

    clip_coef: float = 0.15
    value_coef: float = 0.5

    # Bernoulli entropy is summed over all slow action bits.
    # action_dim can be large, so keep entropy_coef conservative.
    entropy_coef: float = 1e-3
    target_kl: Optional[float] = 0.02

    # ------------------------------------------------------------
    # 3) normalization
    # ------------------------------------------------------------
    normalize_obs: bool = True
    normalize_adv: bool = True
    reward_scale: float = 1e-6

    # ------------------------------------------------------------
    # 4) network
    # ------------------------------------------------------------
    hidden_dims: Tuple[int, ...] = (256, 256)

    rsu_init_logit: float = 0.0             
    hiring_init_logit: float = 0.0
    uav_init_logit: float = 0.0
    min_logit: float = -20.0
    max_logit: float = 20.0

    # ------------------------------------------------------------
    # 5) critic stability
    # ------------------------------------------------------------
    use_value_huber_loss: bool = True
    use_value_clip: bool = True
    value_clip_coef: float = 1.0

    # ------------------------------------------------------------
    # 6) debug / device
    # ------------------------------------------------------------
    fail_on_nan: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if int(self.rollout_rounds) <= 0:
            raise ValueError("rollout_rounds must be positive.")
        if int(self.update_epochs) <= 0:
            raise ValueError("update_epochs must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(self.rollout_rounds) < int(self.batch_size):
            raise ValueError("rollout_rounds must be >= batch_size.")
        if int(self.rollout_rounds) % int(self.batch_size) != 0:
            raise ValueError("rollout_rounds must be divisible by batch_size.")
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if not 0.0 < float(self.gae_lambda) <= 1.0:
            raise ValueError("gae_lambda must be in (0, 1].")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive.")
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive.")
        if not 0.0 < float(self.clip_coef) < 1.0:
            raise ValueError("clip_coef must be in (0, 1).")
        if float(self.value_coef) < 0.0 or float(self.entropy_coef) < 0.0:
            raise ValueError("value_coef and entropy_coef must be non-negative.")
        if self.target_kl is not None and float(self.target_kl) <= 0.0:
            raise ValueError("target_kl must be positive or None.")
        if float(self.reward_scale) <= 0.0:
            raise ValueError("reward_scale must be positive.")
        if not isinstance(self.hidden_dims, tuple):
            self.hidden_dims = tuple(int(x) for x in self.hidden_dims)
        if not self.hidden_dims or any(int(x) <= 0 for x in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        if float(self.min_logit) >= float(self.max_logit):
            raise ValueError("min_logit must be smaller than max_logit.")
        if float(self.value_clip_coef) <= 0.0:
            raise ValueError("value_clip_coef must be positive.")
        

class SlowPPOAgent:
    """
    Slow-timescale PPO Agent.
    """
    def __init__(
        self,
        env_cfg: EnvConfig,
        obs_dim: int,
        ppo_cfg: Optional[SlowPPOConfig] = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg or SlowPPOConfig()

        self.device = get_device(self.ppo_cfg.device)
        self.codec = SlowActionCodec(env_cfg)
        self.obs_dim = obs_dim
        self.action_dim = self.codec.action_dim

        self.model = SlowActorCritic(
            obs_dim=self.obs_dim,
            action_spec=self.codec.spec,
            hidden_dims=self.ppo_cfg.hidden_dims,
            rsu_init_logit=float(self.ppo_cfg.rsu_init_logit),
            hiring_init_logit=float(self.ppo_cfg.hiring_init_logit),
            uav_init_logit=float(self.ppo_cfg.uav_init_logit),
            min_logit=float(self.ppo_cfg.min_logit),
            max_logit=float(self.ppo_cfg.max_logit),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.ppo_cfg.lr,
            eps=1e-5,
        )
        self.buffer = RolloutBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=self.ppo_cfg.rollout_rounds,
            device=self.device,
            gamma=self.ppo_cfg.gamma,
            gae_lambda=self.ppo_cfg.gae_lambda,
        )

        self.obs_normalizer: Optional[ObsNormalizer]
        if bool(self.ppo_cfg.normalize_obs):
            self.obs_normalizer = ObsNormalizer(obs_dim=self.obs_dim)
        else:
            self.obs_normalizer = None
    
    def _check_finite_array(self, name: str, value: np.ndarray) -> None:
        if self.ppo_cfg.fail_on_nan and not np.all(np.isfinite(value)):
            raise RuntimeError(f"{name} contains NaN or Inf.")

    def _check_finite_tensor(self, name: str, value: torch.Tensor) -> None:
        if self.ppo_cfg.fail_on_nan and not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains NaN or Inf.")

    def _check_binary_array(self, name: str, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        is_zero = np.isclose(array, 0.0, atol=1e-6)
        is_one = np.isclose(array, 1.0, atol=1e-6)
        if not np.all(is_zero | is_one):
            bad = np.flatnonzero(~(is_zero | is_one))[:10]
            raise RuntimeError(
                f"{name} must be binary; bad_idx={bad.tolist()}, "
                f"bad_value={array[bad].tolist()}"
            )

    def obs_to_vec(
        self, obs: Mapping[str, Any], update_norm: bool = True
    ) -> np.ndarray:
        # Keep the formulation state exactly s_H=[Z, B, user_region].
        vector = flatten_slow_obs(dict(obs)).astype(np.float32)
        if vector.shape != (self.obs_dim,):
            raise ValueError(
                f"slow obs shape mismatch: expected={(self.obs_dim,)}, "
                f"got={vector.shape}"
            )
        self._check_finite_array("slow_obs_before_norm", vector)
        if self.obs_normalizer is not None:
            vector = self.obs_normalizer.normalize(
                vector, update=bool(update_norm)
            )
        self._check_finite_array("slow_obs_after_norm", vector)
        return vector.astype(np.float32, copy=False)

    @torch.no_grad()
    def select_action(
        self,
        obs: Mapping[str, Any],
        *,
        context: Optional[Any] = None,
        deterministic: bool = False,
        update_norm: bool = True,
    ) -> Dict[str, Any]:
        obs_vec = self.obs_to_vec(obs, update_norm=update_norm)
        static_mask = self.codec.build_static_action_mask(
            obs, context=context
        ).astype(np.float32)
        obs_tensor = to_tensor(obs_vec, device=self.device).unsqueeze(0)
        static_mask_tensor = to_tensor(
            static_mask, device=self.device
        ).unsqueeze(0)

        action_t, log_prob_t, value_t, action_mask_t = self.model.act(
            obs_tensor,
            static_action_mask=static_mask_tensor,
            deterministic=bool(deterministic),
        )
        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        action_mask = (
            action_mask_t.squeeze(0).cpu().numpy().astype(np.float32)
        )
        log_prob = float(log_prob_t.squeeze(0).cpu().item())
        value = float(value_t.squeeze(0).cpu().item())

        self._check_finite_array("slow_action", action)
        self._check_binary_array("slow_action", action)
        self._check_binary_array("slow_action_mask", action_mask)
        if np.any(action * (1.0 - action_mask) != 0.0):
            raise RuntimeError("slow policy emitted an action outside its mask.")

        codec_mask = self.codec.build_effective_action_mask(
            action, obs, context=context
        )
        if not np.array_equal(action_mask, codec_mask):
            raise RuntimeError("network and codec effective masks disagree.")

        env_action, action_info = self.codec.decode_with_info(
            action, obs, context=context
        )
        if int(action_info["projection_count"]) != 0:
            raise RuntimeError(
                "masked slow policy required post-hoc projection; "
                "PPO log-probability would be invalid."
            )
        return {
            "obs_vec": obs_vec,
            "binary_action": action,
            "raw_action": action,
            "env_action": env_action,
            "log_prob": log_prob,
            "value": value,
            "static_action_mask": static_mask,
            "action_mask": action_mask,
            "action_info": action_info,
        }

    def store_transition(
        self,
        obs_vec: np.ndarray,
        binary_action: np.ndarray,
        action_mask: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        for name, scalar in (
            ("reward", reward), ("value", value), ("log_prob", log_prob)
        ):
            if self.ppo_cfg.fail_on_nan and not np.isfinite(float(scalar)):
                raise RuntimeError(f"slow {name} is NaN or Inf: {scalar}")

        action = np.asarray(binary_action, dtype=np.float32).reshape(-1)
        mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        if action.shape != (self.action_dim,) or mask.shape != (self.action_dim,):
            raise ValueError("stored slow action/action_mask shape mismatch.")
        self._check_binary_array("stored_slow_action", action)
        self._check_binary_array("stored_slow_action_mask", mask)
        if np.any(action * (1.0 - mask) != 0.0):
            raise ValueError("stored slow action has nonzero inactive bits.")

        self.buffer.add(
            obs=np.asarray(obs_vec, dtype=np.float32).reshape(-1),
            action=action,
            action_mask=mask,
            reward=float(reward) * float(self.ppo_cfg.reward_scale),
            done=bool(done),
            value=float(value),
            log_prob=float(log_prob),
        )

    @torch.no_grad()
    def estimate_value(
        self, obs: Mapping[str, Any], update_norm: bool = False
    ) -> float:
        obs_vec = self.obs_to_vec(obs, update_norm=update_norm)
        value = self.model.value(
            to_tensor(obs_vec, device=self.device).unsqueeze(0)
        )
        self._check_finite_tensor("slow_estimated_value", value)
        return float(value.squeeze(0).cpu().item())

    def finish_rollout(
        self, last_obs: Mapping[str, Any], last_done: bool
    ) -> None:
        last_value = (
            0.0
            if bool(last_done)
            else self.estimate_value(last_obs, update_norm=False)
        )
        self.buffer.compute_returns_and_advs(
            last_value=last_value,
            last_done=bool(last_done),
            normalize_adv=bool(self.ppo_cfg.normalize_adv),
        )

    def _compute_value_loss(
        self,
        new_value: torch.Tensor,
        old_value: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        if self.ppo_cfg.use_value_clip:
            clipped = old_value + torch.clamp(
                new_value - old_value,
                -float(self.ppo_cfg.value_clip_coef),
                float(self.ppo_cfg.value_clip_coef),
            )
            if self.ppo_cfg.use_value_huber_loss:
                raw_loss = F.smooth_l1_loss(
                    new_value, returns, reduction="none"
                )
                clipped_loss = F.smooth_l1_loss(
                    clipped, returns, reduction="none"
                )
            else:
                raw_loss = (new_value - returns).pow(2)
                clipped_loss = (clipped - returns).pow(2)
            return torch.maximum(raw_loss, clipped_loss).mean()
        if self.ppo_cfg.use_value_huber_loss:
            return F.smooth_l1_loss(new_value, returns)
        return F.mse_loss(new_value, returns)

    def update(self) -> Dict[str, float]:
        if len(self.buffer) == 0:
            raise RuntimeError("cannot update from an empty slow rollout buffer.")

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        rsu_entropies: list[float] = []
        hiring_entropies: list[float] = []
        uav_entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []
        early_stopped = False
        completed_minibatches = 0

        for _ in range(int(self.ppo_cfg.update_epochs)):
            for batch in self.buffer.iter_minibatches(
                batch_size=int(self.ppo_cfg.batch_size),
                shuffle=True,
                include_action_masks=True,
            ):
                if batch.action_masks is None:
                    raise RuntimeError("slow PPO minibatch is missing action masks.")
                (
                    new_log_prob,
                    entropy,
                    new_value,
                    rsu_entropy,
                    hiring_entropy,
                    uav_entropy,
                ) = self.model.evaluate_actions(
                    batch.obs, batch.actions, batch.action_masks
                )
                for name, tensor in (
                    ("new_log_prob", new_log_prob),
                    ("entropy", entropy),
                    ("new_value", new_value),
                ):
                    self._check_finite_tensor(name, tensor)

                log_ratio = new_log_prob - batch.old_log_probs
                ratio = torch.exp(log_ratio)
                self._check_finite_tensor("slow_ppo_ratio", ratio)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = (
                        torch.abs(ratio - 1.0)
                        > float(self.ppo_cfg.clip_coef)
                    ).float().mean()

                if (
                    self.ppo_cfg.target_kl is not None
                    and float(approx_kl.item()) > float(self.ppo_cfg.target_kl)
                ):
                    approx_kls.append(float(approx_kl.item()))
                    clip_fracs.append(float(clip_frac.item()))
                    early_stopped = True
                    break

                unclipped = -batch.advantages * ratio
                clipped = -batch.advantages * torch.clamp(
                    ratio,
                    1.0 - float(self.ppo_cfg.clip_coef),
                    1.0 + float(self.ppo_cfg.clip_coef),
                )
                policy_loss = torch.maximum(unclipped, clipped).mean()
                value_loss = self._compute_value_loss(
                    new_value, batch.old_values, batch.returns
                )
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + float(self.ppo_cfg.value_coef) * value_loss
                    - float(self.ppo_cfg.entropy_coef) * entropy_mean
                )
                self._check_finite_tensor("slow_ppo_loss", loss)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.ppo_cfg.max_grad_norm),
                )
                self.optimizer.step()

                completed_minibatches += 1
                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy_mean.detach().cpu().item()))
                rsu_entropies.append(float(rsu_entropy.mean().detach().cpu().item()))
                hiring_entropies.append(
                    float(hiring_entropy.mean().detach().cpu().item())
                )
                uav_entropies.append(float(uav_entropy.mean().detach().cpu().item()))
                approx_kls.append(float(approx_kl.detach().cpu().item()))
                clip_fracs.append(float(clip_frac.detach().cpu().item()))
            if early_stopped:
                break

        obs, actions, _, returns, _, _ = self.buffer.get_tensors()
        action_masks = self.buffer.get_action_masks_tensor()
        with torch.no_grad():
            _, _, value_after, _, _, _ = self.model.evaluate_actions(
                obs, actions, action_masks
            )
            explained_v = explained_var(
                y_pred=value_after.cpu().numpy(),
                y_true=returns.cpu().numpy(),
            )

        summary = self.buffer.summary()
        entropy_mean = float(np.mean(entropies)) if entropies else 0.0
        logs = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": entropy_mean,
            "rsu_entropy": float(np.mean(rsu_entropies)) if rsu_entropies else 0.0,
            "hiring_entropy": float(np.mean(hiring_entropies)) if hiring_entropies else 0.0,
            "uav_entropy": float(np.mean(uav_entropies)) if uav_entropies else 0.0,
            "entropy_bonus": float(self.ppo_cfg.entropy_coef) * entropy_mean,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "explained_variance": float(explained_v),
            "buffer_scaled_reward_mean": float(summary["reward_mean"]),
            "buffer_scaled_reward_std": float(summary["reward_std"]),
            "buffer_done_ratio": float(summary["done_ratio"]),
            "active_action_dims_mean": float(
                action_masks.sum(dim=-1).mean().cpu().item()
            ),
            "active_action_ratio_mean": float(action_masks.mean().cpu().item()),
            "early_stopped": float(early_stopped),
            "completed_minibatches": float(completed_minibatches),
        }
        self.buffer.reset()
        return logs

    @torch.no_grad()
    def action_summary(
        self,
        obs: Mapping[str, Any],
        *,
        context: Optional[Any] = None,
        update_norm: bool = False,
    ) -> Dict[str, float]:
        obs_vec = self.obs_to_vec(obs, update_norm=update_norm)
        static_mask = self.codec.build_static_action_mask(
            obs, context=context
        )
        return self.model.action_summary(
            to_tensor(obs_vec, device=self.device).unsqueeze(0),
            to_tensor(static_mask, device=self.device).unsqueeze(0),
        )

    def save(
        self, path: str | Path, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        merged_extra: Dict[str, Any] = {
            "policy_type": "autoregressive_conditional_bernoulli_slow_v1",
            "slow_ppo_config": asdict(self.ppo_cfg),
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
        if extra.get("action_dim", self.action_dim) != self.action_dim:
            raise ValueError("slow checkpoint action_dim does not match this environment.")
        normalizer_state = extra.get("obs_normalizer")
        if normalizer_state is not None and self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(normalizer_state)
        return checkpoint