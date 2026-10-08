"""Continuous-latent SAC with state-only masks and common action projection."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.size, self.ptr, self.n = size, 0, 0
        self.data = {k: np.zeros((size, dim), dtype=np.float32) for k, dim in
            (("obs", obs_dim), ("act", act_dim), ("rew", 1), ("next_obs", obs_dim),
             ("done", 1), ("mask", act_dim), ("next_mask", act_dim))}

    def add(self, obs, action, reward, next_obs, done, mask, next_mask):
        for key, value in zip(self.data, (obs, action, reward, next_obs, done, mask, next_mask)):
            self.data[key][self.ptr] = value
        self.ptr, self.n = (self.ptr + 1) % self.size, min(self.n + 1, self.size)

    def sample(self, count, rng):
        indices = rng.integers(0, self.n, size=count)
        return {k: torch.from_numpy(v[indices]) for k, v in self.data.items()}

    def state(self):
        return dict(size=self.size, ptr=self.ptr, n=self.n,
                    data={k: v[:self.n].copy() for k, v in self.data.items()})

    def restore(self, saved):
        if saved["size"] != self.size or not 0 <= saved["n"] <= self.size:
            raise ValueError("Replay geometry differs")
        self.ptr, self.n = saved["ptr"], saved["n"]
        if not 0 <= self.ptr < self.size or (self.n < self.size and self.ptr != self.n):
            raise ValueError("Invalid replay cursor")
        if set(saved["data"]) != set(self.data):
            raise ValueError("Replay schema differs")
        for key, value in saved["data"].items():
            if value.shape != self.data[key][:self.n].shape or not np.isfinite(value).all():
                raise ValueError("Invalid replay array: " + key)
            self.data[key][:self.n] = value


def mlp(i, o, hidden):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(i, width), nn.ReLU()))
        i = width
    return nn.Sequential(*layers, nn.Linear(i, o))


class SACAgent:
    def __init__(self, obs_dim, act_dim, s):
        self.s, self.device, self.act_dim = s, torch.device(s.DEVICE), act_dim
        self.actor = mlp(obs_dim, 2 * act_dim, s.HIDDEN_DIMS).to(self.device)
        self.q1 = mlp(obs_dim + act_dim, 1, s.HIDDEN_DIMS).to(self.device)
        self.q2 = mlp(obs_dim + act_dim, 1, s.HIDDEN_DIMS).to(self.device)
        self.q1_t, self.q2_t = deepcopy(self.q1), deepcopy(self.q2)
        for p in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=s.LR)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=s.LR)
        self.log_alpha = torch.tensor(np.log(s.ALPHA), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=s.LR)
        self.updates = 0

    def distribution(self, obs):
        mu, log_std = self.actor(obs).chunk(2, dim=-1)
        return mu, log_std.clamp(-20., 2.)

    def sample(self, obs, mask):
        mu, log_std = self.distribution(obs)
        z = mu + log_std.exp() * torch.randn_like(mu)
        logp = -.5 * ((z - mu) / log_std.exp()).square() - log_std - .5 * np.log(2 * np.pi)
        logp -= 2 * (np.log(2) - z - nn.functional.softplus(-2 * z))
        return z.tanh() * mask, (logp * mask).sum(-1, keepdim=True)

    def act(self, observations, masks, deterministic=False):
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(np.asarray(masks), dtype=torch.float32, device=self.device)
            if deterministic:
                action = self.distribution(obs)[0].tanh() * mask
            else:
                action, _ = self.sample(obs, mask)
        return action.cpu().numpy()

    def update(self, batch):
        b = {k: v.to(self.device) for k, v in batch.items()}
        alpha = self.log_alpha.exp().detach() if self.s.AUTO_ALPHA else torch.tensor(self.s.ALPHA, device=self.device)
        with torch.no_grad():
            next_action, next_logp = self.sample(b["next_obs"], b["next_mask"])
            pair = torch.cat((b["next_obs"], next_action), -1)
            target = b["rew"] + self.s.GAMMA * (1 - b["done"]) * (
                torch.minimum(self.q1_t(pair), self.q2_t(pair)) - alpha * next_logp)
        pair = torch.cat((b["obs"], b["act"]), -1)
        critic_loss = (self.q1(pair) - target).square().mean() + (self.q2(pair) - target).square().mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        for p in critic_parameters:
            p.requires_grad_(False)
        action, logp = self.sample(b["obs"], b["mask"])
        pair = torch.cat((b["obs"], action), -1)
        actor_loss = (alpha * logp - torch.minimum(self.q1(pair), self.q2(pair))).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for p in critic_parameters:
            p.requires_grad_(True)

        alpha_loss = torch.zeros((), device=self.device)
        if self.s.AUTO_ALPHA:
            target_entropy = -b["mask"].sum(-1, keepdim=True)
            alpha_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
        with torch.no_grad():
            for source, target_net in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, target_parameter in zip(source.parameters(), target_net.parameters()):
                    target_parameter.lerp_(p, self.s.TAU)
        self.updates += 1
        values = [critic_loss.item(), actor_loss.item(), alpha_loss.item(), self.log_alpha.exp().item()]
        if not np.isfinite(values).all() or not all(torch.isfinite(p).all() for net in self.networks().values() for p in net.parameters()):
            raise RuntimeError("Nonfinite SAC update; resume the last episode checkpoint")
        return dict(zip(("critic_loss", "actor_loss", "alpha_loss", "alpha"), values))

    def networks(self):
        return dict(actor=self.actor, q1=self.q1, q2=self.q2, q1_target=self.q1_t, q2_target=self.q2_t)

    def state(self, training=False):
        saved = dict(networks={k: n.state_dict() for k, n in self.networks().items()},
                     log_alpha=self.log_alpha.detach().cpu(), updates=self.updates)
        if training:
            saved["optimizers"] = dict(actor=self.actor_opt.state_dict(), critic=self.critic_opt.state_dict(), alpha=self.alpha_opt.state_dict())
        return saved

    def restore(self, saved, training=False):
        for key, net in self.networks().items():
            net.load_state_dict(saved["networks"][key], strict=True)
            if not all(torch.isfinite(v).all() for v in net.state_dict().values()):
                raise ValueError("Nonfinite SAC checkpoint")
        with torch.no_grad():
            self.log_alpha.copy_(saved["log_alpha"].to(self.device))
        self.updates = int(saved["updates"])
        if not torch.isfinite(self.log_alpha) or self.updates < 0:
            raise ValueError("Invalid temperature/update count")
        if training:
            for key, optimizer in (("actor", self.actor_opt), ("critic", self.critic_opt), ("alpha", self.alpha_opt)):
                optimizer.load_state_dict(saved["optimizers"][key])

    def sizes(self):
        return {k: sum(p.numel() for p in net.parameters()) for k, net in self.networks().items()}
