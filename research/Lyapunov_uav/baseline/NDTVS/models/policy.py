"""NDTVS network, optimizer and policy construction."""
from __future__ import annotations

from dataclasses import replace
import torch
from torch import nn
from torch.distributions import Categorical
from baseline.NDTVS.common.paths import HERE
from hppo.ppo import PPOAgent


def mlp(d):
    return nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh())


class NDTVSNet(nn.Module):
    """One PPO policy: frame-gated scheduling and slot chunk/quality choices.

    J pointer draws select distinct users or STOP. Then each scheduled user
    selects idle or (chunks, quality). Replay reconstructs exactly the same
    conditional support. Unused choices have probability one.
    """
    def __init__(self, cfg):
        super().__init__()
        self.N, self.J = cfg.num_users, cfg.rsu_capacity
        self.D = 1 + cfg.max_chunks_per_slot * cfg.num_quality_levels
        self.obs_dim = 4 + 10 * self.N
        self.actor = mlp(10)
        self.context = mlp(68)
        self.score = nn.Linear(128, 1)
        self.stop = nn.Linear(64, 1)
        self.download = nn.Linear(128, self.D)
        self.critic = mlp(10)
        self.value_head = nn.Sequential(mlp(68), nn.Linear(64, 1))
        for head in (self.score, self.stop, self.download):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def encode(self, obs, encoder):
        x = obs[:, 4:].reshape(-1, self.N, 10)
        e = encoder(x)
        present = x[:, :, 0:1]
        pooled = (e * present).sum(1) / present.sum(1).clamp(min=1)
        return x, e, torch.cat((obs[:, :4], pooled), -1)

    def value(self, obs):
        return self.value_head(self.encode(obs, self.critic)[2]).squeeze(-1)

    def decode(self, obs, masks, actions=None, deterministic=False):
        x, e, ctx = self.encode(obs, self.actor)
        context = self.context(ctx)
        h = torch.cat((e, context[:, None].expand(-1, self.N, -1)), -1)
        scores = torch.cat((self.score(h).squeeze(-1), self.stop(context)), -1)
        boundary = obs[:, 0].bool()
        available = masks[0].clone()
        selected = torch.zeros_like(x[:, :, 0], dtype=torch.bool)
        ended = ~boundary
        chosen, lp, ent = [], obs.new_zeros(len(obs)), obs.new_zeros(len(obs))
        for j in range(self.J):
            valid = available.clone()
            valid[:, :self.N] &= ~ended[:, None]
            dist = Categorical(logits=scores.masked_fill(~valid, -1e9))
            a = actions[:, j] if actions is not None else (dist.logits.argmax(-1) if deterministic else dist.sample())
            if not valid.gather(1, a[:, None]).all():
                raise ValueError("invalid stored pointer action")
            chosen.append(a)
            lp, ent = lp + dist.log_prob(a), ent + dist.entropy()
            one = torch.nn.functional.one_hot(a, self.N + 1).bool()
            selected |= one[:, :self.N]
            available = available & ~one
            available[:, self.N] = True
            ended = ended | (a == self.N)
        scheduled = torch.where(boundary[:, None], selected, x[:, :, 9].bool())
        valid = torch.stack(masks[self.J:], dim=1).clone()
        valid[:, :, 1:] &= scheduled[:, :, None]
        dist = Categorical(logits=self.download(h).masked_fill(~valid, -1e9))
        a = actions[:, self.J:] if actions is not None else (dist.logits.argmax(-1) if deterministic else dist.sample())
        if not valid.gather(2, a[:, :, None]).all():
            raise ValueError("invalid stored download action")
        return (torch.cat((torch.stack(chosen, 1), a), 1),
                lp + dist.log_prob(a).sum(1), self.value(obs), ent + dist.entropy().sum(1))

    @torch.no_grad()
    def act(self, obs, masks, deterministic=False):
        return tuple(x.squeeze(0) for x in self.decode(obs.reshape(1, -1), masks, deterministic=deterministic))

    def evaluate_actions(self, obs, actions, masks):
        _, lp, value, ent = self.decode(obs, masks, actions)
        return lp, ent, value


def ndt_agent(cfg):
    # Learning settings remain those of the existing adapted baseline.
    learning = replace(cfg, hidden_dims=(128, 64), ppo_gamma=0.95,
                       ppo_minibatch_size=64, ppo_gae_lambda=0.95)
    nvec = (cfg.num_users + 1,) * cfg.rsu_capacity + (
        1 + cfg.max_chunks_per_slot * cfg.num_quality_levels,) * cfg.num_users
    agent = PPOAgent(4 + 10 * cfg.num_users, nvec, learning, "ndtvs_ppo")
    agent.net = NDTVSNet(cfg).to(agent.device)
    critic = list(agent.net.critic.parameters()) + list(agent.net.value_head.parameters())
    ids = {id(p) for p in critic}
    actor = [p for p in agent.net.parameters() if id(p) not in ids]
    agent.opt = torch.optim.Adam([{"params": actor, "lr": 1e-5}, {"params": critic, "lr": 1e-4}])
    return agent


def make_agents(cfg, algorithm):
    if algorithm == "ndtvs":
        return [ndt_agent(cfg)]
    return [PPOAgent(cfg.frame_obs_dim, cfg.frame_action_nvec, cfg, "frame_ppo"),
            PPOAgent(cfg.slot_obs_dim, cfg.slot_action_nvec, cfg, "slot_ppo")]
