"""
Soft Actor-Critic (Haarnoja et al. 2018) as used by the reference paper:
Gaussian actor with tanh squashing (reparameterization, eq. 44), twin soft-Q
critics with target networks (soft update, eq. 42), and automatic entropy
temperature tuning.  One policy is shared by all regions/UAVs
(parameter sharing); every region contributes transitions to one replay buffer.

torch is imported lazily so that ``--agent random`` works without it.
"""
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros((size, 1), np.float32)
        self.obs2 = np.zeros((size, obs_dim), np.float32)
        self.done = np.zeros((size, 1), np.float32)
        self.ptr, self.n, self.size = 0, 0, size

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr], self.act[self.ptr], self.rew[self.ptr] = o, a, r
        self.obs2[self.ptr], self.done[self.ptr] = o2, float(d)
        self.ptr = (self.ptr + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, size=batch)
        return self.obs[idx], self.act[idx], self.rew[idx], self.obs2[idx], self.done[idx]


class SACAgent:
    def __init__(self, obs_dim, act_dim, args):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.args = args
        self.device = torch.device(args.device)
        self.act_dim = act_dim
        h = args.hidden

        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))

        self.actor = mlp(obs_dim, 2 * act_dim).to(self.device)
        self.q1 = mlp(obs_dim + act_dim, 1).to(self.device)
        self.q2 = mlp(obs_dim + act_dim, 1).to(self.device)
        self.q1_t = mlp(obs_dim + act_dim, 1).to(self.device)
        self.q2_t = mlp(obs_dim + act_dim, 1).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.log_alpha = torch.tensor(np.log(args.alpha), dtype=torch.float32, device=self.device,
                                      requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(act_dim)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item() if self.args.auto_alpha else self.args.alpha

    # ---------------------------------------------------------------- policy
    def _dist(self, obs):
        out = self.actor(obs)
        mu, log_std = out[:, :self.act_dim], out[:, self.act_dim:]
        log_std = self.torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def _sample(self, obs):
        torch = self.torch
        mu, log_std = self._dist(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        u = mu + eps * std                       # reparameterized (eq. 44)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        logp -= (2 * (np.log(2) - u - torch.nn.functional.softplus(-2 * u))).sum(-1, keepdim=True)
        return a, logp, torch.tanh(mu)

    def act(self, obs, deterministic=False):
        torch = self.torch
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            a, _, mu = self._sample(o)
            return (mu if deterministic else a).squeeze(0).cpu().numpy()

    # ---------------------------------------------------------------- update
    def update(self, batch):
        torch = self.torch
        o, a, r, o2, d = [torch.as_tensor(x, device=self.device) for x in batch]
        gamma, tau = self.args.gamma, self.args.tau
        alpha = self.log_alpha.exp().detach() if self.args.auto_alpha else torch.tensor(self.args.alpha)

        with torch.no_grad():
            a2, logp2, _ = self._sample(o2)
            q_t = torch.min(self.q1_t(torch.cat([o2, a2], -1)), self.q2_t(torch.cat([o2, a2], -1)))
            y = r + gamma * (1 - d) * (q_t - alpha * logp2)        # soft Bellman target (eq. 39)
        q1 = self.q1(torch.cat([o, a], -1))
        q2 = self.q2(torch.cat([o, a], -1))
        critic_loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a_new, logp, _ = self._sample(o)
        q_new = torch.min(self.q1(torch.cat([o, a_new], -1)), self.q2(torch.cat([o, a_new], -1)))
        actor_loss = (alpha * logp - q_new).mean()                # eq. 45
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = torch.tensor(0.0)
        if self.args.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - tau).add_(tau * p)
        self.updates += 1
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "alpha_loss": float(alpha_loss), "alpha": self.alpha}

    # -------------------------------------------------------------- save/load
    def save(self, path):
        self.torch.save({"actor": self.actor.state_dict(), "q1": self.q1.state_dict(),
                         "q2": self.q2.state_dict(), "log_alpha": self.log_alpha.detach().cpu()}, path)

    def load(self, path):
        ck = self.torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.q1.load_state_dict(ck["q1"]); self.q2.load_state_dict(ck["q2"])
        self.q1_t.load_state_dict(ck["q1"]); self.q2_t.load_state_dict(ck["q2"])
        with self.torch.no_grad():
            self.log_alpha.copy_(ck["log_alpha"].to(self.device))
