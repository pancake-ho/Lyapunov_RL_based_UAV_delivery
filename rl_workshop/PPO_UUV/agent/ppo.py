import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical
from .network import PolicyNetwork

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, clip_epsilon=0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.policy(state)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def update(self, trajectories):
        states, actions, rewards, old_log_probs = zip(*trajectories)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.stack(old_log_probs)

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        for _ in range(10):
            logits = self.policy(states)
            distribution = Categorical(logits=logits)
            log_probs = distribution.log_prob(actions)

            ratios = torch.exp(log_probs - old_log_probs.detach())
            advantages = returns.detach() - returns.mean()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(dict))