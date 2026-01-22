import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mu = nn.Linear(64, a_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = np.pi * torch.tanh(self.mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(s_dim, 64)
        self.fc_a = nn.Linear(a_dim, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
    
    def forward(self, s, a):
        h1 = F.relu(self.fc_s(s))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q
    