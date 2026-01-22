import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
import random

seed = 2025
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GridEnvironment:
    def __init__(self, grid_size=6, goal=(5, 5)):
        self.grid_size = grid_size
        self.goal = goal

        self.reset()
    
    def reset(self):
        self.agent_pos = [0, 0]
        self.last_action = None
        self.obstacles = self._generate_obstacles()
        return self._get_state()
    
    def _generate_obstacles(self):
        obstacles = set()
        while len(obstacles) < 3:
            obs = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if obs != tuple(self.agent_pos) and obs != self.goal:
                obstacles.add(obs)
        return obstacles
    
    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        state[self.agent_pos[0], self.agent_pos[1]] = 1
        for obs in self.obstacles:
            state[obs[0], obs[1]] = -1
        return state.flatten()
    
    def step(self, action):
        next_pos = self.agent_pos.copy()
        if action == 0 and self.agent_pos[0] > 0:
            next_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:
            next_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            next_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.grid_size - 1:
            next_pos[1] += 1

        if tuple(next_pos) in self.obstacles:
            reward = -1
            done = True
        else:
            self.agent_pos = next_pos
            reward = 1 if tuple(self.agent_pos) == self.goal else -0.1
            if self.last_action is not None and self.last_action != action:
                reward -= 0.1
            
            done = tuple(self.agent_pos) == self.goal
        
        self.last_action = action
        return self._get_state(), reward, done
    