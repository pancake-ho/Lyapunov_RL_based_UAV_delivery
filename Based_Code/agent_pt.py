import numpy as np
import random
from collections import defaultdict, deque
import copy
import math
import matplotlib.pyplot as plt
import sys
from typing import Tuple, List, Optional
import os

import torch
import torch.nn as nn
import torch.optim as optim

from environment_pt import Env

np.random.seed(7)
torch.manual_seed(1)
random.seed(7)

directory = './DTRL_downgrade/result_plot/'
os.makedirs(directory, exist_ok=True)
log_file_name = 'result_log.txt'
SNR_list = [[10**1.5, '15dB'], [10**2.0, '20dB'], [10**2.5, '25dB'], [10**3.0, '30dB']]

class DQN_MQ(nn.Module):
    """
    Chunk Delivery 전용 DQN
    입력: [Q_t, n_t]
    출력: (action_size_MQ)
    """
    def __init__(self, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(2, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc_out = nn.Linear(24, action_size)
        self.relu = nn.ReLU()
        nn.init.uniform_(self.fc_out.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q = self.fc_out(x)
        return q

class DQN_Scheduling(nn.Module):
    """
    Link Scheduling 전용 DQN
    입력: node_state
    출력: action_size_scheduling
    """
    def __init__(self, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(10, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc_out = nn.Linear(24, action_size)
        self.relu = nn.ReLU()
        nn.init.uniform_(self.fc_out.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q = self.fc_out(x)
        return q

class DQNAgent:
    def __init__(self, state_size: int, action_size_MQ: int, action_size_scheduling: int):
        self.state_size = state_size
        self.action_size_MQ = action_size_MQ
        self.action_size_scheduling = action_size_scheduling
        self.action_list = [
        (0,0),
        (1,1),(1,2),(1,3),
        (2,1),(2,2),(2,3),
        (3,1),(3,2),(3,3),
        (4,1),(4,2),(4,3),
        (5,1),(5,2),(5,3),
        (6,1),(6,2),(6,3),
        (7,1),(7,2),(7,3),
        (8,1),(8,2),(8,3),
        (9,1),(9,2),(9,3)
        ]

        self.lr = 0.001
        self.discount_factor = 0.5
        self.epsilon = 1.0
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.1
        self.batch_size_MQ = 64
        self.batch_size_S = 64
        self.train_start = 500
        self.q_table = defaultdict(float)

        self.memory_MQ: deque = deque(maxlen=2000)
        self.memory_scheduling: deque = deque(maxlen=2000)

        self.behavior_MQ_model = DQN_MQ(self.action_size_MQ)
        self.target_MQ_model = DQN_MQ(self.action_size_MQ)
        self.behavior_S_model = DQN_Scheduling(self.action_size_scheduling)
        self.target_S_model = DQN_Scheduling(self.action_size_scheduling)

        self.optimizer = optim.Adam(list(self.behavior_MQ_model.parameters()) + list(self.behavior_S_model.parameters()), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.update_target_MQ_model()
        self.update_target_S_model()

        self.T = 5 # Large Time Scale

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.behavior_MQ_model.to(self.device)
        self.target_MQ_model.to(self.device)
        self.behavior_S_model.to(self.device)
        self.target_S_model.to(self.device)
    
    @staticmethod
    def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    
    def update_target_MQ_model(self):
        self.target_MQ_model.load_state_dict(self.behavior_MQ_model.state_dict())
    
    def update_target_S_model(self):
        self.target_S_model.load_state_dict(self.behavior_S_model.state_dict())
    
    def append_MQ_sample(self, state, action, reward, next_state, done):
        self.memory_MQ.append((state, action, reward, next_state, done))
    
    def append_S_sample(self, state, action, reward, next_state, done):
        self.memory_scheduling.append((state, action, reward, next_state, done))
    
    def action_transform(self, action: Tuple[int, int]) -> int:
        for i, a in enumerate(self.action_list):
            if action == a:
                return i
        return 0
    
    def train_MQ_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory_MQ, self.batch_size_MQ)

        states = np.array([sample[0] for sample in mini_batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in mini_batch], dtype=np.int64)
        rewards = np.array([sample[2] for sample in mini_batch], dtype=np.float32)
        next_states = np.array([sample[3][0] for sample in mini_batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in mini_batch], dtype=np.float32)
        
        states_t = self._to_tensor(states, self.device)
        next_states_t = self._to_tensor(next_states, self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        q_pred_all = self.behavior_MQ_model(states_t)
        q_pred = q_pred_all.gather(1, actions_t.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next_all = self.target_MQ_model(next_states_t)
            max_q_next, _ = torch.max(q_next_all, dim=1)
            targets = rewards_t + (1.0 - dones_t) * self.discount_factor * max_q_next
        loss = self.loss_fn(q_pred, targets)
        loss.backward()
        self.optimizer.step()
    
    def train_S_model(self):
        mini_batch = random.sample(self.memory_scheduling, self.batch_size_S)

        states = np.array([sample[0][0] for sample in mini_batch], dtype=np.float32) # [B, 10]
        actions = np.array([sample[1] for sample in mini_batch], dtype=np.int64) # index (0..4)
        rewards = np.array([sample[2] for sample in mini_batch], dtype=np.float32)
        next_states = np.array([sample[3][0] for sample in mini_batch], dtype=np.float32) # [B, 10]
        dones = np.array([sample[4] for sample in mini_batch], dtype=np.float32)

        states_t = self._to_tensor(states, self.device)
        next_states_t = self._to_tensor(next_states, self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        q_pred_all = self.behavior_S_model(states_t)
        q_pred = q_pred_all.gather(1, actions_t.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next_all = self.target_S_model(next_states_t)
            max_q_next, _ = torch.max(q_next_all, dim=1)
            targets = rewards_t + (1.0 - dones_t) * self.discount_factor * max_q_next
        loss = self.loss_fn(q_pred, targets)
        loss.backward()
        self.optimizer.step()
    
    def get_Mq(self, state: np.ndarray, alpha: Optional[int], env: Env) -> Tuple[int, int]:
        print("current state : ", state[0])
        if alpha is None:
            print("node not in cell : (0, 0)")
            return (0, 0)
        if state[0][1] == 0:
            state[0][1] = 2
        
        node_level = env.node_quality[alpha]
        if np.random.rand() <= self.epsilon:
            possible = [i for i, a in enumerate(self.action_list) if a[1] <= node_level]
            if not possible:
                return (0, 0)
            idx = random.choice(possible)
            action = self.action_list[idx]
            print("get_action 의 epsilon-greedy policy 에 따른 action : ", action)
            return action
        else:
            state_t = self._to_tensor(state, self.device)
            q_value = self.behavior_MQ_model(state_t).detach().cpu().numpy()[0].tolist()
            copy_action_list = copy.deepcopy(self.action_list)
            while True:
                best_idx = int(np.argmax(q_value))
                action = copy_action_list[best_idx]
                if action[1] > node_level:
                    q_value.pop(best_idx)
                    copy_action_list.pop(best_idx)
                    if not copy_action_list:
                        print("No possible action")
                        return (0, 0)
                else:
                    break
            print("get_action 의 epsilon-greedy policy 에 따른 action : ", action)
            return action
    
    def get_alpha(self, env: Env):
        env.make_distance_list()
        node_state = env.check_in_cell(env.large_time)
        print("node state : ", node_state)
        if node_state is None:
            return None, None, None
        input_node_state = np.reshape(node_state, [1, 10]).astype(np.float32)
        q_value = self.behavior_S_model(self._to_tensor(input_node_state, self.device)).detach().cpu().numpy()[0]
        if np.random.rand() <= self.epsilon:
            link_node_index = random.randint(0, 4)
            while node_state[0][link_node_index] == 1:
                link_node_index = random.randint(0, 4)
            print("random link node index : ", link_node_index)
        else:
            original_q = q_value.tolist()
            copy_q = original_q.copy()
            link_node_index = int(np.argmax(q_value))
            while node_state[0][link_node_index] == 1:
                copy_q.remove(original_q[link_node_index])
                link_node_index = original_q.index(max(copy_q))
        print("link node index : ", link_node_index)
        env_link_node_index = env.distances[env.large_time].index(node_state[0][link_node_index])
        return env_link_node_index, node_state, link_node_index
    
    def test_get_alpha(self, env: Env):
        env.make_distance_list()
        node_state = env.check_in_cell(env.large_time)
        if node_state is None:
            return None, None, None
        input_node_state = np.reshape(node_state, [1, 10]).astype(np.float32)
        q_value = self.target_S_model(self._to_tensor(input_node_state, self.device)).detach().cpu().numpy()[0]
        original_q = q_value.tolist()
        link_node_index = int(np.argmax(q_value))
        while node_state[0][link_node_index] == 1:
            original_q.pop(link_node_index)
            link_node_index = int(np.argmax(original_q))
        print("link node index : ", link_node_index)
        env_link_node_index = env.distances[env.large_time].index(node_state[0][link_node_index])
        return env_link_node_index, node_state, link_node_index

    def test_get_Mq(self, state: np.ndarray, alpha: Optional[int], env: Env) -> Tuple[int, int]:
        print("current state : ", state[0])
        if alpha is None:
            print("node not in cell : (0, 0)")
            return (0, 0)
        if state[0][1] == 0:
            state[0][1] = 2
        node_level = env.node_quality[alpha]
        state_t = self._to_tensor(state, self.device)
        q_value = self.target_MQ_model(state_t).detach().cpu().numpy()[0].tolist()
        copy_action_list = copy.deepcopy(self.action_list)
        while True:
            best_idx = int(np.argmax(q_value))
            action = copy_action_list[best_idx]
            if action[1] > node_level:
                q_value.pop(best_idx)
                copy_action_list.pop(best_idx)
                if not copy_action_list:
                    return (0, 0)
            else:
                break
        print("get_action 의 greedy policy 에 따른 action : ", action)
        return action

if __name__ == "__main__":
    env = Env()
    env.transmit_SNR = SNR_list[3][0] # 실험 시 SNR index 변경

    action_MQ_size = 28
    action_S_size = 5
    state_size = 2

    agent = DQNAgent(state_size, action_MQ_size, action_S_size)

    total_reward = [0.0 for _ in range(env.max_episode)]
    total_mean_bitrate = [0.0 for _ in range(env.max_episode)]
    total_buffering = [0.0 for _ in range(env.max_episode)]

    for episode in range(env.max_episode):
        env.setting()
        prev_state = [env.user_queue, 0]
        prev_state = np.reshape(prev_state, [1, state_size]).astype(np.float32)
        large_reward = 0.0
        prev_node_state = None
        env_alpha = None

        while True:
            if (env.small_time % agent.T == 0):
                env_alpha, node_state, action_alpha = agent.get_alpha(env)
                env.save_alpha(env_alpha)
                done_slow = False
                if large_reward != 0:
                    input_prev_node_state = np.reshape([0] * 10, [1, 10]) if prev_node_state is None else np.reshape(prev_node_state, [1, 10])
                    input_node_state = np.reshape([0] * 10, [1, 10]) if node_state is None else np.reshape(node_state, [1, 10])
                    if env.large_time == len(env.user_position_x):
                        done_slow = True
                    if action_alpha is not None:
                        agent.append_S_sample(input_prev_node_state.astype(np.float32), action_alpha, float(large_reward), input_node_state.astype(np.float32), done_slow)
                prev_node_state = node_state
                large_reward = 0.0
            
            print("episode, small_time, large_time : ", episode, env.small_time, env.large_time)
            action = agent.get_Mq(prev_state, env_alpha, env)
            next_state, reward, done = env.step(env_alpha, action)
            next_state = np.reshape(next_state, [1, state_size]).astype(np.float32)

            large_reward += reward
            total_reward[episode] += reward
            
            idx = agent.action_transform(action)
            agent.append_MQ_sample(prev_state[0], idx, float(reward), next_state, done)
            print("\n\n\n")

            if len(agent.memory_MQ) >= agent.train_start:
                agent.train_MQ_model()
                if len(agent.memory_scheduling) >= agent.batch_size_S:  # 추가
                    agent.train_S_model()
            
            prev_state = next_state

            if done:
                agent.update_target_MQ_model()
                agent.update_target_S_model()
                buffering, meanbitrate = env.get_rewards()
                total_buffering[episode] = buffering
                total_mean_bitrate[episode] = meanbitrate
                print("\n끝\n")
                break
        
        total_buffering[episode] = total_buffering[episode] / max(1, env.small_time)
        if env.episode == env.max_episode:
            break
        env.reset()
    
    sys.stdout = open(f"{directory}{SNR_list[3][1]}{log_file_name}", 'w')
    print("\n\n\n\ntarget model test\n\n\n\n")
    print(f"test in {SNR_list[3][1]}\n\n")

    env2 = Env()
    env2.max_episode = 10
    env2.transmit_SNR = SNR_list[3][0]
    
    total_mean_bitrate2 = [0.0 for _ in range(env2.max_episode)]
    total_buffering2 = [0.0 for _ in range(env2.max_episode)]

    for episode2 in range(env2.max_episode):
        env2.setting()
        prev_state = [env2.user_queue, 0]
        prev_state = np.reshape(prev_state, [1, state_size]).astype(np.float32)
        large_reward = 0.0
        prev_node_state = None
        env_alpha = None

        while True:
            if (env2.small_time % agent.T == 0):
                env_alpha, node_state, action_alpha = agent.test_get_alpha(env2)
                env2.save_alpha(env_alpha)
                prev_node_state = node_state
                large_reward = 0.0
            
            print("episode, small_time, large_time : ", episode2, env2.small_time, env2.large_time)
            action = agent.test_get_Mq(prev_state, env_alpha, env2)
            next_state, reward, done = env2.step(env_alpha, action)
            next_state = np.reshape(next_state, [1, state_size]).astype(np.float32)

            large_reward += reward
            prev_state = next_state
            print("\n\n\n")

            if done:
                buffering, meanbitrate = env2.get_rewards()
                total_buffering2[episode2] = buffering
                total_mean_bitrate2[episode2] = meanbitrate
                print("\n끝\n")
                break
                
        total_buffering2[episode2] = total_buffering2[episode2] / max(1, env2.small_time)
        env2.plot_graph()
        env2.reset()
        if episode2 == env2.max_episode - 1:
            break
    
    # Summaries
    print(f"{SNR_list[3][1]} mean of buffering in Test : ", np.mean(total_buffering2))
    print(f"{SNR_list[3][1]} mean of bitrate in Test : ", np.mean(total_mean_bitrate2))

    # --------------------- Save target models ---------------------
    torch.save(agent.target_S_model.state_dict(), os.path.join(directory, f"{SNR_list[3][1]}_target_S_model.pt"))
    torch.save(agent.target_MQ_model.state_dict(), os.path.join(directory, f"{SNR_list[3][1]}_target_MQ_model.pt"))

    # --------------------- Plots ---------------------
    plt.figure('train_reward')
    plt.xlabel('episode')
    plt.ylabel('total_reward')
    plt.plot(total_reward)
    plt.title('reward at each episode in Train')
    plt.savefig(f"{directory}{SNR_list[3][1]}train_reward.png")
    plt.cla(); plt.clf(); plt.close('all')

    plt.figure('train_bitrate')
    plt.xlabel('episode')
    plt.ylabel('total_mean_bitrate')
    plt.plot(total_mean_bitrate)
    plt.title('bitrate at each episode in Train')
    plt.savefig(f"{directory}{SNR_list[3][1]}train_bitrate.png")
    plt.cla(); plt.clf(); plt.close('all')

    plt.figure('train_buffering')
    plt.xlabel('episode')
    plt.ylabel('total_buffering')
    plt.plot(total_buffering)
    plt.title('buffering at each episode in Train')
    plt.savefig(f"{directory}{SNR_list[3][1]}train_buffering.png")
    plt.cla(); plt.clf(); plt.close('all')

    plt.figure('test_bitrate')
    plt.xlabel('episode')
    plt.ylabel('test_mean_bitrate')
    plt.plot(total_mean_bitrate2)
    plt.title('bitrate at each episode in Test')
    plt.savefig(f"{directory}{SNR_list[3][1]}test_bitrate.png")
    plt.cla(); plt.clf(); plt.close('all')

    plt.figure('test_buffering')
    plt.xlabel('episode')
    plt.ylabel('test_buffering')
    plt.plot(total_buffering2)
    plt.title('buffering at each episode in Test')
    plt.savefig(f"{directory}{SNR_list[3][1]}test_buffering.png")
    plt.cla(); plt.clf(); plt.close('all')

    sys.stdout.close()
