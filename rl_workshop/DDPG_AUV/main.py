import argparse
import sys
import gymnasium as gym
import numpy as np

import torch
import torch.optim as optim

from agent import ReplayBuffer, OrnsteinUhlenbeckNoise, Actor, Critic, train, soft_update

def setup_arguments():
    parser = argparse.ArgumentParser(description="DDPG-AUV Simulation")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--episodes', type=int, default=10000, help="episodes")
    return parser.parse_args()


def main():
    args = setup_arguments()

    env = gym.make('Pendulum_v1', max_episode_stpes=200, autoreset=True)
    memory = ReplayBuffer()

    critic, critic_target = Critic(), Critic()
    critic_target.load_state_dict(critic.state_dict())
    actor, actor_target = Actor(), Actor()
    actor_target.load_state_dict(actor.state_dict())

    score = 0.0
    print_interval = 20
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr)
    ou_noise = OrnsteinUhlenbeckNoise(actor=np.zeros(1))

    for ep in range(args.episodes):
        s, _ = env.reset()
        done = False

        count = 0
        while count < 200 and not done:
            a = actor(torch.from_numpy(s).float())
            a = a.item() + ou_noise()[0]
            s_prime, r, done, truncated, info = env.step([a])
            memory.put((s, a, r/100.0, s_prime, done))
            score += r
            s = s_prime
            count += 1

        if memory.size() > 2000:
            for i in range(10):
                train(actor, actor_target, critic, critic_target, memory, critic_optimizer, actor_optimizer)
                soft_update(actor, actor_target)
                soft_update(critic, critic_target)
        
        if ep % print_interval == 0 and ep != 0:
            print("# of episode: {}, avg score: {:.1f}".format(ep, score/print_interval))
            score = 0.0

    env.close()


if __name__ == "__main__":
    main()