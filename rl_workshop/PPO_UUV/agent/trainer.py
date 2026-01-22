from env import GridEnvironment
from agent import PPOAgent

def train():
    env = GridEnvironment()
    state_dim = env.grid_size ** 2
    action_dim = 4
    agent = PPOAgent(state_dim, action_dim)
    model_path = "ppo_policy.pth"
    rewards_history = []

    num_episode = 1000

    for episode in range(num_episode):
        state = env.reset()
        done = False
        episode_rewards = 0
        trajectory = []

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)

            trajectory.append((state, action, reward, log_prob))
            state = next_state
            episode_rewards += reward

        rewards_history.append(episode_rewards)
        agent.update(trajectory)
        print(f"Episode {episode}: Total Reward = {episode_rewards}")

        if (episode + 1) % 100 == 0:
            agent.save_model(model_path)
    
    print("Training Complete. Model saved.")
    return agent, env