import numpy as np
import matplotlib.pyplot as plt

def visualize_policy(env, agent, episodes=5):
    successful_episodes = 0
    max_attempts = 100
    while successful_episodes < episodes and max_attempts > 0:
        state = env.reset()
        done = False
        path = [env.agent_pos.copy()]

        while not done:
            action, _ = agent.select_action(state)
            state, reward, done = env.step(action)
            path.append(env.agent_pos.copy())
        
        if tuple(env.agent_pos) == env.goal:
            successful_episodes += 1

            grid = np.zeros((env.grid_size, env.grid_size))
            for obs in env.obstacles:
                grid[obs[0], obs[1]] = -1
            for pos in path:
                grid[pos[0], pos[1]] = 0.5
            grid[env.goal[0], env.goal[1]] = 1
            plt.imshow(grid, cmap="gray")
            plt.title(f"Success Episode {successful_episodes}")
            plt.show()
        
        max_attempts = -1