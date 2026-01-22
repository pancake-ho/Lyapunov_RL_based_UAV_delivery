from agent import train
from utils import visualize_policy

def main():
    trained_agent, env = train()
    print("Visualizing policy...")
    visualize_policy(env, trained_agent)

if __name__ == "__main__":
    main()