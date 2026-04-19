from pettingzoo.classic import go_v5
import numpy as np

# Create a 9x9 Go environment
# board_size=9 is manageable for training, 19x19 would take months
env = go_v5.env(board_size=9, komi=7.5)

# Reset the environment to start a new game
env.reset()

print(f"Environment: {env}")
print(f"Agents: {env.agents}")
print(f"Board size: 9x9")

# Look at what the environment gives us
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    print(f"\nAgent: {agent}")
    print(f"Observation shape: {observation['observation'].shape}")
    print(f"Action space: {env.action_space(agent)}")
    print(f"Number of possible actions: {env.action_space(agent).n}")
    print(f"Reward: {reward}")

    if termination or truncation:
        action = None
    else:
        # Take a random action for now
        action = env.action_space(agent).sample()
        print(f"Taking action: {action}")

    env.step(action)

    # Just show first two turns for now
    if agent == env.agents[-1]:
        break

env.close()
print("\nGo environment is working correctly!")