from pettingzoo.classic import go_v5
import numpy as np

BOARD_SIZE = 9
NUM_EPISODES = 50
PRINT_EVERY = 10

episode_rewards = {
    "black_0": [],
    "white_0": []
}

print("=" * 50)
print("Random Agent - Go Environment")
print(f"Board: {BOARD_SIZE}x{BOARD_SIZE}")
print(f"Playing {NUM_EPISODES} episodes...")
print("=" * 50)

for episode in range(NUM_EPISODES):

    env = go_v5.env(board_size=BOARD_SIZE, komi=7.5)
    env.reset()

    episode_reward = {"black_0": 0, "white_0": 0}
    step_count = 0

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        episode_reward[agent] += reward

        if termination or truncation:
            action = None
        else:
            # ── THE FIX: only sample from LEGAL moves ──
            action_mask = observation["action_mask"]
            legal_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(legal_actions)

        env.step(action)
        step_count += 1

    env.close()

    episode_rewards["black_0"].append(episode_reward["black_0"])
    episode_rewards["white_0"].append(episode_reward["white_0"])

    if (episode + 1) % PRINT_EVERY == 0:
        black_rewards = episode_rewards["black_0"][-PRINT_EVERY:]
        white_rewards = episode_rewards["white_0"][-PRINT_EVERY:]

        black_wins = sum(1 for r in black_rewards if r > 0)
        white_wins = sum(1 for r in white_rewards if r > 0)

        print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")
        print(f"  Last {PRINT_EVERY} games:")
        print(f"  Black wins: {black_wins}/{PRINT_EVERY}")
        print(f"  White wins: {white_wins}/{PRINT_EVERY}")
        print(f"  Steps in last game: {step_count}")

print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)

total_black_wins = sum(1 for r in episode_rewards["black_0"] if r > 0)
total_white_wins = sum(1 for r in episode_rewards["white_0"] if r > 0)
draws = NUM_EPISODES - total_black_wins - total_white_wins

print(f"Total episodes: {NUM_EPISODES}")
print(f"Black wins: {total_black_wins} ({total_black_wins/NUM_EPISODES*100:.1f}%)")
print(f"White wins: {total_white_wins} ({total_white_wins/NUM_EPISODES*100:.1f}%)")
print(f"Draws/incomplete: {draws}")
print(f"\nAverage reward per episode:")
print(f"  Black: {np.mean(episode_rewards['black_0']):.3f}")
print(f"  White: {np.mean(episode_rewards['white_0']):.3f}")
print("\nBaseline established! Random agent complete.")