from pettingzoo.classic import go_v5
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ── paste GoEnvWrapper here (same as above) ───────────────────
class GoEnvWrapper(gym.Env):
    def __init__(self, board_size=19, komi=7.5):
        super().__init__()
        self.board_size = board_size
        self.komi = komi
        self.env = go_v5.env(board_size=board_size, komi=komi)
        self.action_space = spaces.Discrete(board_size * board_size + 1)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(board_size * board_size * 17,),
            dtype=np.float32
        )
        self.episode_count = 0
        self.win_count = 0

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        obs, _, _, _, _ = self.env.last()
        return self._process_obs(obs), {}

    def _process_obs(self, obs):
        return obs['observation'].flatten().astype(np.float32)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.last()
        action_mask = obs['action_mask']
        if action_mask[action] == 0:
            legal = np.where(action_mask == 1)[0]
            action = int(np.random.choice(legal))
        self.env.step(int(action))
        if not term and not trunc:
            opp_obs, _, opp_term, opp_trunc, _ = self.env.last()
            if not opp_term and not opp_trunc:
                opp_mask = opp_obs['action_mask']
                legal_opp = np.where(opp_mask == 1)[0]
                self.env.step(int(np.random.choice(legal_opp)))
        next_obs, next_reward, next_term, next_trunc, _ = self.env.last()
        if next_term or next_trunc:
            self.episode_count += 1
            if next_reward > 0:
                self.win_count += 1
        return self._process_obs(next_obs), float(next_reward), next_term, next_trunc, {}

    def close(self):
        self.env.close()

# ── Configuration ──────────────────────────────────────────────
CHECKPOINT = "./models/ppo/ppo_go_final.zip"
NUM_EVAL_GAMES = 100
RANDOM_BASELINE = 0.40

# ── Load ───────────────────────────────────────────────────────
print(f"Loading: {CHECKPOINT}")
env = GoEnvWrapper(board_size=19)
model = PPO.load(CHECKPOINT, env=env)
print("Loaded successfully!\n")

# ── Evaluate ───────────────────────────────────────────────────
print(f"Running {NUM_EVAL_GAMES} evaluation games...")
print("-" * 40)

wins = 0
total_steps = 0
rewards = []

for game in range(NUM_EVAL_GAMES):
    obs, _ = env.reset()
    done = False
    game_steps = 0
    game_reward = 0

    while not done:
        # deterministic=True means always pick the best move, no randomness
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        game_steps += 1
        game_reward = reward

    total_steps += game_steps
    rewards.append(game_reward)
    if game_reward > 0:
        wins += 1

    # Print every 10 games
    if (game + 1) % 10 == 0:
        current_wr = wins / (game + 1)
        print(f"Game {game+1:3d}/{NUM_EVAL_GAMES} | "
              f"Win rate so far: {current_wr:.1%}")

# ── Results ────────────────────────────────────────────────────
print("\n" + "=" * 40)
print("EVALUATION RESULTS")
print("=" * 40)
win_rate = wins / NUM_EVAL_GAMES
print(f"Games played:      {NUM_EVAL_GAMES}")
print(f"Wins:              {wins}")
print(f"Win rate:          {win_rate:.1%}")
print(f"Random baseline:   {RANDOM_BASELINE:.1%}")
print(f"Improvement:       {(win_rate - RANDOM_BASELINE)*100:+.1f}%")
print(f"Avg game length:   {total_steps//NUM_EVAL_GAMES} steps")
print(f"Avg reward:        {np.mean(rewards):.3f}")

if win_rate > RANDOM_BASELINE:
    print(f"\n✓ Agent is BETTER than random baseline")
else:
    print(f"\n✗ Agent is NOT yet better than random — needs more training")

env.close()