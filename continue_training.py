from pettingzoo.classic import go_v5
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os

# ── paste your GoEnvWrapper class here ────────────────────────
# (copy it from ppo_agent.py — you need it every time you load)
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

# ── Load the saved model ───────────────────────────────────────
CHECKPOINT = "./models/ppo/ppo_go_final.zip"  # change to any checkpoint
LOG_DIR = "./logs/ppo/"

print(f"Loading model from: {CHECKPOINT}")
env = GoEnvWrapper(board_size=19)

# Load model and give it the environment
model = PPO.load(CHECKPOINT, env=env)
print("Model loaded successfully!")

# ── Continue training from where it left off ──────────────────
print("Continuing training for another 500,000 steps...")
model.learn(
    total_timesteps=500_000,
    reset_num_timesteps=False,  # keeps the step counter going from where it was
    tb_log_name="ppo_run_2",    # new run name in TensorBoard
    tensorboard_log=LOG_DIR
)

# Save the updated model
model.save("./models/ppo/ppo_go_continued.zip")
print("Continued training saved!")