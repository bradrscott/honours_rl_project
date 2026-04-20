from pettingzoo.classic import go_v5
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import os

# ── Configuration ──────────────────────────────────────────────
BOARD_SIZE = 19
TOTAL_TIMESTEPS = 500_000      # increase this for real training
SAVE_EVERY = 50_000            # save a checkpoint every N steps
LOG_DIR = "./logs/ppo/"        # tensorboard logs go here
SAVE_DIR = "./models/ppo/"     # saved models go here

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Step 1: Wrap the Go environment for SB3 ───────────────────
# SB3 expects a single-agent gymnasium environment.
# This wrapper makes PettingZoo's two-agent Go look like
# a single-agent env — our agent plays black, opponent plays
# randomly as white.

class GoEnvWrapper(gym.Env):
    def __init__(self, board_size=19, komi=7.5):
        super().__init__()
        self.board_size = board_size
        self.komi = komi
        self.env = go_v5.env(board_size=board_size, komi=komi)

        # 362 possible actions: 19x19 positions + 1 pass move
        self.action_space = spaces.Discrete(board_size * board_size + 1)

        # Flatten the (19, 19, 17) board into a 1D vector for the network
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
        # Flatten board from (19,19,17) to (6137,) float32 vector
        return obs['observation'].flatten().astype(np.float32)

    def step(self, action):
        # Get current observation and action mask
        obs, reward, term, trunc, info = self.env.last()
        action_mask = obs['action_mask']

        # If PPO picks an illegal move, redirect to a random legal one
        # This prevents instant game-over penalties during early training
        if action_mask[action] == 0:
            legal = np.where(action_mask == 1)[0]
            action = int(np.random.choice(legal))

        # Our agent (black) takes its move
        self.env.step(int(action))

        # Opponent (white) takes a random legal move immediately after
        if not term and not trunc:
            opp_obs, _, opp_term, opp_trunc, _ = self.env.last()
            if not opp_term and not opp_trunc:
                opp_mask = opp_obs['action_mask']
                legal_opp = np.where(opp_mask == 1)[0]
                self.env.step(int(np.random.choice(legal_opp)))

        # Get the result from our agent's perspective
        next_obs, next_reward, next_term, next_trunc, _ = self.env.last()

        # Track wins for logging
        if next_term or next_trunc:
            self.episode_count += 1
            if next_reward > 0:
                self.win_count += 1

        return (
            self._process_obs(next_obs),
            float(next_reward),
            next_term,
            next_trunc,
            {}
        )

    def close(self):
        self.env.close()


# ── Step 2: Custom callback for extra logging ─────────────────
# This logs win rate to TensorBoard so you can track it live

class WinRateCallback(BaseCallback):
    def __init__(self, check_every=10_000, verbose=0):
        super().__init__(verbose)
        self.check_every = check_every
        self.last_check = 0

    def _on_step(self):
        if self.num_timesteps - self.last_check >= self.check_every:
            # SB3 wraps env in Monitor then DummyVecEnv
            # so we need to unwrap to get to our custom attributes
            env = self.training_env.envs[0].env

            if env.episode_count > 0:
                win_rate = env.win_count / env.episode_count
                self.logger.record("custom/win_rate", win_rate)
                self.logger.record("custom/total_episodes", env.episode_count)
                print(f"  Timestep {self.num_timesteps:,} | "
                      f"Episodes: {env.episode_count} | "
                      f"Win rate: {win_rate:.1%}")

            self.last_check = self.num_timesteps
        return True

# ── Step 3: Checkpoint callback ───────────────────────────────
# Saves your model every N steps so you never lose progress

class CheckpointCallback(BaseCallback):
    def __init__(self, save_every, save_dir, verbose=0):
        super().__init__(verbose)
        self.save_every = save_every
        self.save_dir = save_dir
        self.last_save = 0

    def _on_step(self):
        if self.num_timesteps - self.last_save >= self.save_every:
            path = os.path.join(
                self.save_dir,
                f"ppo_go_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            print(f"  Checkpoint saved: {path}")
            self.last_save = self.num_timesteps
        return True


# ── Step 4: Create and verify the environment ─────────────────
print("=" * 55)
print("PPO Agent Training — 19x19 Go")
print("=" * 55)

print("\nSetting up environment...")
env = GoEnvWrapper(board_size=BOARD_SIZE)
check_env(env, warn=True)
print("Environment check passed!")

# ── Step 5: Create the PPO model ──────────────────────────────
print("\nCreating PPO model...")
model = PPO(
    "MlpPolicy",          # multilayer perceptron — standard neural network
    env,
    verbose=1,            # print training progress
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,   # how fast the network learns
    n_steps=2048,         # steps collected before each update
    batch_size=64,        # how many samples per gradient update
    n_epochs=10,          # how many times to reuse each batch
    gamma=0.99,           # discount factor — how much future rewards matter
    clip_range=0.2,       # the PPO clipping parameter
    ent_coef=0.01,        # encourages exploration
    device="cpu",         # use cpu (mps can cause issues with SB3)
)

print(f"\nModel policy network:")
print(model.policy)

# ── Step 6: Train! ────────────────────────────────────────────
print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...")
print(f"TensorBoard logs: {LOG_DIR}")
print(f"Model checkpoints: {SAVE_DIR}")
print(f"\nTo watch training live, open a NEW terminal and run:")
print(f"  tensorboard --logdir {LOG_DIR}")
print(f"Then open: http://localhost:6006\n")

callbacks = [
    WinRateCallback(check_every=10_000),
    CheckpointCallback(save_every=SAVE_EVERY, save_dir=SAVE_DIR)
]

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    tb_log_name="ppo_run_1"
)

# ── Step 7: Save the final model ──────────────────────────────
final_path = os.path.join(SAVE_DIR, "ppo_go_final")
model.save(final_path)
print(f"\nFinal model saved to: {final_path}")

# ── Step 8: Quick evaluation ──────────────────────────────────
print("\nEvaluating final model (50 games)...")
wins = 0
total = 50

for i in range(total):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    if reward > 0:
        wins += 1

print(f"\nFinal evaluation:")
print(f"  Wins: {wins}/{total} ({wins/total:.1%})")
print(f"  Random baseline was: 40%")
print(f"  Improvement: {(wins/total - 0.40)*100:+.1f}%")
env.close()