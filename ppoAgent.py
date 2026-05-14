# ── Imports ───────────────────────────────────────────────────
# PPO algorithm: Stable-Baselines3 (Raffin et al., 2021)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Go board environment: PettingZoo (Terry et al., 2021)
from pettingzoo.classic import go_v5

# Action masking support for SB3
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Heuristic opponent
from randomOpponent import RandomOpponent

import wandb
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

# ── Configuration ──────────────────────────────────────────────
BOARD_SIZE      = 19
TOTAL_TIMESTEPS = 10_000_000
SAVE_EVERY      = 500_000
SAVE_DIR        = "./models/ppo/"
LOG_DIR         = "./logs/ppo/"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# GO ENVIRONMENT WRAPPER
# ══════════════════════════════════════════════════════════════

class GoEnvWrapper(gym.Env):
    """
    Wraps PettingZoo's two-agent Go into a single-agent Gymnasium env.
    Our agent plays Black; White plays via the RandomOpponent class.

    Action masking is used so the agent can only ever select a legal
    move — illegal moves are never sampled. PettingZoo's action_mask
    in each observation defines exactly which moves are legal.
    """

    def __init__(self, board_size=BOARD_SIZE, komi=7.5):
        super().__init__()
        self.board_size  = board_size
        self.komi        = komi
        self.env         = go_v5.env(board_size=board_size, komi=komi)
        self.action_mask = None
        self.opponent    = RandomOpponent()

        self.action_space = spaces.Discrete(board_size * board_size + 1)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(board_size * board_size * 17,),
            dtype=np.float32
        )

        self.episode_count = 0
        self.win_count     = 0

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        obs, _, _, _, _ = self.env.last()
        self.action_mask = obs['action_mask'].astype(bool)
        return self._process_obs(obs), {}

    def _process_obs(self, obs):
        return obs['observation'].flatten().astype(np.float32)

    def get_action_mask(self):
        return self.action_mask

    def step(self, action):
        obs, _, term, trunc, _ = self.env.last()

        # Action is guaranteed legal — MaskablePPO only samples
        # from actions where action_mask == True
        self.env.step(int(action))

        # White opponent plays a legal move via RandomOpponent class
        if not term and not trunc:
            opp_obs, _, opp_term, opp_trunc, _ = self.env.last()
            if not opp_term and not opp_trunc:
                opp_action = self.opponent.select_action(opp_obs)
                self.env.step(opp_action)

        next_obs, reward, next_term, next_trunc, _ = self.env.last()
        self.action_mask = next_obs['action_mask'].astype(bool)

        if next_term or next_trunc:
            self.episode_count += 1
            if reward > 0:
                self.win_count += 1

        return self._process_obs(next_obs), float(reward), next_term, next_trunc, {}

    def close(self):
        self.env.close()


def get_go_env(vec_env):
    """Safely unwrap to GoEnvWrapper regardless of wrapper depth."""
    env = vec_env.envs[0]
    while not isinstance(env, GoEnvWrapper):
        env = env.env
    return env


# ══════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════

class WandbCallback(BaseCallback):
    """
    - sync_tensorboard=True handles ALL SB3 graphs including
      ep_rew_mean and ep_len_mean from Monitor automatically.
    - Custom win rate logged directly with correct timestep.
    - Checkpoints saved every N steps.
    """

    def __init__(self, save_every, save_dir, verbose=0):
        super().__init__(verbose)
        self.save_every         = save_every
        self.save_dir           = save_dir
        self.last_save          = 0
        self.last_episode_count = 0

    def _on_step(self):
        env = get_go_env(self.training_env)

        if env.episode_count > self.last_episode_count:
            wr = env.win_count / env.episode_count
            wandb.log({
                'custom/win_rate':       wr,
                'custom/total_episodes': env.episode_count,
                'custom/total_wins':     env.win_count,
            }, step=self.num_timesteps)

            print(f"  Game {env.episode_count:>5,} | "
                  f"Step {self.num_timesteps:>8,} | "
                  f"Win rate: {wr:.1%}", flush=True)

            self.last_episode_count = env.episode_count

        if self.num_timesteps - self.last_save >= self.save_every:
            path = os.path.join(self.save_dir, f"ppo_go_{self.num_timesteps}_steps")
            self.model.save(path)
            print(f"  ✓ Checkpoint saved: {path}", flush=True)
            self.last_save = self.num_timesteps

        return True

    def _on_training_end(self):
        wandb.finish()


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 55)
    print(f"  PPO Agent — Go {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"  PPO:   Stable-Baselines3 (MaskablePPO)")
    print(f"  Board: PettingZoo go_v5")
    print(f"  Opponent: RandomOpponent")
    print("=" * 55)

    wandb.tensorboard.patch(root_logdir=LOG_DIR)
    wandb.init(
        project          = "honours-rl-go",
        name             = "ppo-vs-random",
        sync_tensorboard = True,
        config           = {
            "board_size":      BOARD_SIZE,
            "total_timesteps": TOTAL_TIMESTEPS,
            "learning_rate":   3e-4,
            "n_steps":         2048,
            "batch_size":      64,
            "n_epochs":        10,
            "gamma":           0.99,
            "clip_range":      0.2,
            "ent_coef":        0.01,
            "opponent":        "random",
        }
    )

    print("\nSetting up environment...")
    env = GoEnvWrapper(board_size=BOARD_SIZE)
    env = Monitor(env)
    env = ActionMasker(env, lambda e: e.env.get_action_mask())
    print("Environment ready!\n")

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        device="cuda",
    )

    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"Logs: https://wandb.ai\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=WandbCallback(save_every=SAVE_EVERY, save_dir=SAVE_DIR),
        tb_log_name="ppo_run_1"
    )

    final_path = os.path.join(SAVE_DIR, "ppo_go_final")
    model.save(final_path)
    print(f"\n  ✓ Final model saved: {final_path}")

    print("\nRunning final evaluation (50 games)...")
    wins = 0
    for _ in range(50):
        obs, _ = env.reset()
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True,
                                      action_masks=env.env.get_action_mask())
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
        if reward > 0:
            wins += 1

    print(f"\n  Final win rate: {wins/50:.1%}")
    env.close()
