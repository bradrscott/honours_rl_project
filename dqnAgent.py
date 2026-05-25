# ── DQN Agent for Go ──────────────────────────────────────────
#
# Base implementation: PyTorch official DQN tutorial
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#   (Mnih et al., 2015 — "Human-level control through deep reinforcement learning")
#
# Adaptations for Go:
#   - MLP Q-network (flat 6137-dim board observation)
#   - Action masking: illegal moves set to -inf before argmax
#   - Replay buffer stores next_action_mask for masked target Q-values
#   - Epsilon-greedy explores only legal moves
#   - WandB logging matching PPO/Feudal style

from pettingzoo.classic import go_v5
from randomOpponent import RandomOpponent
from config_dqn import *

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import os
from collections import deque

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Q-NETWORK
# ══════════════════════════════════════════════════════════════

class QNetwork(nn.Module):

    def __init__(self, input_dim, n_actions, net_arch):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden in net_arch:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, action_mask=None):
        q_values = self.net(x)
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, float('-inf'))
        return q_values


# ══════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ══════════════════════════════════════════════════════════════

class ReplayBuffer:

    def __init__(self, capacity, obs_dim, n_actions, device):
        self.capacity  = capacity
        self.device    = device
        self.pos       = 0
        self.size      = 0

        self.obs               = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.next_obs          = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions           = np.zeros(capacity,               dtype=np.int64)
        self.rewards           = np.zeros(capacity,               dtype=np.float32)
        self.dones             = np.zeros(capacity,               dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, n_actions),  dtype=bool)

    def add(self, obs, action, reward, next_obs, done, next_action_mask):
        self.obs[self.pos]               = obs
        self.actions[self.pos]           = action
        self.rewards[self.pos]           = reward
        self.next_obs[self.pos]          = next_obs
        self.dones[self.pos]             = done
        self.next_action_masks[self.pos] = next_action_mask
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(self.device),
            torch.LongTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_obs[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
            torch.BoolTensor(self.next_action_masks[idx]).to(self.device),
        )

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════════
# GO ENVIRONMENT WRAPPER
# ══════════════════════════════════════════════════════════════

class GoEnvWrapper(gym.Env):

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

        self.env.step(int(action))

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


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def select_action(obs, action_mask, q_net, epsilon, device):
    """Epsilon-greedy with action masking — random only from legal moves."""
    if random.random() < epsilon:
        legal_moves = np.where(action_mask)[0]
        return int(np.random.choice(legal_moves))
    else:
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(device)
            return q_net(obs_t, mask_t).argmax(dim=1).item()


def optimize_model(q_net, target_net, replay_buffer, optimizer, device):
    """
    One gradient update step.
    Target: r + gamma * max_a Q_target(s', a)  [masked to legal actions]
    Loss:   MSE(Q(s,a), target)
    Returns loss value and mean Q-value for logging.
    """
    if len(replay_buffer) < BATCH_SIZE:
        return None, None

    obs, actions, rewards, next_obs, dones, next_masks = replay_buffer.sample(BATCH_SIZE)

    # Current Q-values for taken actions
    current_q_all = q_net(obs)
    current_q     = current_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q-values — mask illegal next-state actions
    with torch.no_grad():
        next_q     = target_net(next_obs, next_masks)
        max_next_q = next_q.max(dim=1)[0]
        target_q   = rewards + GAMMA * (1.0 - dones) * max_next_q

    loss = nn.MSELoss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()

    # Mean Q-value across batch (monitors overestimation)
    mean_q = current_q_all.max(dim=1)[0].mean().item()

    return loss.item(), mean_q


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train():
    print("=" * 55)
    print(f"  DQN Agent — Go {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"  Base: PyTorch DQN Tutorial (Mnih et al., 2015)")
    print(f"  Board: PettingZoo go_v5")
    print(f"  Opponent: RandomOpponent")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    wandb.init(
        project = "honours-rl-go",
        name    = "dqn-vs-random",
        config  = {
            "board_size":           BOARD_SIZE,
            "total_timesteps":      TOTAL_TIMESTEPS,
            "learning_rate":        LEARNING_RATE,
            "buffer_size":          BUFFER_SIZE,
            "learning_starts":      LEARNING_STARTS,
            "batch_size":           BATCH_SIZE,
            "gamma":                GAMMA,
            "train_freq":           TRAIN_FREQ,
            "target_update_freq":   TARGET_UPDATE_FREQ,
            "epsilon_start":        EPSILON_START,
            "epsilon_end":          EPSILON_END,
            "epsilon_decay_steps":  EPSILON_DECAY_STEPS,
            "net_arch":             NET_ARCH,
            "opponent":             "random",
        }
    )

    env           = GoEnvWrapper(board_size=BOARD_SIZE)
    q_net         = QNetwork(OBS_DIM, N_ACTIONS, NET_ARCH).to(device)
    target_net    = QNetwork(OBS_DIM, N_ACTIONS, NET_ARCH).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer     = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE, OBS_DIM, N_ACTIONS, device)

    # Epsilon linear decay
    epsilon  = EPSILON_START
    eps_step = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS

    # Rolling windows for ep_rew_mean and ep_len_mean (last 100 episodes)
    ep_rewards  = deque(maxlen=100)
    ep_lengths  = deque(maxlen=100)
    ep_reward   = 0.0
    ep_length   = 0

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"Logs: https://wandb.ai\n")

    obs, _        = env.reset()
    global_step   = 0
    last_save     = 0
    last_ep_count = 0

    while global_step < TOTAL_TIMESTEPS:

        action_mask = env.get_action_mask()
        action      = select_action(obs, action_mask, q_net, epsilon, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add(
            obs, action, reward, next_obs,
            float(done), env.get_action_mask()
        )

        # Track episode stats
        ep_reward += reward
        ep_length += 1

        obs = next_obs
        if done:
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            ep_reward = 0.0
            ep_length = 0
            obs, _ = env.reset()

        global_step += 1
        epsilon = max(EPSILON_END, epsilon - eps_step)

        # ── Learn ─────────────────────────────────────────────
        loss, mean_q = None, None
        if global_step >= LEARNING_STARTS and global_step % TRAIN_FREQ == 0:
            loss, mean_q = optimize_model(q_net, target_net, replay_buffer, optimizer, device)

        # ── Sync target network ───────────────────────────────
        if global_step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ── Logging ───────────────────────────────────────────
        if env.episode_count > last_ep_count:
            wr = env.win_count / env.episode_count

            log_dict = {
                # Custom (same as PPO/Feudal)
                'custom/win_rate':       wr,
                'custom/total_episodes': env.episode_count,
                'custom/total_wins':     env.win_count,
                # Rollout (equivalent to SB3 rollout graphs)
                'rollout/ep_rew_mean':   np.mean(ep_rewards) if ep_rewards else 0.0,
                'rollout/ep_len_mean':   np.mean(ep_lengths) if ep_lengths else 0.0,
                # Train
                'train/epsilon':         epsilon,
                'train/learning_rate':   LEARNING_RATE,
            }

            if loss is not None:
                log_dict['train/loss']          = loss
                log_dict['train/q_values_mean'] = mean_q

            wandb.log(log_dict, step=global_step)

            print(f"  Game {env.episode_count:>5,} | "
                  f"Step {global_step:>8,} | "
                  f"Win rate: {wr:.1%} | "
                  f"Epsilon: {epsilon:.3f}", flush=True)

            last_ep_count = env.episode_count

        # ── Checkpoint ────────────────────────────────────────
        if global_step - last_save >= SAVE_EVERY:
            path = os.path.join(SAVE_DIR, f"dqn_go_{global_step}_steps.pt")
            torch.save(q_net.state_dict(), path)
            print(f"  ✓ Checkpoint saved: {path}")
            last_save = global_step

    # ── Final save ────────────────────────────────────────────
    final_path = os.path.join(SAVE_DIR, "dqn_go_final.pt")
    torch.save(q_net.state_dict(), final_path)
    print(f"\n  ✓ Final model saved: {final_path}")

    # ── Final evaluation ──────────────────────────────────────
    print("\nRunning final evaluation (50 games)...")
    q_net.eval()
    wins = 0
    for _ in range(50):
        obs, _ = env.reset()
        done   = False
        while not done:
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mask_t = torch.BoolTensor(env.get_action_mask()).unsqueeze(0).to(device)
                action = q_net(obs_t, mask_t).argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward > 0:
            wins += 1

    print(f"\n  Final win rate: {wins/50:.1%}")
    wandb.finish()
    env.close()


if __name__ == '__main__':
    train()
