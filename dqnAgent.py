# ── DQN Agent for Go ──────────────────────────────────────────
#
# Base: PyTorch DQN tutorial (Mnih et al., 2015)
#
# Improvements over standard DQN:
#   1. CNN Q-network — spatial board understanding (17, 19, 19)
#   2. Action masking — only legal moves selected/evaluated
#   3. Huber loss — prevents loss explosion
#   4. Double DQN — reduces Q-value overestimation
#   5. N-step returns — better credit assignment for sparse Go rewards

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
# CNN Q-NETWORK
# ══════════════════════════════════════════════════════════════

class QNetwork(nn.Module):

    def __init__(self, board_size, n_channels, n_actions,
                 n_filters, n_layers, head_arch):
        super().__init__()
        self.board_size = board_size
        self.n_channels = n_channels

        cnn = []
        in_ch = n_channels
        for _ in range(n_layers):
            cnn.append(nn.Conv2d(in_ch, n_filters, kernel_size=3, padding=1))
            cnn.append(nn.ReLU())
            in_ch = n_filters
        cnn.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn)

        cnn_out = n_filters * board_size * board_size
        mlp = []
        in_dim = cnn_out
        for hidden in head_arch:
            mlp.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        mlp.append(nn.Linear(in_dim, n_actions))
        self.mlp = nn.Sequential(*mlp)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, action_mask=None):
        batch = x.shape[0]
        x = x.view(batch, self.n_channels, self.board_size, self.board_size)
        q_values = self.mlp(self.cnn(x))
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

        self.obs               = np.zeros((capacity, obs_dim),   dtype=np.float32)
        self.next_obs          = np.zeros((capacity, obs_dim),   dtype=np.float32)
        self.actions           = np.zeros(capacity,              dtype=np.int64)
        self.rewards           = np.zeros(capacity,              dtype=np.float32)
        self.dones             = np.zeros(capacity,              dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, n_actions), dtype=bool)

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
# N-STEP BUFFER
#
# Accumulates N_STEP transitions before storing in replay buffer.
# Computes the discounted n-step return:
#   R = r_0 + gamma*r_1 + ... + gamma^(n-1)*r_{n-1}
#
# The n-step transition stored is:
#   (s_0, a_0, R, s_n, done, next_mask_n)
#
# Target becomes: R + gamma^n * max Q(s_n, a)
# This directly connects s_0 to rewards n steps ahead instead
# of just 1 step — critical for 300-move sparse-reward Go.
# ══════════════════════════════════════════════════════════════

class NStepBuffer:

    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma  = gamma
        self.buffer = deque(maxlen=n_step)

    def add(self, obs, action, reward, next_obs, done, next_mask):
        self.buffer.append((obs, action, reward, next_obs, done, next_mask))

    def is_ready(self):
        return len(self.buffer) == self.n_step

    def get(self):
        """
        Returns the n-step transition starting from the oldest entry.
        Stops accumulating rewards if a done=True is encountered.
        """
        first_obs, first_action = self.buffer[0][0], self.buffer[0][1]

        n_step_return = 0.0
        episode_done  = False
        last_next_obs  = self.buffer[-1][3]
        last_done      = self.buffer[-1][4]
        last_next_mask = self.buffer[-1][5]

        for i, (_, _, r, next_o, d, next_m) in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * r
            if d:
                # Episode ended — bootstrap from zero beyond this point
                episode_done  = True
                last_next_obs  = next_o
                last_done      = True
                last_next_mask = next_m
                break

        return (first_obs, first_action, n_step_return,
                last_next_obs, float(episode_done or last_done),
                last_next_mask)

    def reset(self):
        self.buffer.clear()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def select_action(obs, action_mask, q_net, epsilon, device):
    """Epsilon-greedy with action masking — random only from legal moves."""
    if random.random() < epsilon:
        return int(np.random.choice(np.where(action_mask)[0]))
    else:
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(device)
            return q_net(obs_t, mask_t).argmax(dim=1).item()


def optimize_model(q_net, target_net, replay_buffer, optimizer, device):
    """
    One gradient update step.

    Improvements over standard DQN:
    - Huber loss (SmoothL1) — prevents loss explosion
    - Double DQN — online net selects action, target net evaluates
      it. Reduces Q-value overestimation.
    - N-step target: R + gamma^n * Q(s_n, a*)
    """
    if len(replay_buffer) < BATCH_SIZE:
        return None, None

    obs, actions, rewards, next_obs, dones, next_masks = replay_buffer.sample(BATCH_SIZE)

    current_q_all = q_net(obs)
    current_q     = current_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if DOUBLE_DQN:
            # Double DQN: online net selects, target net evaluates
            next_actions = q_net(next_obs, next_masks).argmax(dim=1, keepdim=True)
            next_q       = target_net(next_obs, next_masks)
            max_next_q   = next_q.gather(1, next_actions).squeeze(1)
        else:
            next_q     = target_net(next_obs, next_masks)
            max_next_q = next_q.max(dim=1)[0]

        # N-step target: discount bootstrap by gamma^n
        target_q = rewards + (GAMMA ** N_STEP) * (1.0 - dones) * max_next_q

    loss = nn.SmoothL1Loss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()

    mean_q = current_q_all.max(dim=1)[0].mean().item()
    return loss.item(), mean_q


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train():
    print("=" * 55)
    print(f"  DQN Agent — Go {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"  Policy: CNN ({CNN_LAYERS} layers, {CNN_FILTERS} filters)")
    print(f"  Improvements: Double DQN, {N_STEP}-step returns")
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
            "n_step":               N_STEP,
            "double_dqn":           DOUBLE_DQN,
            "train_freq":           TRAIN_FREQ,
            "target_update_freq":   TARGET_UPDATE_FREQ,
            "epsilon_start":        EPSILON_START,
            "epsilon_end":          EPSILON_END,
            "epsilon_decay_steps":  EPSILON_DECAY_STEPS,
            "cnn_filters":          CNN_FILTERS,
            "cnn_layers":           CNN_LAYERS,
            "head_arch":            HEAD_ARCH,
            "opponent":             "random",
        }
    )

    env        = GoEnvWrapper(board_size=BOARD_SIZE)
    q_net      = QNetwork(BOARD_SIZE, N_CHANNELS, N_ACTIONS,
                          CNN_FILTERS, CNN_LAYERS, HEAD_ARCH).to(device)
    target_net = QNetwork(BOARD_SIZE, N_CHANNELS, N_ACTIONS,
                          CNN_FILTERS, CNN_LAYERS, HEAD_ARCH).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer     = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE, OBS_DIM, N_ACTIONS, device)
    n_step_buf    = NStepBuffer(N_STEP, GAMMA)

    epsilon  = EPSILON_START
    eps_step = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS

    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    ep_reward  = 0.0
    ep_length  = 0

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

        # Add to n-step buffer
        n_step_buf.add(obs, action, reward, next_obs,
                       done, env.get_action_mask())

        # Once buffer is full, store n-step transition in replay
        if n_step_buf.is_ready():
            replay_buffer.add(*n_step_buf.get())

        # On episode end, flush remaining transitions
        if done:
            # Drain remaining transitions from n-step buffer
            while len(n_step_buf.buffer) > 0:
                replay_buffer.add(*n_step_buf.get())
                n_step_buf.buffer.popleft()
            n_step_buf.reset()

            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            ep_reward = 0.0
            ep_length = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

        ep_reward += reward
        ep_length += 1
        global_step += 1
        epsilon = max(EPSILON_END, epsilon - eps_step)

        # ── Learn ─────────────────────────────────────────────
        loss, mean_q = None, None
        if global_step >= LEARNING_STARTS and global_step % TRAIN_FREQ == 0:
            loss, mean_q = optimize_model(q_net, target_net,
                                          replay_buffer, optimizer, device)

        # ── Sync target network ───────────────────────────────
        if global_step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ── Logging ───────────────────────────────────────────
        if env.episode_count > last_ep_count:
            wr = env.win_count / env.episode_count

            log_dict = {
                'custom/win_rate':       wr,
                'custom/total_episodes': env.episode_count,
                'custom/total_wins':     env.win_count,
                'rollout/ep_rew_mean':   np.mean(ep_rewards) if ep_rewards else 0.0,
                'rollout/ep_len_mean':   np.mean(ep_lengths) if ep_lengths else 0.0,
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
