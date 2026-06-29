# ══════════════════════════════════════════════════════════════
# PPO for Go  (new, self-contained implementation)
#
# Base: adi3e08/PPO — a clean minimal PPO. That reference is for
# CONTINUOUS control (Gaussian policy, MLP, DM Control Suite). Here
# the PPO *core* (GAE, clipped surrogate, advantage normalisation,
# KL early-stop, grad clipping) is kept, but everything around it is
# rebuilt for Go:
#
#   - Discrete, ACTION-MASKED Categorical policy (not Gaussian)
#   - CNN trunk over the (N, N, 17) board (not an MLP over a flat vec)
#   - PettingZoo go_v5, agent plays Black vs a heuristic opponent
#   - Correct per-agent reward (reads env.rewards[agent], NOT last())
#   - Rolling-window win rate (not a lifetime cumulative average)
#
# Why those four matter is documented inline — they are exactly the
# bugs/limitations that stopped the previous pipeline from learning.
# ══════════════════════════════════════════════════════════════

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pettingzoo.classic import go_v5

from config_ppo_go import *

# ── Opponent selection ─────────────────────────────────────────
if OPPONENT == "greedy":
    from greedyOpponent import GreedyOpponent as OpponentClass
elif OPPONENT == "aggressive":
    from aggressiveOpponent import AggressiveOpponent as OpponentClass
elif OPPONENT == "defensive":
    from defensiveOpponent import DefensiveOpponent as OpponentClass
elif OPPONENT == "corner":
    from cornerOpponent import CornerOpponent as OpponentClass
elif OPPONENT == "edge":
    from edgeOpponent import EdgeOpponent as OpponentClass
else:
    from randomOpponent import RandomOpponent as OpponentClass

N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1


# ══════════════════════════════════════════════════════════════
# ENVIRONMENT WRAPPER
#
# Single-agent view of two-player Go. Our agent is Black (moves
# first); White is the heuristic opponent.
#
# Reward handling (the key fix vs the old wrapper):
#   The old code read `reward` from env.last() AFTER both players
#   moved. In PettingZoo's AEC API last() returns the reward of the
#   agent whose turn it is — when OUR move ends the game that is the
#   OPPONENT, so the sign was flipped on a subset of terminal states.
#   Here we always read env.rewards["black_0"] explicitly, which is
#   our agent's own reward regardless of whose turn it is next.
# ══════════════════════════════════════════════════════════════

class GoEnv:
    AGENT = "black_0"
    OPP   = "white_0"

    def __init__(self, board_size=BOARD_SIZE, komi=KOMI, seed=SEED):
        self.board_size = board_size
        self.env        = go_v5.env(board_size=board_size, komi=komi)
        self.opponent   = OpponentClass(board_size=board_size)
        self._seed      = seed
        self.action_mask = None

    # -- helpers ---------------------------------------------------
    def _obs_chw(self, obs_dict):
        # go_v5 obs is (N, N, 17) HWC. Return (17, N, N) CHW float32.
        # (The old DQN reshaped HWC-flat straight into CHW, scrambling
        #  the board — done correctly here via transpose.)
        o = obs_dict["observation"].astype(np.float32)
        return np.transpose(o, (2, 0, 1)).copy()

    def _done(self):
        t = self.env.terminations
        r = self.env.truncations
        return (not self.env.agents
                or t.get(self.AGENT, False) or r.get(self.AGENT, False))

    def reset(self):
        self.env.reset(seed=self._seed)
        self._seed += 1  # vary episodes while staying reproducible
        obs, _, _, _, _ = self.env.last()
        self.action_mask = obs["action_mask"].astype(bool)
        return self._obs_chw(obs)

    def step(self, action):
        # our move
        self.env.step(int(action))

        # opponent replies if the game is still going and it's their turn
        if (self.env.agents
                and self.env.agent_selection == self.OPP
                and not (self.env.terminations.get(self.OPP, False)
                         or self.env.truncations.get(self.OPP, False))):
            opp_obs, _, _, _, _ = self.env.last()
            opp_action = self.opponent.select_action(opp_obs)
            self.env.step(opp_action)

        done   = self._done()
        reward = float(self.env.rewards.get(self.AGENT, 0.0))  # OUR reward

        next_obs, _, _, _, _ = self.env.last()
        self.action_mask = next_obs["action_mask"].astype(bool)
        return self._obs_chw(next_obs), reward, done

    def get_action_mask(self):
        return self.action_mask


# ══════════════════════════════════════════════════════════════
# ACTOR-CRITIC NETWORK  (shared CNN trunk, separate heads)
# ══════════════════════════════════════════════════════════════

def _orthogonal(module, gain):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    def __init__(self, board_size, n_channels, n_actions,
                 n_filters, n_layers, hidden_dim):
        super().__init__()
        self.board_size = board_size
        self.n_channels = n_channels

        conv, in_ch = [], n_channels
        for _ in range(n_layers):
            conv += [nn.Conv2d(in_ch, n_filters, 3, padding=1), nn.ReLU()]
            in_ch = n_filters
        conv.append(nn.Flatten())
        self.trunk = nn.Sequential(*conv)

        flat = n_filters * board_size * board_size
        self.shared = nn.Sequential(nn.Linear(flat, hidden_dim), nn.ReLU())
        self.actor  = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        # orthogonal init; small gain on the policy head, std=1 on value
        self.apply(lambda m: _orthogonal(m, gain=np.sqrt(2)))
        _orthogonal(self.actor, gain=0.01)
        _orthogonal(self.critic, gain=1.0)

    def _features(self, obs):
        return self.shared(self.trunk(obs))

    def _masked_dist(self, logits, mask):
        # set illegal-move logits to -inf so they get zero probability
        logits = logits.masked_fill(~mask, -1e8)
        return Categorical(logits=logits)

    def act(self, obs, mask):
        """Sample an action (used during rollout collection)."""
        f = self._features(obs)
        dist  = self._masked_dist(self.actor(f), mask)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(f).squeeze(-1)

    def evaluate(self, obs, mask, actions):
        """Re-evaluate stored actions (used during the PPO update)."""
        f = self._features(obs)
        dist = self._masked_dist(self.actor(f), mask)
        return (dist.log_prob(actions), dist.entropy(),
                self.critic(f).squeeze(-1))

    def value(self, obs):
        return self.critic(self._features(obs)).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised Advantage Estimation. `values` has len T,
    `last_value` bootstraps the step after the rollout."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        gae   = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


def train():
    print("=" * 60)
    print(f"  PPO (new) — Go {BOARD_SIZE}x{BOARD_SIZE}  vs {OPPONENT}")
    print(f"  Discrete masked Categorical policy + CNN trunk")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    import wandb
    # Log config with lowercase keys matching the old runs so they line
    # up in the same Runs-table columns (otherwise this run shows dashes).
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"ppo-go-{BOARD_SIZE}x{BOARD_SIZE}-vs-{OPPONENT}",
        dir     = LOG_DIR,
        config  = {
            "board_size":      BOARD_SIZE,
            "komi":            KOMI,
            "opponent":        OPPONENT,
            "total_timesteps": TOTAL_TIMESTEPS,
            "learning_rate":   LEARNING_RATE,
            "n_steps":         N_STEPS,
            "batch_size":      BATCH_SIZE,
            "n_epochs":        N_EPOCHS,
            "gamma":           GAMMA,
            "gae_lambda":      GAE_LAMBDA,
            "clip_range":      CLIP_RANGE,
            "ent_coef":        ENT_COEF,
            "vf_coef":         VF_COEF,
            "max_grad_norm":   MAX_GRAD_NORM,
            "target_kl":       TARGET_KL,
            "cnn_filters":     CNN_FILTERS,
            "cnn_layers":      CNN_LAYERS,
            "hidden_dim":      HIDDEN_DIM,
            "seed":            SEED,
        },
    )

    # set the opponent difficulty from config (no-op for random — it has
    # no EPSILON knob and is the zero-strategy baseline)
    _OPP_MODULES = {"greedy": "greedyOpponent", "aggressive": "aggressiveOpponent",
                    "defensive": "defensiveOpponent", "corner": "cornerOpponent",
                    "edge": "edgeOpponent"}
    if OPPONENT in _OPP_MODULES:
        import importlib
        _m = importlib.import_module(_OPP_MODULES[OPPONENT])
        _m.EPSILON = OPPONENT_EPSILON
        print(f"  {OPPONENT} epsilon (difficulty): {OPPONENT_EPSILON}")

    env = GoEnv()
    net = ActorCritic(BOARD_SIZE, N_CHANNELS, N_ACTIONS,
                      CNN_FILTERS, CNN_LAYERS, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # rolling-window win rate (NOT a lifetime cumulative average —
    # the old code's cumulative metric could never show a trend)
    win_window = deque(maxlen=WINDOW)
    ep_rew_window = deque(maxlen=WINDOW)
    ep_len_window = deque(maxlen=WINDOW)

    obs = env.reset()
    mask = env.get_action_mask()
    ep_rew, ep_len = 0.0, 0
    global_step, episodes, last_save = 0, 0, 0

    while global_step < TOTAL_TIMESTEPS:
        # ── 1. Collect a rollout of N_STEPS transitions ───────────
        b_obs, b_mask, b_act, b_logp, b_val, b_rew, b_done = [], [], [], [], [], [], []

        for _ in range(N_STEPS):
            obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                action, logp, value = net.act(obs_t, mask_t)

            next_obs, reward, done = env.step(action.item())

            b_obs.append(obs); b_mask.append(mask); b_act.append(action.item())
            b_logp.append(logp.item()); b_val.append(value.item())
            b_rew.append(reward); b_done.append(float(done))

            ep_rew += reward; ep_len += 1; global_step += 1
            obs, mask = next_obs, env.get_action_mask()

            if done:
                episodes += 1
                win_window.append(1.0 if reward > 0 else 0.0)
                ep_rew_window.append(ep_rew); ep_len_window.append(ep_len)
                ep_rew, ep_len = 0.0, 0
                obs = env.reset(); mask = env.get_action_mask()

        # ── 2. GAE + returns ──────────────────────────────────────
        with torch.no_grad():
            last_v = net.value(
                torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            ).item()
        adv, returns = compute_gae(b_rew, b_val, b_done, last_v, GAMMA, GAE_LAMBDA)

        # to tensors
        obs_t    = torch.as_tensor(np.array(b_obs), dtype=torch.float32, device=device)
        mask_t   = torch.as_tensor(np.array(b_mask), dtype=torch.bool, device=device)
        act_t    = torch.as_tensor(b_act, dtype=torch.long, device=device)
        oldlogp_t= torch.as_tensor(b_logp, dtype=torch.float32, device=device)
        adv_t    = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t    = torch.as_tensor(returns, dtype=torch.float32, device=device)

        # ── 3. PPO update ─────────────────────────────────────────
        idx = np.arange(N_STEPS)
        approx_kl = 0.0
        clipfracs, pg_losses, v_losses, ent_losses = [], [], [], []
        stop = False
        for epoch in range(N_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, N_STEPS, BATCH_SIZE):
                mb = idx[start:start + BATCH_SIZE]
                new_logp, entropy, value = net.evaluate(obs_t[mb], mask_t[mb], act_t[mb])

                # advantage normalisation (batch level)
                mb_adv = adv_t[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ratio = torch.exp(new_logp - oldlogp_t[mb])
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_logp - oldlogp_t[mb])).mean().item()
                    clipfracs.append(((ratio - 1).abs() > CLIP_RANGE).float().mean().item())

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss  = F.mse_loss(value, ret_t[mb])
                ent     = entropy.mean()

                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * ent

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                pg_losses.append(pg_loss.item()); v_losses.append(v_loss.item())
                ent_losses.append(ent.item())

            # KL early-stop (per the reference implementation)
            if approx_kl > 1.5 * TARGET_KL:
                stop = True
                break
        # explained variance of the value function
        y_pred = np.array(b_val); y_true = returns
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # ── 4. Logging ────────────────────────────────────────────
        win_rate = float(np.mean(win_window)) if win_window else 0.0
        metrics = {
            "custom/win_rate":        win_rate,
            "custom/total_episodes":  episodes,
            "rollout/ep_rew_mean":    float(np.mean(ep_rew_window)) if ep_rew_window else 0.0,
            "rollout/ep_len_mean":    float(np.mean(ep_len_window)) if ep_len_window else 0.0,
            "train/policy_loss":      float(np.mean(pg_losses)),
            "train/value_loss":       float(np.mean(v_losses)),
            "train/entropy":          float(np.mean(ent_losses)),
            "train/approx_kl":        approx_kl,
            "train/clip_fraction":    float(np.mean(clipfracs)),
            "train/explained_var":    explained_var,
            "train/early_stopped":    float(stop),
        }
        wandb.log(metrics, step=global_step)

        print(f"  Step {global_step:>9,} | Games {episodes:>6,} | "
              f"WinRate(last{WINDOW}) {win_rate:5.1%} | "
              f"epLen {metrics['rollout/ep_len_mean']:5.1f} | "
              f"KL {approx_kl:.4f}", flush=True)

        # ── 5. Checkpoint ─────────────────────────────────────────
        if global_step - last_save >= SAVE_EVERY:
            path = os.path.join(SAVE_DIR, f"ppo_go_{global_step}.pt")
            torch.save(net.state_dict(), path)
            print(f"  ✓ saved {path}", flush=True)
            last_save = global_step

    torch.save(net.state_dict(), os.path.join(SAVE_DIR, "ppo_go_final.pt"))
    print("\n  ✓ Final model saved.")
    wandb.finish()


if __name__ == "__main__":
    train()
