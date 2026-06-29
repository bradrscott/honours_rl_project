# ══════════════════════════════════════════════════════════════
# FEUDAL NETWORK AGENT for Go  (FuN — Vezhnevets et al. 2017)
# Architecture adapted from lweitkamp/feudalnets-pytorch (MIT).
#
# This training harness is kept DELIBERATELY PARALLEL to ppo_go.py so
# the Feudal-vs-flat-PPO comparison (RQ3) is fair. Identical to PPO:
#   - the GoEnv wrapper (PettingZoo go_v5, agent = Black vs a heuristic)
#   - correct per-agent reward  (reads env.rewards["black_0"], NOT last())
#   - the same six strict opponents, selected from config
#   - the same CNN feature extractor (in feudalNetwork.Perception)
#   - rolling-window win rate (NOT a lifetime cumulative average)
# The ONLY difference from PPO is the architecture: a Manager-Worker
# hierarchy (this file drives the FuN rollout + feudal_loss) instead of
# a flat actor-critic + PPO update.
# ══════════════════════════════════════════════════════════════

import os
import random
from collections import deque

import numpy as np
import torch
from pettingzoo.classic import go_v5

from feudalNetwork import FeudalNetwork, Storage, feudal_loss
from config_feudal import *

# ── Opponent selection (identical to ppo_go.py) ───────────────────
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


# ══════════════════════════════════════════════════════════════
# ENVIRONMENT WRAPPER  (identical logic to ppo_go.GoEnv)
#
# Single-agent view of two-player Go. Our agent is Black (moves first);
# White is the heuristic opponent. We always read env.rewards["black_0"]
# explicitly — last() returns the reward of whoever moves NEXT, which is
# the OPPONENT when our move ends the game (sign-flipped on a subset of
# terminal states — the bug the old wrapper had).
# ══════════════════════════════════════════════════════════════

class GoEnv:
    AGENT = "black_0"
    OPP   = "white_0"

    def __init__(self, board_size=BOARD_SIZE, komi=KOMI, seed=SEED):
        self.board_size  = board_size
        self.env         = go_v5.env(board_size=board_size, komi=komi)
        self.opponent    = (OpponentClass(board_size=board_size)
                            if OPPONENT != "random" else OpponentClass())
        self._seed       = seed
        self.action_mask = None

    def _obs_chw(self, obs_dict):
        # go_v5 obs is (N, N, 17) HWC -> (17, N, N) CHW float32
        o = obs_dict["observation"].astype(np.float32)
        return np.transpose(o, (2, 0, 1)).copy()

    def _done(self):
        t, r = self.env.terminations, self.env.truncations
        return (not self.env.agents
                or t.get(self.AGENT, False) or r.get(self.AGENT, False))

    def reset(self):
        self.env.reset(seed=self._seed)
        self._seed += 1
        obs, _, _, _, _ = self.env.last()
        self.action_mask = obs["action_mask"].astype(bool)
        return self._obs_chw(obs)

    def step(self, action):
        # our move
        self.env.step(int(action))

        # opponent replies if the game continues and it's their turn
        if (self.env.agents
                and self.env.agent_selection == self.OPP
                and not (self.env.terminations.get(self.OPP, False)
                         or self.env.truncations.get(self.OPP, False))):
            opp_obs, _, _, _, _ = self.env.last()
            self.env.step(self.opponent.select_action(opp_obs))

        done   = self._done()
        reward = float(self.env.rewards.get(self.AGENT, 0.0))   # OUR reward
        next_obs, _, _, _, _ = self.env.last()
        self.action_mask = next_obs["action_mask"].astype(bool)
        return self._obs_chw(next_obs), reward, done

    def get_action_mask(self):
        return self.action_mask

    def close(self):
        self.env.close()


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def obs_to_tensor(obs_chw, device):
    """(17, N, N) CHW numpy -> (1, 17, N, N) float tensor."""
    return torch.as_tensor(obs_chw, dtype=torch.float32, device=device).unsqueeze(0)


def train():
    print("=" * 60)
    print(f"  FEUDAL (FuN) — Go {BOARD_SIZE}x{BOARD_SIZE}  vs {OPPONENT}")
    print(f"  Manager-Worker hierarchy + CNN perception (matches PPO trunk)")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # strict opponent difficulty from config (no-op for random)
    _OPP_MODULES = {"greedy": "greedyOpponent", "aggressive": "aggressiveOpponent",
                    "defensive": "defensiveOpponent", "corner": "cornerOpponent",
                    "edge": "edgeOpponent"}
    if OPPONENT in _OPP_MODULES:
        import importlib
        importlib.import_module(_OPP_MODULES[OPPONENT]).EPSILON = OPPONENT_EPSILON
        print(f"  {OPPONENT} epsilon (difficulty): {OPPONENT_EPSILON}")

    import wandb
    wandb.init(
        project = WANDB_PROJECT,
        name    = f"feudal-{BOARD_SIZE}x{BOARD_SIZE}-vs-{OPPONENT}",
        dir     = LOG_DIR,
        config  = {
            "agent":            "feudal",
            "board_size":       BOARD_SIZE,
            "komi":             KOMI,
            "opponent":         OPPONENT,
            "opponent_epsilon": OPPONENT_EPSILON,
            "total_timesteps":  TOTAL_TIMESTEPS,
            "learning_rate":    LEARNING_RATE,
            "num_steps":        NUM_STEPS,
            "hidden_dim_m":     HIDDEN_DIM_M,
            "hidden_dim_w":     HIDDEN_DIM_W,
            "time_horizon":     TIME_HORIZON,
            "dilation":         DILATION,
            "eps":              EPS,
            "alpha":            ALPHA,
            "gamma_m":          GAMMA_M,
            "gamma_w":          GAMMA_W,
            "entropy_coef":     ENTROPY_COEF,
            "grad_clip":        GRAD_CLIP,
            "cnn_filters":      CNN_FILTERS,
            "cnn_layers":       CNN_LAYERS,
            "seed":             SEED,
        },
    )

    env   = GoEnv()
    model = FeudalNetwork(
        board_size         = BOARD_SIZE,
        n_channels         = N_CHANNELS,
        n_actions          = N_ACTIONS,
        hidden_dim_manager = HIDDEN_DIM_M,
        hidden_dim_worker  = HIDDEN_DIM_W,
        time_horizon       = TIME_HORIZON,
        dilation           = DILATION,
        eps                = EPS,
        num_workers        = 1,
        device             = str(device),
        n_filters          = CNN_FILTERS,
        n_layers           = CNN_LAYERS,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # rolling-window metrics (same as ppo_go.py)
    win_window    = deque(maxlen=WINDOW)
    ep_rew_window = deque(maxlen=WINDOW)
    ep_len_window = deque(maxlen=WINDOW)

    obs  = env.reset()
    mask = env.get_action_mask()
    goals, states, masks = model.init_obj()
    storage = Storage(NUM_STEPS)

    ep_rew, ep_len = 0.0, 0
    global_step, episodes, last_save = 0, 0, 0
    done = False

    while global_step < TOTAL_TIMESTEPS:
        # ── 1. Collect a rollout of NUM_STEPS transitions ─────────
        for _ in range(NUM_STEPS):
            obs_t         = obs_to_tensor(obs, device)
            mask_t        = torch.tensor([[0.0 if done else 1.0]], device=device)
            action_mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            dist, goals, states, value_m, value_w = model(
                obs_t, goals, states, mask_t, action_mask_t)
            model.repackage_hidden()
            goals  = [g.detach() for g in goals]
            states = [s.detach() for s in states]

            action  = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()

            next_obs, reward, done = env.step(action.item())

            masks.append(mask_t)
            if len(masks) > (2 * TIME_HORIZON + 1):
                masks.pop(0)

            r_i        = model.intrinsic_reward(states, goals, masks)
            s_goal_cos = model.state_goal_cosine(states, goals, masks)

            storage.add({
                'r':          torch.tensor([[reward]], device=device),
                'r_i':        r_i,
                'v_m':        value_m,
                'v_w':        value_w,
                'logp':       log_prob.unsqueeze(-1),
                'entropy':    entropy.unsqueeze(-1),
                's_goal_cos': s_goal_cos,
                'm':          mask_t,
            })

            ep_rew += reward; ep_len += 1; global_step += 1
            obs  = next_obs
            mask = env.get_action_mask()

            if done:
                episodes += 1
                win_window.append(1.0 if reward > 0 else 0.0)
                ep_rew_window.append(ep_rew); ep_len_window.append(ep_len)
                ep_rew, ep_len = 0.0, 0
                obs  = env.reset()
                mask = env.get_action_mask()

        # ── 2. Bootstrap values for the final step ────────────────
        obs_t  = obs_to_tensor(obs, device)
        mask_t = torch.tensor([[0.0 if done else 1.0]], device=device)
        next_v_m, next_v_w = model.get_next_values(obs_t, goals, states, mask_t)

        # ── 3. FuN loss + update ──────────────────────────────────
        loss, metrics = feudal_loss(
            storage, next_v_m, next_v_w,
            GAMMA_M, GAMMA_W, ALPHA, ENTROPY_COEF, NUM_STEPS)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.repackage_hidden()
        storage.reset()

        # ── 4. Logging (same metric names as ppo_go.py) ───────────
        win_rate = float(np.mean(win_window)) if win_window else 0.0
        wandb.log({
            "custom/win_rate":       win_rate,
            "custom/total_episodes": episodes,
            "rollout/ep_rew_mean":   float(np.mean(ep_rew_window)) if ep_rew_window else 0.0,
            "rollout/ep_len_mean":   float(np.mean(ep_len_window)) if ep_len_window else 0.0,
            **metrics,
        }, step=global_step)

        print(f"  Step {global_step:>9,} | Games {episodes:>6,} | "
              f"WinRate(last{WINDOW}) {win_rate:5.1%} | "
              f"epLen {float(np.mean(ep_len_window)) if ep_len_window else 0.0:5.1f} | "
              f"loss {metrics['loss/total']:.3f}", flush=True)

        # ── 5. Checkpoint ─────────────────────────────────────────
        if global_step - last_save >= SAVE_EVERY:
            path = os.path.join(SAVE_DIR, f"feudal_go_{global_step}.pt")
            torch.save(model.state_dict(), path)
            print(f"  ✓ saved {path}", flush=True)
            last_save = global_step

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "feudal_go_final.pt"))
    print("\n  ✓ Final model saved.")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    train()
