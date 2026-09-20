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
import sys
import random
from collections import deque

# Ensure the project ROOT (parent of feudal/) is importable, so this runs both
# as `python -m feudal.feudalAgent` and `python feudal/feudalAgent.py` from root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pettingzoo.classic import go_v5

from feudal.feudalNetwork import FeudalNetwork, Storage, feudal_loss
from feudal.config_feudal import *

# Opponent selection is now a RUNTIME factory (opponents.py) — identical to
# ppo_go.py. Same classes, same EPSILON mechanism, but Phase 2 can swap the
# opponent mid-training (GoEnv.set_opponent).
from opponents import make_opponent
from phase2 import ShiftManager, RecoveryTracker, GameLog


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
        self.opponent    = make_opponent(OPPONENT, board_size, OPPONENT_EPSILON)
        self.opponent_name = OPPONENT
        self._seed       = seed
        self.action_mask = None
        self.move_count  = 0

    def set_opponent(self, name):
        """Phase 2: swap the opponent mid-training (call at a game boundary)."""
        self.opponent = make_opponent(name, self.board_size, OPPONENT_EPSILON)
        self.opponent_name = name

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
        self.move_count = 0
        obs, _, _, _, _ = self.env.last()
        self.action_mask = obs["action_mask"].astype(bool)
        return self._obs_chw(obs)

    def step(self, action):
        # our move
        self.env.step(int(action)); self.move_count += 1

        # opponent replies if the game continues and it's their turn
        if (self.env.agents
                and self.env.agent_selection == self.OPP
                and not (self.env.terminations.get(self.OPP, False)
                         or self.env.truncations.get(self.OPP, False))):
            opp_obs, _, _, _, _ = self.env.last()
            self.env.step(self.opponent.select_action(opp_obs)); self.move_count += 1

        done = self._done()

        # Move-cap safeguard (see ppo_go.GoEnv): force passes at MAX_MOVES so
        # go_v5 area-scores the board. Our move is already applied + recorded.
        if not done and self.move_count >= MAX_MOVES:
            self._force_finish()
            done = True

        reward = float(self.env.rewards.get(self.AGENT, 0.0))   # OUR reward
        next_obs, _, _, _, _ = self.env.last()
        self.action_mask = next_obs["action_mask"].astype(bool)
        return self._obs_chw(next_obs), reward, done

    def _force_finish(self):
        """Force consecutive passes so go_v5 terminates and area-scores."""
        pass_action = self.board_size * self.board_size
        guard = 0
        while self.env.agents and not self._done() and guard < 4:
            cur = self.env.agent_selection
            if (self.env.terminations.get(cur, False)
                    or self.env.truncations.get(cur, False)):
                self.env.step(None)
            else:
                self.env.step(pass_action)
            guard += 1

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

    # Opponent difficulty is applied inside make_opponent (same module-level
    # EPSILON mechanism as before — no-op for random, the zero-strategy bot).
    print(f"  opponent epsilon (difficulty): {OPPONENT_EPSILON}")

    # wandb ON by default → UCT/existing runs are unchanged. Set USE_WANDB=0 to
    # skip it (e.g. Lengau, which has no internet); or USE_WANDB=1 with
    # WANDB_MODE=offline to log locally there and `wandb sync` later.
    USE_WANDB = os.environ.get("USE_WANDB", "1") == "1"
    if USE_WANDB:
        import wandb
    else:
        class _NoWandb:                       # no-op shim
            def init(self, *a, **k): pass
            def log(self, *a, **k): pass
            def finish(self, *a, **k): pass
        wandb = _NoWandb()
    run_name = f"feudal-{BOARD_SIZE}x{BOARD_SIZE}-vs-{OPPONENT}"
    if RUN_TAG:
        run_name += f"-{RUN_TAG}"
    wandb.init(
        project = WANDB_PROJECT,
        name    = run_name,
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
            "resume_from":      RESUME_FROM,
            "shift_schedule":   SHIFT_SCHEDULE,
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

    # ── PHASE 2: resume from a Phase-1 checkpoint + shift machinery ───
    # All inert when RESUME_FROM/SHIFT_SCHEDULE are unset (Phase-1 mode).
    # Identical logic to ppo_go.py so both agents are measured the same way.
    if RESUME_FROM:
        model.load_state_dict(torch.load(RESUME_FROM, map_location=device))
        print(f"  resumed weights from {RESUME_FROM}")
        # optimizer restarts fresh — identical treatment for both agents
    shifter = ShiftManager(SHIFT_SCHEDULE)
    tracker = RecoveryTracker(W_RECOVERY)
    gamelog = GameLog(SAVE_DIR) if shifter.active else None
    if shifter.active:
        print(f"  Phase-2 shift schedule: {shifter.schedule}")

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
        # Sever goals from the PREVIOUS rollout's (already-freed) graph ONCE
        # here, at the rollout boundary. Do NOT detach goals every step: the
        # Manager is trained by the transition policy gradient, which flows
        # through the state-goal cosine INTO the goal. Detaching the goal each
        # step (the old code) zeroed that gradient, so the Manager never
        # learned to point goals usefully (manager/cosines stayed ~0) and the
        # whole hierarchy was dead. States never carry gradient (Manager
        # detaches them), so they are safe to leave as-is.
        goals = [g.detach() for g in goals]
        for _ in range(NUM_STEPS):
            obs_t         = obs_to_tensor(obs, device)
            mask_t        = torch.tensor([[0.0 if done else 1.0]], device=device)
            action_mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            dist, goals, states, value_m, value_w = model(
                obs_t, goals, states, mask_t, action_mask_t)
            # NOTE: do NOT repackage_hidden() here. Detaching the LSTM hidden
            # state every step limited BPTT to a single step, so the Manager's
            # dilated LSTM (whose whole job is temporal memory over c steps)
            # could not learn any temporal dependency. Hidden state is now
            # detached ONCE per rollout at the boundary (after the update, and
            # via the goals-detach above), so gradient flows through the whole
            # rollout as truncated BPTT — the way FuN is meant to train.

            action  = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()

            next_obs, reward, done = env.step(action.item())

            # Maintain the mask window with the SAME discipline as the goals/
            # states windows in FeudalNetwork.forward (pop-then-append), so all
            # three settle at the SAME length and index c refers to the SAME
            # step in every list. (Previously masks used append-then-pop and
            # settled one shorter than goals/states, misaligning the episode-
            # boundary masking in intrinsic_reward / state_goal_cosine.)
            if len(masks) > (2 * TIME_HORIZON + 1):
                masks.pop(0)
            masks.append(mask_t)

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
                # nonterminal flag for THIS step (0 if this step ended the game).
                # This is NOT the same as mask_t (which is 0 on the step AFTER a
                # terminal, for hidden-state reset). GAE must gate its bootstrap
                # with the terminal-step flag, or it leaks the next episode's
                # value across the boundary — the bug that made GAE thrash.
                'nt':         torch.tensor([[0.0 if done else 1.0]], device=device),
            })

            ep_rew += reward; ep_len += 1; global_step += 1
            obs  = next_obs
            mask = env.get_action_mask()

            if done:
                episodes += 1
                win = reward > 0
                win_window.append(1.0 if win else 0.0)
                ep_rew_window.append(ep_rew); ep_len_window.append(ep_len)
                ep_rew, ep_len = 0.0, 0

                # ── PHASE 2 hooks (inert without a shift schedule) ────
                if shifter.active:
                    rolling = tracker.on_game(win)
                    gamelog.log(tracker.game_idx, global_step,
                                env.opponent_name, win, rolling,
                                len(tracker.shifts) - 1,
                                tracker.games_since_shift())
                    new_opp = shifter.check(global_step)
                    if new_opp:
                        env.set_opponent(new_opp)         # next game = new opp
                        tracker.on_shift(global_step, new_opp)
                        print(f"  SHIFT @ step {global_step:,} / game "
                              f"{tracker.game_idx:,} -> {new_opp} "
                              f"(baseline rolling{W_RECOVERY} = "
                              f"{tracker.shifts[-1]['baseline']:.3f})", flush=True)

                obs  = env.reset()
                mask = env.get_action_mask()

        # ── 2. Bootstrap values for the final step ────────────────
        obs_t  = obs_to_tensor(obs, device)
        mask_t = torch.tensor([[0.0 if done else 1.0]], device=device)
        next_v_m, next_v_w = model.get_next_values(obs_t, goals, states, mask_t)

        # ── 3. FuN loss + update ──────────────────────────────────
        loss, metrics = feudal_loss(
            storage, next_v_m, next_v_w,
            GAMMA_M, GAMMA_W, ALPHA, ENTROPY_COEF, NUM_STEPS, GAE_LAMBDA)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        model.repackage_hidden()
        storage.reset()

        # ── 4. Logging (same metric names as ppo_go.py) ───────────
        win_rate = float(np.mean(win_window)) if win_window else 0.0
        log_dict = {
            "custom/win_rate":       win_rate,
            "custom/total_episodes": episodes,
            "rollout/ep_rew_mean":   float(np.mean(ep_rew_window)) if ep_rew_window else 0.0,
            "rollout/ep_len_mean":   float(np.mean(ep_len_window)) if ep_len_window else 0.0,
            **metrics,
        }
        # ── PHASE 2 metrics (only when a shift schedule is active) ────
        if shifter.active:
            log_dict["phase2/rolling"]           = tracker.rolling()
            log_dict["phase2/games_since_shift"] = tracker.games_since_shift()
            if tracker.shifts:
                s = tracker.shifts[-1]
                log_dict["phase2/shift_idx"] = s["shift_idx"]
                log_dict["phase2/baseline"]  = s["baseline"]
                if s["recovery_games"] is not None:
                    log_dict["phase2/recovery_games_last_shift"] = s["recovery_games"]
        wandb.log(log_dict, step=global_step)

        print(f"  Step {global_step:>9,} | Games {episodes:>6,} | "
              f"WinRate(last{WINDOW}) {win_rate:5.1%} | "
              f"epLen {float(np.mean(ep_len_window)) if ep_len_window else 0.0:5.1f} | "
              f"loss {metrics['loss/total']:.3f}", flush=True)

        # ── 5. Checkpoint ─────────────────────────────────────────
        # Phase 2 (shifter.active) does NOT need intermediate checkpoints — its
        # results live in games.csv and it always resumes from the Phase-1 final,
        # never from a mid-Phase-2 checkpoint. Saving them just burns scratch
        # quota (feudal checkpoints are large), so skip them in Phase 2.
        if not shifter.active and global_step - last_save >= SAVE_EVERY:
            path = os.path.join(SAVE_DIR, f"feudal_go_{global_step}.pt")
            torch.save(model.state_dict(), path)
            print(f"  saved {path}", flush=True)
            last_save = global_step

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "feudal_go_final.pt"))
    print("\n  Final model saved.")
    if shifter.active:
        gamelog.close()
        print(f"  Phase-2 games.csv saved to {SAVE_DIR}")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    train()
