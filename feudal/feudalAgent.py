# Feudal Network agent for Go
# Architecture adapted from lweitkamp/feudalnets-pytorch (MIT)

# standard libraries
import os
import sys
import random
from collections import deque

# Make the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# external libraries
import numpy as np
import torch
from pettingzoo.classic import go_v5
from feudal.feudalNetwork import FeudalNetwork, Storage, feudal_loss
from feudal.config_feudal import *

# runtime opponent factory so Phase 2 can swap the opponent mid-training
from opponents import make_opponent
from phase2 import ShiftManager, RecoveryTracker, GameLog

# Environment wrapper — Single-agent view of two-player Go 
# (our agent is Black, the opponent is White). Reward is
# read from env.rewards["black_0"] directly,
# so it is always our reward regardless of whose turn is next.

class GoEnv:
    AGENT = "black_0"
    OPP   = "white_0"

    def __init__(self, board_size=BOARD_SIZE, komi=KOMI, seed=SEED):

        # store config, build the go_v5 env and the starting opponent
        self.board_size = board_size
        self.env = go_v5.env(board_size=board_size, komi=komi)
        self.opponent = make_opponent(OPPONENT, board_size, OPPONENT_EPSILON)
        self.opponent_name = OPPONENT
        self._seed = seed
        self.action_mask = None
        self.move_count = 0

    # Phase 2 - swap the opponent mid-training 
    def set_opponent(self, name):
        self.opponent = make_opponent(name, self.board_size, OPPONENT_EPSILON)
        self.opponent_name = name

    # go_v5 gives the board as (N, N, 17) — height, width, channels-last (HWC),
    # but PyTorch's convolutional layers need channels-first (17, N, N) (CHW).
    # np.transpose reorders the actual axes to do this correctly
    def _obs_chw(self, obs_dict):
        o = obs_dict["observation"].astype(np.float32)
        return np.transpose(o, (2, 0, 1)).copy()

    # True once our agent's episode has terminated or truncated
    def _done(self):
        t, r = self.env.terminations, self.env.truncations
        return (not self.env.agents
                or t.get(self.AGENT, False) or r.get(self.AGENT, False))

    # start a new game - return the first observation
    def reset(self):
        self.env.reset(seed=self._seed)
        self._seed += 1
        self.move_count = 0
        obs, _, _, _, _ = self.env.last()
        self.action_mask = obs["action_mask"].astype(bool)
        return self._obs_chw(obs)

    # apply our move, let the opponent reply, return (obs, our_reward, done)
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

        # Move-cap safeguard - force passes at MAX_MOVES so
        # go_v5 area-scores the board.
        if not done and self.move_count >= MAX_MOVES:
            self._force_finish()
            done = True

        reward = float(self.env.rewards.get(self.AGENT, 0.0))   # OUR reward
        next_obs, _, _, _, _ = self.env.last()
        self.action_mask = next_obs["action_mask"].astype(bool)
        return self._obs_chw(next_obs), reward, done

    # force consecutive passes so go_v5 terminates and area-scores the game
    def _force_finish(self):
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

    # expose the current legal-move mask
    def get_action_mask(self):
        return self.action_mask

    # release the underlying go_v5 environment
    def close(self):
        self.env.close()




# Everything below builds and runs the FuN training loop - set up the run
# (device, W&B, the manager-worker network, Phase-2 resume), then repeatedly
# collect a rollout of games from GoEnv and update the network on it with
# feudal_loss, logging metrics and checkpoints as it goes.

# convert one raw (17, N, N) CHW numpy observation from GoEnv into the
# (1, 17, N, N) batched float tensor the network's forward pass expects
def obs_to_tensor(obs_chw, device):
    return torch.as_tensor(obs_chw, dtype=torch.float32, device=device).unsqueeze(0)


# set up and run one full FuN training run, for the
# board/opponent (Phase 1) or this shift schedule (Phase 2)
def train():

    # banner
    print("=" * 60)
    print(f"  FEUDAL (FuN) — Go {BOARD_SIZE}x{BOARD_SIZE}  vs {OPPONENT}")
    print(f"  Manager-Worker hierarchy + CNN perception (matches PPO trunk)")
    print("=" * 60)

    # output dirs, seeding and device
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  opponent epsilon (difficulty): {OPPONENT_EPSILON}")

    # Weights & Biases logging. 
    # USE_WANDB=0 disables it (e.g. offline nodes)
    # USE_WANDB=1 with WANDB_MODE=offline logs locally to wandb sync later.
    USE_WANDB = os.environ.get("USE_WANDB", "1") == "1"
    if USE_WANDB:
        import wandb
    else:

        # no operation block so the rest of the code can call wandb.* unconditionally
        class _NoWandb:                       
            def init(self, *a, **k): pass
            def log(self, *a, **k): pass
            def finish(self, *a, **k): pass
        wandb = _NoWandb()

    # build the run name - Phase-2 runs get the RUN_TAG suffix
    run_name = f"feudal-{BOARD_SIZE}x{BOARD_SIZE}-vs-{OPPONENT}"
    if RUN_TAG:
        run_name += f"-{RUN_TAG}"
    wandb.init(
        project = WANDB_PROJECT,
        name = run_name,
        dir = LOG_DIR,
        config = {
            "agent": "feudal",
            "board_size": BOARD_SIZE,
            "komi": KOMI,
            "opponent": OPPONENT,
            "opponent_epsilon": OPPONENT_EPSILON,
            "total_timesteps": TOTAL_TIMESTEPS,
            "learning_rate": LEARNING_RATE,
            "num_steps": NUM_STEPS,
            "hidden_dim_m": HIDDEN_DIM_M,
            "hidden_dim_w": HIDDEN_DIM_W,
            "time_horizon": TIME_HORIZON,
            "dilation": DILATION,
            "eps": EPS,
            "alpha": ALPHA,
            "gamma_m": GAMMA_M,
            "gamma_w": GAMMA_W,
            "entropy_coef": ENTROPY_COEF,
            "grad_clip": GRAD_CLIP,
            "cnn_filters": CNN_FILTERS,
            "cnn_layers": CNN_LAYERS,
            "seed": SEED,
            "resume_from": RESUME_FROM,
            "shift_schedule": SHIFT_SCHEDULE,
        },
    )

    # environment + manager-worker network + optimizer
    env = GoEnv()
    model = FeudalNetwork(
        board_size = BOARD_SIZE,
        n_channels = N_CHANNELS,
        n_actions = N_ACTIONS,
        hidden_dim_manager = HIDDEN_DIM_M,
        hidden_dim_worker = HIDDEN_DIM_W,
        time_horizon = TIME_HORIZON,
        dilation = DILATION,
        eps = EPS,
        num_workers = 1,
        device = str(device),
        n_filters = CNN_FILTERS,
        n_layers = CNN_LAYERS,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Phase 2 - resume a Phase-1 checkpoint and set up shift + recovery tracking.
    # All idle in Phase 1 (RESUME_FROM, SHIFT_SCHEDULE unset). The optimizer
    # restarts fresh either way, so both agents get identical treatment.
    if RESUME_FROM:
        model.load_state_dict(torch.load(RESUME_FROM, map_location=device))
        print(f"  resumed weights from {RESUME_FROM}")
    shifter = ShiftManager(SHIFT_SCHEDULE)
    tracker = RecoveryTracker(W_RECOVERY)
    gamelog = GameLog(SAVE_DIR) if shifter.active else None
    if shifter.active:
        print(f"  Phase-2 shift schedule: {shifter.schedule}")

    # rolling-window training metrics (win rate, episode reward, episode length)
    win_window = deque(maxlen=WINDOW)
    ep_rew_window = deque(maxlen=WINDOW)
    ep_len_window = deque(maxlen=WINDOW)

    # first observation + the manager/worker's recurrent state + rollout storage
    obs = env.reset()
    mask = env.get_action_mask()
    goals, states, masks = model.init_obj()
    storage = Storage(NUM_STEPS)
    ep_rew, ep_len = 0.0, 0
    global_step, episodes, last_save = 0, 0, 0
    done = False

    # main training loop - collect a rollout then run a FuN update until budget
    while global_step < TOTAL_TIMESTEPS:
        # 1. Collect a rollout of NUM_STEPS transitions 
        # Detach goals once at the rollout boundary — not every step.
        # The manager trains through the state-goal cosine gradient flowing
        # into the goal - detaching every step zeroed that gradient and the 
        # manager never learned (manager/cosines stayed close to 0) - the fix. 
        # States never carry gradient (manager detaches them already).
        goals = [g.detach() for g in goals]
        for _ in range(NUM_STEPS):
            obs_t = obs_to_tensor(obs, device)
            mask_t = torch.tensor([[0.0 if done else 1.0]], device=device)
            action_mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            dist, goals, states, value_m, value_w = model(
                obs_t, goals, states, mask_t, action_mask_t)
            
            # don't detach the hidden state every step — the dilated LSTM
            # needs memory over memory horizon steps. It's detached once per rollout instead.
            action  = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()
            next_obs, reward, done = env.step(action.item())

            # Keep the mask window in step with the goals/states windows
            # so masks, goals and states stay the same length
            # index memory horizon means the same step in every list — otherwise the 
            # episode-boundary masking used by intrinsic_reward, state_goal_cosine misaligns by one step.
            if len(masks) > (2 * TIME_HORIZON + 1):
                masks.pop(0)
            masks.append(mask_t)

            r_i = model.intrinsic_reward(states, goals, masks)
            s_goal_cos = model.state_goal_cosine(states, goals, masks)

            storage.add({
                'r': torch.tensor([[reward]], device=device),
                'r_i': r_i,
                'v_m': value_m,
                'v_w': value_w,
                'logp': log_prob.unsqueeze(-1),
                'entropy': entropy.unsqueeze(-1),
                's_goal_cos': s_goal_cos,
                'm': mask_t,

                # whether the game is still going after this step (0 if it just ended)
                'nt': torch.tensor([[0.0 if done else 1.0]], device=device),
            })

            ep_rew += reward; ep_len += 1; global_step += 1
            obs  = next_obs
            mask = env.get_action_mask()

            if done:

                # episode ended - record win,reward, length into the windows
                episodes += 1
                win = reward > 0
                win_window.append(1.0 if win else 0.0)
                ep_rew_window.append(ep_rew); ep_len_window.append(ep_len)
                ep_rew, ep_len = 0.0, 0

                # Phase-2 hooks - log the game and apply a shift if one is due
                if shifter.active:
                    rolling = tracker.on_game(win)
                    gamelog.log(tracker.game_idx, global_step,
                                env.opponent_name, win, rolling,
                                len(tracker.shifts) - 1,
                                tracker.games_since_shift())
                    new_opp = shifter.check(global_step)
                    if new_opp:
                        env.set_opponent(new_opp)         
                        tracker.on_shift(global_step, new_opp)
                        print(f"  SHIFT @ step {global_step:,} / game "
                              f"{tracker.game_idx:,} -> {new_opp} "
                              f"(baseline rolling{W_RECOVERY} = "
                              f"{tracker.shifts[-1]['baseline']:.3f})", flush=True)

                # reset for the next episode
                obs  = env.reset()
                mask = env.get_action_mask()

        # 2. Bootstrap values for the final step 
        obs_t  = obs_to_tensor(obs, device)
        mask_t = torch.tensor([[0.0 if done else 1.0]], device=device)
        next_v_m, next_v_w = model.get_next_values(obs_t, goals, states, mask_t)

        # 3. FuN loss + update 
        loss, metrics = feudal_loss(
            storage, next_v_m, next_v_w,
            GAMMA_M, GAMMA_W, ALPHA, ENTROPY_COEF, NUM_STEPS, GAE_LAMBDA)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # detach hidden state once per rollout then clear rollout storage
        model.repackage_hidden()
        storage.reset()

        # 4. Logging (same metric names as ppo_go.py, plus Phase-2)
        win_rate = float(np.mean(win_window)) if win_window else 0.0
        log_dict = {
            "custom/win_rate": win_rate,
            "custom/total_episodes": episodes,
            "rollout/ep_rew_mean": float(np.mean(ep_rew_window)) if ep_rew_window else 0.0,
            "rollout/ep_len_mean": float(np.mean(ep_len_window)) if ep_len_window else 0.0,
            **metrics,
        }

        # Phase-2 recovery metrics (only when a shift schedule is active)
        if shifter.active:
            log_dict["phase2/rolling"] = tracker.rolling()
            log_dict["phase2/games_since_shift"] = tracker.games_since_shift()
            if tracker.shifts:
                s = tracker.shifts[-1]
                log_dict["phase2/shift_idx"] = s["shift_idx"]
                log_dict["phase2/baseline"] = s["baseline"]
                if s["recovery_games"] is not None:
                    log_dict["phase2/recovery_games_last_shift"] = s["recovery_games"]
        wandb.log(log_dict, step=global_step)

        # console progress line
        print(f"  Step {global_step:>9,} | Games {episodes:>6,} | "
              f"WinRate(last{WINDOW}) {win_rate:5.1%} | "
              f"epLen {float(np.mean(ep_len_window)) if ep_len_window else 0.0:5.1f} | "
              f"loss {metrics['loss/total']:.3f}", flush=True)

        # 5. Checkpoint 
        # Phase 2 skips intermediate checkpoints — its results live in games.csv
        # and it always resumes from the Phase-1 final, never a mid-Phase-2 one
        if not shifter.active and global_step - last_save >= SAVE_EVERY:
            path = os.path.join(SAVE_DIR, f"feudal_go_{global_step}.pt")
            torch.save(model.state_dict(), path)
            print(f"  saved {path}", flush=True)
            last_save = global_step

    # final model + Phase-2 per-game log
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "feudal_go_final.pt"))
    print("\n  Final model saved.")
    if shifter.active:
        gamelog.close()
        print(f"  Phase-2 games.csv saved to {SAVE_DIR}")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    train()
