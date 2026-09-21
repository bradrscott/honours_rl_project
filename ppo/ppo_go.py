# PPO for Go — flat baseline agent. Discrete, action-masked categorical policy
# over a CNN trunk on the (N, N, 17) board, trained in PettingZoo
# go_v5 (agent = black) against a heuristic opponent.
#
# The PPO core (GAE, clipped surrogate, advantage normalisation, KL early-stop, grad clip) is
# adapted from adi3e08/PPO; the environment, masking, and metrics are rebuilt for Go.


# standard library imports
import os
import sys
import random
from collections import deque

# Make the project root importable 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# external libraries - numeric/array operations, PyTorch (network + training), the
# categorical action distribution and the go_v5 environment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pettingzoo.classic import go_v5

# config import
from ppo.config_ppo_go import *

# runtime opponent factory so Phase 2 can swap the opponent mid-training
from opponents import make_opponent
from phase2 import ShiftManager, RecoveryTracker, GameLog

# one action per board point, plus one for pass
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1



# environment wrapper — single-agent view of two-player Go (our agent is
# black, the opponent is white).
# Reward handling - env.last() returns the reward for whichever agent's turn
# is next, not necessarily the agent that just moved — so at a game-ending
# move it can silently return the opponent's reward instead of ours. We avoid
# this by always reading env.rewards["black_0"] directly, which looks up our
# agent's own reward by name and is correct no matter whose turn is next.
class GoEnv:
    AGENT = "black_0"
    OPP = "white_0"

    # store config, build the go_v5 env and the starting opponent
    def __init__(self, board_size=BOARD_SIZE, komi=KOMI, seed=SEED):
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

    # returns True once our agent's episode has ended (either terminated
    # normally, e.g. both players passed, or truncated, e.g. hit a step limit)
    def _done(self):
        t = self.env.terminations
        r = self.env.truncations
        return (not self.env.agents
                or t.get(self.AGENT, False) or r.get(self.AGENT, False))

    # start a new game and return the first observation
    def reset(self):
        self.env.reset(seed=self._seed)
        self._seed += 1  
        self.move_count = 0
        # only the observation is used; the other 4 fields (reward, terminated,
        # truncated, info) are for whichever agent moves next, not ours
        obs, _, _, _, _ = self.env.last()
        self.action_mask = obs["action_mask"].astype(bool)
        return self._obs_chw(obs)

    # apply our move, let the opponent reply and return (obs, our_reward, done)
    def step(self, action):
        # play our move
        self.env.step(int(action)); self.move_count += 1

        # opponent replies if the game is still going and it's their turn
        if (self.env.agents
                and self.env.agent_selection == self.OPP
                and not (self.env.terminations.get(self.OPP, False)
                         or self.env.truncations.get(self.OPP, False))):
            # again, only the observation is needed here
            opp_obs, _, _, _, _ = self.env.last()
            opp_action = self.opponent.select_action(opp_obs)
            self.env.step(opp_action); self.move_count += 1

        # has the game ended on its own
        done = self._done()

        # Move-cap safeguard - go_v5 has no move limit and only ends on two
        # consecutive passes, so at MAX_MOVES we force passes so it area-scores
        # the board and we read the real result. Our move is already recorded.
        if not done and self.move_count >= MAX_MOVES:
            self._force_finish()
            done = True

        # read our own reward (see class note on why not env.last())
        reward = float(self.env.rewards.get(self.AGENT, 0.0))  # OUR reward

        # refresh the observation + legal-move mask for the returned state
        next_obs, _, _, _, _ = self.env.last()
        self.action_mask = next_obs["action_mask"].astype(bool)
        return self._obs_chw(next_obs), reward, done

    # force consecutive passes so go_v5 terminates and area-scores the game
    def _force_finish(self):
        pass_action = self.board_size * self.board_size
        guard = 0

        # keep acting until the game ends 
        while self.env.agents and not self._done() and guard < 4:
            cur = self.env.agent_selection
            # a finished agent must be stepped with None; a live one passes
            if (self.env.terminations.get(cur, False)
                    or self.env.truncations.get(cur, False)):
                self.env.step(None)
            else:
                self.env.step(pass_action)
            guard += 1

    # expose the current legal move mask
    def get_action_mask(self):
        return self.action_mask



# Actor-Critic network — shared CNN trunk, separate policy/value heads.
# Orthogonal weight initialisation for Linear/Convolutional layers with the
# bias zeroed. 
# keeps the weight matrix's rows independent of each other (like a scaled rotation),
# which helps preserve signal size as it passes through many layers at the start of 
# training instead of vanishing or exploding 
def _orthogonal(module, gain):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        # zero the bias if the layer has one
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# The full network PPO trains - one CNN trunk that reads the board, feeding
# two separate heads. The actor (policy) head outputs a probability over
# every possible move, the critic (value) head outputs a single number
# estimating how good the current position is. Sharing the trunk between
# them is the standard PPO design — it means both heads reuse the same
# learned board features instead of each learning their own from scratch.
class ActorCritic(nn.Module):
    # build the conv trunk, shared FC layer, and policy/value heads
    def __init__(self, board_size, n_channels, n_actions,
                 n_filters, n_layers, hidden_dim):
        super().__init__()
        self.board_size = board_size
        self.n_channels = n_channels

        # conv trunk over the board planes: stack n_layers of conv+ReLU, then
        # flatten the resulting feature maps into a single vector so they can
        # feed the fully connected layer below
        conv, in_ch = [], n_channels
        for _ in range(n_layers):
            conv += [nn.Conv2d(in_ch, n_filters, 3, padding=1), nn.ReLU()]
            in_ch = n_filters
        conv.append(nn.Flatten())
        self.trunk = nn.Sequential(*conv)

        # shared fully connected layer then separate policy (actor) and value (critic) heads
        flat = n_filters * board_size * board_size
        self.shared = nn.Sequential(nn.Linear(flat, hidden_dim), nn.ReLU())
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        # orthogonal init - small gain on the policy head, std=1 on value
        self.apply(lambda m: _orthogonal(m, gain=np.sqrt(2)))
        _orthogonal(self.actor, gain=0.01)
        _orthogonal(self.critic, gain=1.0)

    # conv trunk to shared FC to feature vector
    def _features(self, obs):
        return self.shared(self.trunk(obs))

    # set illegal-move logits to -inf so they get zero probability
    def _masked_dist(self, logits, mask):
        logits = logits.masked_fill(~mask, -1e8)
        return Categorical(logits=logits)

    # sample an action (used during rollout collection)
    def act(self, obs, mask):
        f = self._features(obs)
        dist = self._masked_dist(self.actor(f), mask)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(f).squeeze(-1)

    # re-evaluate stored actions (used during the PPO update)
    def evaluate(self, obs, mask, actions):
        f = self._features(obs)
        dist = self._masked_dist(self.actor(f), mask)
        return (dist.log_prob(actions), dist.entropy(),
                self.critic(f).squeeze(-1))

    # critic-only forward pass (state value)
    def value(self, obs):
        return self.critic(self._features(obs)).squeeze(-1)




# TRAINING — the advantage-estimation helper below and train() -
# loop that repeatedly collects a rollout of games, runs a PPO update on it,
# logs the results and checkpoints the network until the step budget is used.


# Generalised Advantage Estimation (GAE) - estimates how much better each
# action was than the critic expected, which is what PPO's
# policy update is based on. Rather than using the raw reward alone (noisy)
# or the critic's value alone (biased), GAE blends the two across the whole
# rollout, walking backwards from the end so each step's estimate also
# incorporates the ones that came after it.
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0

    # walk backwards through the rollout accumulating the GAE recursion
    for t in reversed(range(T)):
        # bootstrap from the next state's value (or last_value at the end)
        next_v = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        gae   = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae

    # returns = advantages + baseline values
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


# main training entry point - sets up the env/network/logging, then loops
# rollout collection -> PPO update -> checkpointing until the step budget is spent
def train():
    # banner
    print("=" * 60)
    print(f"  PPO (new) — Go {BOARD_SIZE}x{BOARD_SIZE}  vs {OPPONENT}")
    print(f"  Discrete masked Categorical policy + CNN trunk")
    print("=" * 60)

    # output dirs, seeding and device
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Weights & Biases logging. USE_WANDB=0 disables it (e.g. offline nodes)
    # USE_WANDB=1 with WANDB_MODE=offline logs locally to wandb sync later.
    USE_WANDB = os.environ.get("USE_WANDB", "1") == "1"
    if USE_WANDB:
        import wandb
    else:

        # fake wandb with matching methods that do nothing, so the rest of train()
        # can call wandb.* without checking USE_WANDB every time
        class _NoWandb:
            def init(self, *a, **k): pass
            def log(self, *a, **k): pass
            def finish(self, *a, **k): pass
        wandb = _NoWandb()

    # build the run name - Phase-2 runs get the RUN_TAG suffix
    run_name = f"ppo-go-{BOARD_SIZE}x{BOARD_SIZE}-vs-{OPPONENT}"
    if RUN_TAG:
        run_name += f"-{RUN_TAG}"

    # lowercase config keys match the earlier runs so they share table columns
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        dir=LOG_DIR,
        config={
            "board_size": BOARD_SIZE,
            "komi": KOMI,
            "opponent": OPPONENT,
            "total_timesteps": TOTAL_TIMESTEPS,
            "learning_rate": LEARNING_RATE,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
            "target_kl": TARGET_KL,
            "cnn_filters": CNN_FILTERS,
            "cnn_layers": CNN_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "seed": SEED,
            "resume_from": RESUME_FROM,
            "shift_schedule": SHIFT_SCHEDULE,
        },
    )

    print(f"  opponent epsilon (difficulty): {OPPONENT_EPSILON}")

    # environment, network, optimizer
    env = GoEnv()
    net = ActorCritic(BOARD_SIZE, N_CHANNELS, N_ACTIONS, CNN_FILTERS, CNN_LAYERS, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Phase 2 - resume a Phase-1 checkpoint and set up shift + recovery tracking.
    # In Phase 1, RESUME_FROM and SHIFT_SCHEDULE are both unset, so this block
    # does nothing and training just starts from a fresh network. The optimizer
    # restarts fresh either way, so both agents get identical treatment.
    if RESUME_FROM:
        net.load_state_dict(torch.load(RESUME_FROM, map_location=device))
        print(f"  resumed weights from {RESUME_FROM}")
    shifter = ShiftManager(SHIFT_SCHEDULE)
    tracker = RecoveryTracker(W_RECOVERY)
    gamelog = GameLog(SAVE_DIR) if shifter.active else None

    # announce the schedule in Phase 2
    if shifter.active:
        print(f"  Phase-2 shift schedule: {shifter.schedule}")

    # rolling-window training metrics (win rate, episode reward, episode length)
    win_window = deque(maxlen=WINDOW)
    ep_rew_window = deque(maxlen=WINDOW)
    ep_len_window = deque(maxlen=WINDOW)

    # first observation + per-episode / global counters
    obs = env.reset()
    mask = env.get_action_mask()
    ep_rew, ep_len = 0.0, 0
    global_step, episodes, last_save = 0, 0, 0

    # main training loop - collect a rollout then run a PPO update, until budget
    while global_step < TOTAL_TIMESTEPS:
        # 1. Collect a rollout of N_STEPS transitions
        b_obs, b_mask, b_act, b_logp, b_val, b_rew, b_done = [], [], [], [], [], [], []

        # step the policy through the env, storing each transition
        for _ in range(N_STEPS):
            obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            # sample an action (no gradient tracking during collection)
            with torch.no_grad():
                action, logp, value = net.act(obs_t, mask_t)

            next_obs, reward, done = env.step(action.item())

            # store the transition
            b_obs.append(obs); b_mask.append(mask); b_act.append(action.item())
            b_logp.append(logp.item()); b_val.append(value.item())
            b_rew.append(reward); b_done.append(float(done))

            ep_rew += reward; ep_len += 1; global_step += 1
            obs, mask = next_obs, env.get_action_mask()

            if done:
                # episode ended - record win / reward / length into the windows
                episodes += 1
                win = reward > 0
                win_window.append(1.0 if win else 0.0)
                ep_rew_window.append(ep_rew); ep_len_window.append(ep_len)
                ep_rew, ep_len = 0.0, 0

                # Phase-2 hooks - log the game and apply a shift if one is due
                # (does nothing without a shift schedule).
                if shifter.active:
                    rolling = tracker.on_game(win)
                    gamelog.log(tracker.game_idx, global_step,
                                env.opponent_name, win, rolling,
                                len(tracker.shifts) - 1,
                                tracker.games_since_shift())
                    new_opp = shifter.check(global_step)

                    # a shift is due -> switch opponent for the next game
                    if new_opp:
                        env.set_opponent(new_opp)         
                        tracker.on_shift(global_step, new_opp)
                        print(f"  SHIFT @ step {global_step:,} / game "
                              f"{tracker.game_idx:,} -> {new_opp} "
                              f"(baseline rolling{W_RECOVERY} = "
                              f"{tracker.shifts[-1]['baseline']:.3f})", flush=True)

                # reset for the next episode
                obs = env.reset(); mask = env.get_action_mask()

        # 2. GAE + returns
        # bootstrap value for the state after the rollout
        with torch.no_grad():
            last_v = net.value(
                torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            ).item()
        adv, returns = compute_gae(b_rew, b_val, b_done, last_v, GAMMA, GAE_LAMBDA)

        # batch -> tensors
        obs_t = torch.as_tensor(np.array(b_obs), dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(np.array(b_mask), dtype=torch.bool, device=device)
        act_t = torch.as_tensor(b_act, dtype=torch.long, device=device)
        oldlogp_t = torch.as_tensor(b_logp, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        # 3. PPO update (minibatch epochs, clipped surrogate, KL early-stop)
        idx = np.arange(N_STEPS)
        approx_kl = 0.0
        clipfracs, pg_losses, v_losses, ent_losses = [], [], [], []
        stop = False

        # multiple epochs over the rollout
        for epoch in range(N_EPOCHS):
            np.random.shuffle(idx)

            # one gradient step per shuffled minibatch
            for start in range(0, N_STEPS, BATCH_SIZE):
                mb = idx[start:start + BATCH_SIZE]
                new_logp, entropy, value = net.evaluate(obs_t[mb], mask_t[mb], act_t[mb])

                # advantage normalisation (batch level)
                mb_adv = adv_t[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # probability ratio between new and old policy
                ratio = torch.exp(new_logp - oldlogp_t[mb])

                # diagnostics: approx KL + clip fraction (no grad)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_logp - oldlogp_t[mb])).mean().item()
                    clipfracs.append(((ratio - 1).abs() > CLIP_RANGE).float().mean().item())

                # clipped surrogate policy loss
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(value, ret_t[mb])
                ent = entropy.mean()

                # total loss - policy + value − entropy bonus
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * ent

                # gradient step with norm clipping
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

        # 4. Logging (training metrics, plus Phase-2 recovery metrics)
        win_rate = float(np.mean(win_window)) if win_window else 0.0
        metrics = {
            "custom/win_rate": win_rate,
            "custom/total_episodes": episodes,
            "rollout/ep_rew_mean": float(np.mean(ep_rew_window)) if ep_rew_window else 0.0,
            "rollout/ep_len_mean": float(np.mean(ep_len_window)) if ep_len_window else 0.0,
            "train/policy_loss": float(np.mean(pg_losses)),
            "train/value_loss": float(np.mean(v_losses)),
            "train/entropy": float(np.mean(ent_losses)),
            "train/approx_kl": approx_kl,
            "train/clip_fraction": float(np.mean(clipfracs)),
            "train/explained_var": explained_var,
            "train/early_stopped": float(stop),
        }
        # Phase-2 recovery metrics (only when a shift schedule is active)
        if shifter.active:
            metrics["phase2/rolling"] = tracker.rolling()
            metrics["phase2/games_since_shift"] = tracker.games_since_shift()

            # attach the most recent shift's details
            if tracker.shifts:
                s = tracker.shifts[-1]
                metrics["phase2/shift_idx"] = s["shift_idx"]
                metrics["phase2/baseline"] = s["baseline"]

                # only once recovery has actually been measured
                if s["recovery_games"] is not None:
                    metrics["phase2/recovery_games_last_shift"] = s["recovery_games"]
        wandb.log(metrics, step=global_step)

        # console progress line
        print(f"  Step {global_step:>9,} | Games {episodes:>6,} | "
              f"WinRate(last{WINDOW}) {win_rate:5.1%} | "
              f"epLen {metrics['rollout/ep_len_mean']:5.1f} | "
              f"KL {approx_kl:.4f}", flush=True)

        # 5. Checkpoint
        # Phase 2 skips intermediate checkpoints — its results live in games.csv
        # and it always resumes from the Phase-1 final, never a mid-Phase-2 one.
        if not shifter.active and global_step - last_save >= SAVE_EVERY:
            path = os.path.join(SAVE_DIR, f"ppo_go_{global_step}.pt")
            torch.save(net.state_dict(), path)
            print(f"  saved {path}", flush=True)
            last_save = global_step

    # final model + Phase-2 recovery summary
    torch.save(net.state_dict(), os.path.join(SAVE_DIR, "ppo_go_final.pt"))
    print("\n  Final model saved.")

    # Phase 2 - close the per-game log
    if shifter.active:
        gamelog.close()
        print(f"  Phase-2 games.csv saved to {SAVE_DIR}")
    wandb.finish()


if __name__ == "__main__":
    train()
