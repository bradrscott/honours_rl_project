# ── Imports ───────────────────────────────────────────────────
# Feudal Network: adapted from lweitkamp/feudalnets-pytorch (MIT)
#   Architecture: Vezhnevets et al. (2017) https://arxiv.org/abs/1703.01161

# Go board environment: PettingZoo (Terry et al., 2021)
from pettingzoo.classic import go_v5

# Heuristic opponent
from randomOpponent import RandomOpponent

from feudalNetwork import FeudalNetwork, Storage, feudal_loss

import wandb
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

# ── Configuration ──────────────────────────────────────────────
BOARD_SIZE     = 19
OBS_DIM        = BOARD_SIZE * BOARD_SIZE * 17   # 6137
N_ACTIONS      = BOARD_SIZE * BOARD_SIZE + 1    # 362

# FuN hyperparameters
HIDDEN_DIM_M   = 256     # Manager hidden dimension (d)
HIDDEN_DIM_W   = 16      # Worker hidden dimension (k)
TIME_HORIZON   = 10      # Manager time horizon (c)
DILATION       = 10      # Dilated LSTM radius (r)
EPS            = 0.1     # Random goal probability (exploration)
ALPHA          = 0.5     # Intrinsic reward mixing coefficient
GAMMA_M        = 0.99    # Manager discount factor
GAMMA_W        = 0.95    # Worker discount factor
ENTROPY_COEF   = 0.01    # Entropy bonus
LEARNING_RATE  = 3e-4
NUM_STEPS      = 400     # Steps per rollout (K from paper)

TOTAL_TIMESTEPS = 50_000_000
SAVE_EVERY      = 500_000
SAVE_DIR        = "./models/feudal/"
LOG_DIR         = "./logs/feudal/"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# GO ENVIRONMENT WRAPPER
# Identical to ppoAgent.py — same environment, same rules.
# ══════════════════════════════════════════════════════════════

class GoEnvWrapper(gym.Env):
    """
    Wraps PettingZoo's two-agent Go into a single-agent Gymnasium env.
    Our agent plays Black; White plays via a heuristic opponent class.

    Action masking ensures only legal moves are ever selected.
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

        # Action is guaranteed legal — FeudalNetwork applies action mask
        self.env.step(int(action))

        # White (heuristic opponent) plays a legal move
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
# TRAINING
# ══════════════════════════════════════════════════════════════

def obs_to_tensor(obs, device):
    """Convert numpy observation to (1, obs_dim) tensor."""
    return torch.FloatTensor(obs).unsqueeze(0).to(device)


def train():
    print("=" * 55)
    print(f"  Feudal Network Agent — Go {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"  Architecture: Vezhnevets et al. (2017) FuN")
    print(f"  Source: lweitkamp/feudalnets-pytorch (adapted)")
    print(f"  Board: PettingZoo go_v5")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Wandb — mirrors TensorBoard exactly
    wandb.tensorboard.patch(root_logdir=LOG_DIR)
    wandb.init(
        project          = "honours-rl-go",
        name             = "feudal-vs-random",
        sync_tensorboard = True,
        config           = {
            "board_size":       BOARD_SIZE,
            "total_timesteps":  TOTAL_TIMESTEPS,
            "hidden_dim_m":     HIDDEN_DIM_M,
            "hidden_dim_w":     HIDDEN_DIM_W,
            "time_horizon":     TIME_HORIZON,
            "dilation":         DILATION,
            "alpha":            ALPHA,
            "gamma_m":          GAMMA_M,
            "gamma_w":          GAMMA_W,
            "entropy_coef":     ENTROPY_COEF,
            "learning_rate":    LEARNING_RATE,
            "num_steps":        NUM_STEPS,
            "opponent":         "random",
        }
    )

    # Environment and model
    env   = GoEnvWrapper(board_size=BOARD_SIZE)
    model = FeudalNetwork(
        input_dim        = OBS_DIM,
        n_actions        = N_ACTIONS,
        hidden_dim_manager = HIDDEN_DIM_M,
        hidden_dim_worker  = HIDDEN_DIM_W,
        time_horizon     = TIME_HORIZON,
        dilation         = DILATION,
        eps              = EPS,
        num_workers      = 1,
        device           = str(device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"Logs: https://wandb.ai\n")

    # Initialise environment and model state
    obs, _          = env.reset()
    goals, states, init_masks = model.init_obj()
    masks           = init_masks
    storage         = Storage(NUM_STEPS)

    global_step     = 0
    last_save       = 0
    last_ep_count   = 0
    done            = False

    while global_step < TOTAL_TIMESTEPS:

        # ── Collect NUM_STEPS rollout ─────────────────────────
        for step in range(NUM_STEPS):
            obs_t        = obs_to_tensor(obs, device)
            mask_t       = torch.FloatTensor([[0.0 if done else 1.0]]).to(device)
            action_mask_t = torch.BoolTensor(env.get_action_mask()).unsqueeze(0).to(device)

            # Forward pass
            dist, goals, states, value_m, value_w = model(
                obs_t, goals, states, mask_t, action_mask_t
            )

            # Sample action
            action   = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Intrinsic reward and Manager loss signal
            masks.append(mask_t)
            if len(masks) > (2 * TIME_HORIZON + 1):
                masks.pop(0)

            r_i         = model.intrinsic_reward(states, goals, masks)
            s_goal_cos  = model.state_goal_cosine(states, goals, masks)

            # Store experience
            storage.add({
                'r':          torch.FloatTensor([[reward]]).to(device),
                'r_i':        r_i,
                'v_m':        value_m,
                'v_w':        value_w,
                'logp':       log_prob.unsqueeze(-1),
                'entropy':    entropy.unsqueeze(-1),
                's_goal_cos': s_goal_cos,
                'm':          mask_t,
            })

            obs = next_obs
            global_step += 1

            if done:
                obs, _ = env.reset()

        # ── Bootstrap values for final step ──────────────────
        obs_t     = obs_to_tensor(obs, device)
        mask_t    = torch.FloatTensor([[0.0 if done else 1.0]]).to(device)
        next_v_m, next_v_w = model.get_next_values(obs_t, goals, states, mask_t)

        # ── Compute loss and update ───────────────────────────
        loss, metrics = feudal_loss(
            storage, next_v_m, next_v_w,
            GAMMA_M, GAMMA_W, ALPHA, ENTROPY_COEF, NUM_STEPS
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────
        # Always log training metrics after every update
        wandb.log(metrics, step=global_step)

        # Log win rate after every completed game
        if env.episode_count > last_ep_count:
            wr = env.win_count / env.episode_count
            print(f"  Game {env.episode_count:>5,} | "
                  f"Step {global_step:>8,} | "
                  f"Win rate: {wr:.1%}", flush=True)
            wandb.log({
                'custom/win_rate':       wr,
                'custom/total_episodes': env.episode_count,
                'custom/total_wins':     env.win_count,
            }, step=global_step)
            last_ep_count = env.episode_count

        # ── Checkpoint ────────────────────────────────────────
        if global_step - last_save >= SAVE_EVERY:
            path = os.path.join(SAVE_DIR, f"feudal_go_{global_step}_steps.pt")
            torch.save(model.state_dict(), path)
            print(f"  ✓ Checkpoint saved: {path}")
            last_save = global_step

        # Detach hidden states to prevent gradient accumulation
        model.repackage_hidden()
        storage.reset()

    # Final save
    final_path = os.path.join(SAVE_DIR, "feudal_go_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n  ✓ Final model saved: {final_path}")

    # Final evaluation
    print("\nRunning final evaluation (50 games)...")
    model.eval()
    wins = 0
    for _ in range(50):
        obs, _         = env.reset()
        done           = False
        eval_goals, eval_states, eval_masks = model.init_obj()
        while not done:
            obs_t        = obs_to_tensor(obs, device)
            mask_t       = torch.FloatTensor([[1.0]]).to(device)
            action_mask_t = torch.BoolTensor(env.get_action_mask()).unsqueeze(0).to(device)
            with torch.no_grad():
                dist, eval_goals, eval_states, _, _ = model(
                    obs_t, eval_goals, eval_states, mask_t, action_mask_t
                )
            action = dist.probs.argmax(dim=-1)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
        if reward > 0:
            wins += 1

    print(f"\n  Final win rate: {wins/50:.1%}")
    wandb.finish()
    env.close()


if __name__ == '__main__':
    train()
