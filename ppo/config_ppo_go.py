# Adapted from adi3e08/PPO (clean minimal PPO) 
# rebuilt for the discrete, action-masked and CNN setting that Go requires.

import os

# Environment: board size, komi, and the per-game move cap (env-overridable so one codebase runs 9x9 and 13x13).
BOARD_SIZE = int(os.environ.get("BOARD_SIZE", 13))
KOMI = float(os.environ.get("KOMI", 7.5))
MAX_MOVES = 16 * BOARD_SIZE * BOARD_SIZE

# Opponent faced this run (set per run) and its randomness (0 = strict, plays purely by its rules).
OPPONENT = os.environ["OPPONENT"]
OPPONENT_EPSILON = 0.0

# Training budget (env steps) and checkpoint interval.
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 5_000_000))
SAVE_EVERY = 200_000

# Phase-2 shift/recovery controls (all default off, so an unset run behaves like Phase 1),
# the recovery window, the derived per-run output paths and the seed.
RESUME_FROM = os.environ.get("RESUME_FROM", "")
SHIFT_SCHEDULE = os.environ.get("SHIFT_SCHEDULE", "")
RUN_TAG = os.environ.get("RUN_TAG", "")
W_RECOVERY = 100
_DIR_KEY = f"{OPPONENT}-{RUN_TAG}" if RUN_TAG else OPPONENT
_BOARD = f"{BOARD_SIZE}x{BOARD_SIZE}"
SAVE_DIR = (f"./phase2_checkpoints/ppo_go/{_BOARD}/{_DIR_KEY}/" 
                   if RUN_TAG 
                   else f"./phase1_checkpoints/ppo_go/{_BOARD}/{_DIR_KEY}/")
LOG_DIR = f"./logs/ppo_go/{_BOARD}/{_DIR_KEY}/"
SEED = int(os.environ.get("SEED", 0))

# Weights & Biases project all runs log to.
WANDB_PROJECT = "honours-rl-go"

# PPO hyperparameters.
LEARNING_RATE = 3e-4 # Adam learning rate
N_STEPS = 2048 # transitions collected per update (rollout length)
BATCH_SIZE = 256 # minibatch size
N_EPOCHS = 4 # optimisation epochs per update
GAMMA = 0.99 # discount factor
GAE_LAMBDA = 0.95 # GAE(λ) trace decay
CLIP_RANGE = 0.2 # PPO clip range (ε)
ENT_COEF = 0.01 # entropy bonus (exploration incentive)
VF_COEF = 0.5 # value-loss weight
MAX_GRAD_NORM = 0.5 # gradient-norm clip
TARGET_KL = 0.03 # early-stop an update if approx KL exceeds ~1.5× this

# CNN feature.
N_CHANNELS = 17 # observation planes (go_v5 encoding)
CNN_FILTERS = 32 # conv filters per layer
CNN_LAYERS = 3 # conv layers in the shared trunk
HIDDEN_DIM = 128 # width of the shared FC layer after the conv trunk

# Phase-1 training-curve win-rate smoothing window.
WINDOW = 200
