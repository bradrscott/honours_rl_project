# FuN (Feudal Network) config for feudalAgent.py.

# standard library
import os

# Environment - board size, komi and the per-game ply cap (env-overridable so one codebase runs 9x9 and 13x13).
BOARD_SIZE = int(os.environ.get("BOARD_SIZE", 13))
KOMI = float(os.environ.get("KOMI", 7.5))
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1     
MAX_MOVES = 16 * BOARD_SIZE * BOARD_SIZE    

# Opponent faced this run (set per run) and its randomness.
OPPONENT = os.environ["OPPONENT"]
OPPONENT_EPSILON = 0.0

# CNN feature
N_CHANNELS = 17 # observation planes (go_v5 encoding)
CNN_FILTERS = 32 # conv filters per layer
CNN_LAYERS = 3 # conv layers in the shared trunk

# Training budget (env steps), checkpoint interval, seed, W&B project and the
# Phase-1 training-curve smoothing window.
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 5_000_000))
SAVE_EVERY = 200_000
SEED = int(os.environ.get("SEED", 0))
WANDB_PROJECT = "honours-rl-go"
WINDOW = 200

# Phase-2 shift/recovery controls (all default off, so an unset run behaves like Phase 1),
# the recovery window, the derived per-run output paths.
RESUME_FROM = os.environ.get("RESUME_FROM", "")
SHIFT_SCHEDULE = os.environ.get("SHIFT_SCHEDULE", "")
RUN_TAG = os.environ.get("RUN_TAG", "")
W_RECOVERY = 100
_DIR_KEY = f"{OPPONENT}-{RUN_TAG}" if RUN_TAG else OPPONENT
_BOARD = f"{BOARD_SIZE}x{BOARD_SIZE}"
SAVE_DIR = (f"./phase2_checkpoints/feudal/{_BOARD}/{_DIR_KEY}/" 
            if RUN_TAG
            else f"./phase1_checkpoints/feudal/{_BOARD}/{_DIR_KEY}/")
LOG_DIR = f"./logs/feudal/{_BOARD}/{_DIR_KEY}/"

# FuN hyperparameters 
LEARNING_RATE = 2e-4 # learning rate (tuned for FuN's tighter step budget - PPO uses 3e-4)
HIDDEN_DIM_M = 256 # manager/perception latent dimension (d)
HIDDEN_DIM_W = 16 # worker embedding dimension (k)
TIME_HORIZON = 15 # steps the manager commits to a goal (c)
DILATION = 15 # dilated LSTM radius (r); kept in sync with TIME_HORIZON
EPS = 0.1 # probability of a random goal (manager exploration)
ALPHA = 0.5 # intrinsic-reward mixing weight in the worker advantage
GAMMA_M = 0.995 # manager discount factor
GAMMA_W = 0.99 # worker discount factor (matches PPO's GAMMA)
GAE_LAMBDA = 0.95 # GAE(λ) trace decay (matches PPO)
ENTROPY_COEF = 0.01 # worker entropy bonus (matches PPO)
NUM_STEPS = 400 # steps collected per rollout/update (K)
GRAD_CLIP = 0.5 # max gradient norm
