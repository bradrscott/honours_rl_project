# ══════════════════════════════════════════════════════════════
# FEUDAL CONFIG  (feudalAgent.py — FuN on Go)
# Edit values here before a run. Import into feudalAgent.py with:
#
#   from config_feudal import *
#
# This is kept DELIBERATELY PARALLEL to config_ppo_go.py so the
# Feudal vs flat-PPO comparison (RQ3) is fair: SAME board, SAME komi,
# SAME opponents at the SAME strict difficulty, SAME CNN feature
# extractor, SAME rolling-window win-rate metric. The ONLY intended
# difference between the two agents is the architecture (flat actor-
# critic vs Manager-Worker hierarchy).
# ══════════════════════════════════════════════════════════════

# ── Environment (matched to config_ppo_go.py) ─────────────────────
BOARD_SIZE      = 9
KOMI            = 5.5
N_ACTIONS       = BOARD_SIZE * BOARD_SIZE + 1     # 82 on 9x9

# Max plies before a game is force-ended (see config_ppo_go.py for the full
# reasoning). Counts PLIES; per-player cap is MAX_MOVES/2. Set above natural
# game length so games aren't sliced off before the agent can win. MUST match
# the PPO setting for a fair comparison.
MAX_MOVES       = 16 * BOARD_SIZE * BOARD_SIZE   # 2704 plies ~= 1352 of our moves

# Opponent: "random" | "greedy" | "aggressive" | "defensive" | "corner" | "edge"
# Change THIS line to switch opponent, then re-upload + resubmit the job.
OPPONENT        = "greedy"

# Strict opponents: 0.0 = plays purely by its rules (no random moves).
# Same shared difficulty PPO faces. (No effect on the random opponent.)
OPPONENT_EPSILON = 0.0

# ── CNN feature extractor (IDENTICAL to config_ppo_go.py) ─────────
# The original scaffold (lweitkamp/feudalnets-pytorch) uses a CNN
# Perception by default; the flat board MLP was an adaptation. We use
# the SAME conv trunk as PPO so the only architectural difference is
# the Manager-Worker hierarchy, not the feature extractor.
N_CHANNELS      = 17
CNN_FILTERS     = 32
CNN_LAYERS      = 3

# ── Training (matched to config_ppo_go.py) ────────────────────────
TOTAL_TIMESTEPS = 3_000_000
SAVE_EVERY      = 200_000
SEED            = 0
WANDB_PROJECT   = "honours-rl-go"
WINDOW          = 200     # rolling window (games) for the win-rate metric

# Per-opponent dirs so different opponent runs never overwrite each other.
SAVE_DIR        = f"./models/feudal/{OPPONENT}/"
LOG_DIR         = f"./logs/feudal/{OPPONENT}/"

# ══════════════════════════════════════════════════════════════
# FuN hyperparameters (architecture-specific — NOT shared with PPO)
# ══════════════════════════════════════════════════════════════

# learning_rate: Adam, 1e-4. FuN has two compounding loss streams
# (manager + worker), so it is more sensitive than PPO; 3e-4 made the
# win rate decline. (Scaffold used RMSprop 5e-4 for Atari.)
LEARNING_RATE   = 1e-4

# hidden_dim_manager (d): manager/perception latent dimension.
HIDDEN_DIM_M    = 256

# hidden_dim_worker (k): worker embedding dimension. Worker LSTM is
# LSTMCell(d, k*n_actions); on 9x9 that is LSTMCell(256, 16*82=1312).
HIDDEN_DIM_W    = 16

# time_horizon (c): how many steps the manager commits to a goal.
# Lowered to 10 for 9x9 (games are ~40-80 moves, far shorter than the
# 200-300 of 19x19). The scaffold default is also 10. Try 10 -> 15 if
# manager/cosines stay flat near 0.
TIME_HORIZON    = 10

# dilation (r): dilated LSTM radius. Keep in sync with TIME_HORIZON.
DILATION        = 10

# eps: probability of a random goal (manager exploration).
EPS             = 0.1

# alpha: intrinsic-reward mixing weight in the worker advantage.
ALPHA           = 0.7

# gamma_manager / gamma_worker: separate discount factors.
GAMMA_M         = 0.99
GAMMA_W         = 0.95

# entropy_coef: worker action-distribution entropy bonus.
ENTROPY_COEF    = 0.02

# num_steps (K): steps collected per rollout / update.
NUM_STEPS       = 600

# grad_clip: max gradient norm.
GRAD_CLIP       = 0.5
