# ══════════════════════════════════════════════════════════════
# FEUDAL CONFIG
# Edit values here before a run. Import into feudalAgent.py with:
#
#   from config_feudal import *
# ══════════════════════════════════════════════════════════════

# ── Environment ───────────────────────────────────────────────
BOARD_SIZE      = 19
OBS_DIM         = BOARD_SIZE * BOARD_SIZE * 17   # 6137
N_ACTIONS       = BOARD_SIZE * BOARD_SIZE + 1    # 362

# ── Training ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 10_000_000
SAVE_EVERY      = 500_000
SAVE_DIR        = "./models/feudal/"
LOG_DIR         = "./logs/feudal/"

# ── FuN hyperparameters ───────────────────────────────────────

# learning_rate: reduced from 3e-4 to 1e-4.
#   FuN has two loss streams (manager + worker) compounding gradient
#   updates — more sensitive than PPO. 3e-4 was causing win rate to
#   decline. Same fix that worked for PPO.
LEARNING_RATE   = 1e-4

# hidden_dim_manager (d): manager LSTM hidden dimension.
#   Keep at 256 — perception already maps 6137 → 256 → d.
#   Increasing doesn't help much; decreasing loses capacity.
HIDDEN_DIM_M    = 256

# hidden_dim_worker (k): worker embedding dimension.
#   Keep at 16 — worker LSTM is already LSTMCell(256, 16*362=5792).
#   Increasing k would double checkpoint size (already 545MB).
HIDDEN_DIM_W    = 16

# time_horizon (c): how many steps the manager commits to a goal.
#   THIS WAS THE KEY PROBLEM. At c=10 in 300-move Go games, the
#   manager was changing goals every 10 steps — too fast for the
#   worker to follow, causing cosines to stay near 0.
#   Increased to 15 to give the worker more time to follow each goal.
#   Try: 10 → 15 → 20 if cosines still don't trend upward.
TIME_HORIZON    = 15

# dilation (r): dilated LSTM radius.
#   Always keep in sync with TIME_HORIZON.
DILATION        = 15

# eps: probability of a random goal (manager exploration).
EPS             = 0.1

# alpha: intrinsic reward mixing coefficient.
#   Increased from 0.5 to 0.7 to pull the worker more strongly
#   toward manager goals. Helps fix the low cosines problem.
#   Too high → worker over-constrained, ignores local board threats.
#   Try: 0.5 → 0.7 → 0.9 if cosines still near 0.
ALPHA           = 0.7

# gamma_manager: manager discount factor.
#   Increased from 0.99 to 0.995 — Go games are 200-300 moves,
#   manager needs to value outcomes further in the future.
GAMMA_M         = 0.995

# gamma_worker: worker discount factor.
#   Slightly lower than manager — worker focuses on immediate sub-goals.
GAMMA_W         = 0.95

# entropy_coef: entropy bonus for worker action distribution.
#   Increased from 0.01 to 0.02 — worker entropy was trending
#   downward, risking collapse to a narrow action set.
ENTROPY_COEF    = 0.02

# num_steps (K): steps collected per rollout.
#   Increased from 400 to 600 — more steps gives better return
#   estimates for sparse Go rewards.
NUM_STEPS       = 600

# grad_clip: max norm for gradient clipping.
#   Keep at 0.5 — important for hierarchical agents where
#   manager and worker gradients can compound.
GRAD_CLIP       = 0.5
