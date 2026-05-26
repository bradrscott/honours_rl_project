# ══════════════════════════════════════════════════════════════
# DQN CONFIG
# Edit values here before a run. Import into dqnAgent.py with:
#
#   from config_dqn import *
# ══════════════════════════════════════════════════════════════

# ── Environment ───────────────────────────────────────────────
BOARD_SIZE      = 19
OBS_DIM         = BOARD_SIZE * BOARD_SIZE * 17   # 6137
N_ACTIONS       = BOARD_SIZE * BOARD_SIZE + 1    # 362
N_CHANNELS      = 17                              # observation planes

# ── Training ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 10_000_000
SAVE_EVERY      = 500_000
SAVE_DIR        = "./models/dqn/"
LOG_DIR         = "./logs/dqn/"

# ── DQN hyperparameters ───────────────────────────────────────
LEARNING_RATE       = 5e-5
BUFFER_SIZE         = 50_000
LEARNING_STARTS     = 10_000
BATCH_SIZE          = 64
GAMMA               = 0.99
TRAIN_FREQ          = 4
TARGET_UPDATE_FREQ  = 1_000
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 1_000_000

# ── CNN architecture ──────────────────────────────────────────
# Same reasoning as PPO — Go is spatial, CNN is required.
# Input reshaped from flat (6137,) to (17, 19, 19) inside QNetwork.
CNN_FILTERS     = 64    # filters per conv layer
CNN_LAYERS      = 5     # was 3 — more layers = better spatial reasoning
HEAD_ARCH       = [256] # MLP head sizes after CNN flatten

# ── Advanced DQN improvements ─────────────────────────────────

# n_step: n-step returns — most impactful fix for sparse Go rewards.
#   Standard DQN bootstraps 1 step ahead — signal barely propagates
#   over 300-move games. N-step accumulates n actual rewards before
#   bootstrapping, connecting actions to outcomes n moves ahead.
#   Try: 10 → 20 if win rate still struggles.
N_STEP          = 10

# double_dqn: prevents Q-value overestimation.
#   Online net selects action, target net evaluates it.
#   Reduces the Q-value explosion risk seen in earlier runs.
DOUBLE_DQN      = True
