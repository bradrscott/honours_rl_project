# ══════════════════════════════════════════════════════════════
# PPO CONFIG
# Edit values here before a run. Import into ppoAgent.py with:
#
#   from config_ppo import *
# ══════════════════════════════════════════════════════════════

# ── Environment ───────────────────────────────────────────────
BOARD_SIZE      = 19
KOMI            = 0.0    # reduced from 7.5 — greedy bot already has
                         # attacking advantage; komi=7.5 made winning
                         # impossible for a learning agent from scratch

# ── Training ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 10_000_000
SAVE_EVERY      = 500_000
SAVE_DIR        = "./models/ppo/"
LOG_DIR         = "./logs/ppo/"

# ── PPO hyperparameters ───────────────────────────────────────
LEARNING_RATE   = 5e-5
N_STEPS         = 2048
BATCH_SIZE      = 128
N_EPOCHS        = 3
GAMMA           = 0.995
GAE_LAMBDA      = 0.90
CLIP_RANGE      = 0.15
ENT_COEF        = 0.05
VF_COEF         = 1.0

# ── CNN architecture ──────────────────────────────────────────
# The observation (19,19,17) is treated as a spatial grid.
# CNN learns spatial patterns: adjacency, groups, territory, atari.
# This is the standard approach for Go RL (AlphaGo, KataGo etc).
#
# CNN_FILTERS: number of filters per conv layer.
#   More filters = more patterns learned. 64 is a practical start.
#   Try: 32 → 64 → 128 if win rate plateaus.
CNN_FILTERS     = 64

# CNN_LAYERS: number of conv layers.
#   More layers = more complex spatial reasoning.
#   5 layers gives meaningfully better spatial understanding than 3
#   without being too slow for a 10M step budget.
CNN_LAYERS      = 5

# CNN_FEATURES: output dimension of the CNN feature extractor.
#   Fed into the policy and value heads.
CNN_FEATURES    = 256

# NET_ARCH: MLP head sizes after the CNN features.
NET_ARCH        = [256]

# ── Opponent ──────────────────────────────────────────────────
# Options: "random", "greedy", "aggressive"
OPPONENT        = "aggressive"
