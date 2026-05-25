# ══════════════════════════════════════════════════════════════
# PPO CONFIG
# Edit values here before a run. Import into ppoAgent.py with:
#
#   from config_ppo import *
# ══════════════════════════════════════════════════════════════

# ── Environment ───────────────────────────────────────────────
BOARD_SIZE      = 19

# ── Training ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 20_000_000
SAVE_EVERY      = 500_000
SAVE_DIR        = "./models/ppo/"
LOG_DIR         = "./logs/ppo/"

# ── PPO hyperparameters ───────────────────────────────────────

# learning_rate: reduced to 5e-5 to compensate for the larger
#   [256,256] network making bigger parameter updates.
#   Keeps clip_fraction stable around 0.06.
LEARNING_RATE   = 1e-4

# n_steps: covers ~10 full Go games per update for better
#   credit assignment with sparse rewards.
N_STEPS         = 2048

# batch_size: must divide evenly into n_steps.
BATCH_SIZE      = 128

# n_epochs: keep low to prevent KL overshoot.
N_EPOCHS        = 5

# gamma: high for long Go episodes so terminal reward
#   propagates back through the whole game.
GAMMA           = 0.995

# gae_lambda: lower than default (0.95) for sparse rewards —
#   shorter effective horizon gives more reliable advantage estimates.
GAE_LAMBDA      = 0.90

# clip_range: tighter clip to prevent large policy jumps.
CLIP_RANGE      = 0.15

# ent_coef: entropy bonus to keep exploration alive.
ENT_COEF        = 0.02

# vf_coef: increased from default 0.5 to give value function
#   more training signal — fixed the explained_variance problem.
VF_COEF         = 1.0

# net_arch: larger network than SB3 default [64,64] to handle
#   the 6137-dim Go observation. Fixed explained_variance near 0.
NET_ARCH        = [256, 256]
