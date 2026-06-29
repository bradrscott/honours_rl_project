# ══════════════════════════════════════════════════════════════
# PPO CONFIG  (new implementation — ppo_go.py)
#
# Adapted from adi3e08/PPO (clean minimal PPO) but rebuilt for the
# discrete, action-masked, CNN setting that Go requires.
#
# Import into ppo_go.py with:
#   from config_ppo_go import *
#
# NOTE: this is the NEW pipeline. The old config_ppo.py / ppoAgent.py
# (19x19, stable-baselines3 MaskablePPO) are left untouched.
# ══════════════════════════════════════════════════════════════

# ── Environment ───────────────────────────────────────────────
# Start SMALL. 19x19 from scratch with model-free PPO does not learn
# in any realistic budget. On 7x7 vs a random opponent the win rate
# climbs visibly — which is the goal here (demonstrate learning).
# Scale up (9 -> 13 -> 19) only once 7x7 clearly works.
BOARD_SIZE      = 13
KOMI            = 7.5

# Opponent: "random" | "greedy" | "aggressive" | "defensive" | "corner" | "edge"
# Change THIS line to switch opponent, then re-upload + resubmit the job.
OPPONENT        = "edge"

# Opponent difficulty — fraction of the opponent's moves that are random.
# FIXED AT 0.0 (STRICT): every strategic bot plays purely by its own rules,
# at full strength, never a random move. Each opponent tries to beat the
# agent through its distinct strategy, not via injected noise. The SAME
# strict setting is shared by the flat (PPO) and feudal agents so the
# comparison is fair. (Does not apply to the random opponent.)
OPPONENT_EPSILON = 0.0

# ── Training ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 5_000_000
SAVE_EVERY      = 200_000
# Per-opponent dirs so parallel array jobs never overwrite each other's
# checkpoints (e.g. ./models/ppo_go/aggressive/ppo_go_final.pt).
SAVE_DIR        = f"./models/ppo_go/{OPPONENT}/"
LOG_DIR         = f"./logs/ppo_go/{OPPONENT}/"   # local dir wandb writes its run files to
SEED            = 0

# ── Logging (Weights & Biases) ────────────────────────────────
# All metrics go to wandb — no TensorBoard. Every scalar below shows
# up as its own panel in the wandb dashboard:
#   custom/win_rate, custom/total_episodes,
#   rollout/ep_rew_mean, rollout/ep_len_mean,
#   train/policy_loss, train/value_loss, train/entropy,
#   train/approx_kl, train/clip_fraction, train/explained_var,
#   train/early_stopped
WANDB_PROJECT   = "honours-rl-go"

# ── PPO hyperparameters ───────────────────────────────────────
LEARNING_RATE   = 3e-4
N_STEPS         = 2048    # transitions collected per policy update
BATCH_SIZE      = 256
N_EPOCHS        = 4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
ENT_COEF        = 0.01    # small exploration bonus (reference used 0)
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5
TARGET_KL       = 0.03    # early-stop a PPO update if KL exceeds 1.5x this

# ── CNN architecture ──────────────────────────────────────────
# Observation is (BOARD_SIZE, BOARD_SIZE, 17). Treated as a spatial
# grid (17 channels) -> shared conv trunk -> separate policy/value heads.
N_CHANNELS      = 17
CNN_FILTERS     = 32
CNN_LAYERS      = 3
HIDDEN_DIM      = 128     # size of the shared FC layer after the conv trunk

# ── Logging ───────────────────────────────────────────────────
WINDOW          = 200     # rolling window (games) for the win-rate metric
