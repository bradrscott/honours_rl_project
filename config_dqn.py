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

# ── Training ──────────────────────────────────────────────────
TOTAL_TIMESTEPS = 10_000_000
SAVE_EVERY      = 500_000
SAVE_DIR        = "./models/dqn/"
LOG_DIR         = "./logs/dqn/"

# ── DQN hyperparameters ───────────────────────────────────────

# learning_rate: same as PPO for fair comparison.
LEARNING_RATE   = 1e-4

# buffer_size: number of transitions in replay buffer.
#   50k * 6137 floats * 4 bytes * 2 (obs + next_obs) ≈ 2.4GB.
#   Reduce to 20_000 if HPC runs out of memory.
BUFFER_SIZE     = 50_000

# learning_starts: random steps before learning begins.
#   Fills buffer with diverse experiences first.
LEARNING_STARTS = 10_000

# batch_size: transitions sampled per gradient update.
BATCH_SIZE      = 64

# gamma: discount factor. Same as PPO — Go needs long horizon.
GAMMA           = 0.995

# train_freq: update Q-network every N environment steps.
TRAIN_FREQ      = 4

# target_update_freq: steps between target network syncs.
#   Too low → unstable. Too high → slow learning.
TARGET_UPDATE_FREQ = 1_000

# epsilon_start: initial exploration rate.
EPSILON_START   = 1.0

# epsilon_end: final exploration rate after decay.
EPSILON_END     = 0.05

# epsilon_decay_steps: steps to decay epsilon from start to end.
#   10% of total timesteps.
EPSILON_DECAY_STEPS = 1_000_000

# net_arch: hidden layer sizes — matches PPO for fair comparison.
NET_ARCH        = [256, 256]
