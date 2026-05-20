# ══════════════════════════════════════════════════════════════
# FEUDAL CONFIG
# Edit values here before a run. Import into feudalAgent.py with:
#
#   from config_feudal import *
#
# Then remove or comment out the matching constants at the top
# of feudalAgent.py so these values take over.
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

# learning_rate: shared across manager and worker.
#   FuN is more sensitive to this than PPO — two loss streams compound errors.
#   Too high → both worker and manager losses diverge (unlearning).
#   Try: 3e-4 → 1e-4 → 5e-5 if you see reward/win rate dropping.
LEARNING_RATE   = 3e-4

# hidden_dim_manager (d): manager LSTM hidden dimension.
#   Too small → manager can't model board-level strategy, cosines stay low.
#   Try: 256 → 512 if manager/cosines graph doesn't trend up.
HIDDEN_DIM_M    = 256

# hidden_dim_worker (k): worker embedding dimension.
#   Increasing this scales compute — only touch if worker loss won't converge.
#   Try: 16 → 32.
HIDDEN_DIM_W    = 16

# time_horizon (c): how many steps the manager commits to a goal.
#   Too short → goals change too fast for worker to follow, cosines stay near 0.
#   Too long  → manager is too slow to adapt mid-game.
#   Try: 10 → 15 for Go (long games need longer manager horizon).
TIME_HORIZON    = 10

# dilation (r): dilated LSTM radius — sets manager's effective memory window.
#   Should be close to or equal to TIME_HORIZON.
#   Try: keep in sync with TIME_HORIZON.
DILATION        = 10

# eps: probability of a random goal (manager exploration).
#   Higher early in training = more state space explored by worker.
#   Too high late in training → manager never commits to a useful goal.
#   Try: 0.1 → 0.2 early, then anneal down manually once win rate rises.
EPS             = 0.1

# alpha: intrinsic reward mixing coefficient.
#   Controls how strongly the worker is pulled toward manager goals.
#   Too low  → worker ignores manager, hierarchical structure breaks down.
#   Too high → worker over-constrained, can't react to local board threats.
#   Try: 0.5 → 0.7 if cosines are low (worker not following goals).
ALPHA           = 0.5

# gamma_manager: manager discount factor.
#   Keep high — manager needs to value game outcomes far in the future.
#   Try: 0.99 → 0.995 if manager value loss won't converge.
GAMMA_M         = 0.99

# gamma_worker: worker discount factor.
#   Slightly lower than manager so worker focuses on immediate sub-goals.
#   Try: 0.95 → 0.97 → 0.90 depending on worker value loss behaviour.
GAMMA_W         = 0.95

# entropy_coef: entropy bonus for worker action distribution.
#   Critical in hierarchical agents — worker entropy collapsing is common.
#   Try: 0.01 → 0.02 → 0.05 if worker entropy graph drops sharply.
ENTROPY_COEF    = 0.01

# num_steps (K): steps collected per rollout.
#   Go has sparse, delayed rewards — more steps = better return estimates.
#   Try: 400 → 600 → 800 if losses are very noisy.
NUM_STEPS       = 400

# grad_clip: max norm for gradient clipping.
#   Prevents exploding gradients, especially important in hierarchical agents.
#   Too high → gradients can explode, losses diverge.
#   Too low  → updates become too small, learning slows.
#   Try: 0.5 → 0.25 if losses are spiking or diverging.
GRAD_CLIP       = 0.5
