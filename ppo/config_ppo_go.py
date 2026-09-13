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
import os   # needed for the env-overridable board/komi below
# BOARD_SIZE / KOMI env-overridable so ONE codebase runs 9x9 AND 13x13.
# Defaults = 13x13. For 9x9:  BOARD_SIZE=9 KOMI=5.5 OPPONENT=greedy python ppo/ppo_go.py
BOARD_SIZE      = int(os.environ.get("BOARD_SIZE", 13))
KOMI            = float(os.environ.get("KOMI", 7.5))

# Max PLIES (half-moves: our move + opponent move) before a game is
# force-ended. go_v5 only terminates on two consecutive passes and has NO
# move limit, so a competitive game vs a never-passing opponent can run to
# 30k+ moves. At the cap we force passes so go_v5 area-scores the board.
#
# IMPORTANT: this counts PLIES, so the per-player move cap is MAX_MOVES/2.
# It must sit ABOVE the natural game length, or it slices games off before
# the agent can play them out and win (4*N*N=676 plies = only 338 of our
# moves cut below the ~560-move natural aggressive game -> agent got 0% and
# could not bootstrap). Set generously above that; only the 15k runaway is
# cut. Lower it later, once the agent is clearly learning, to tighten games.
MAX_MOVES       = 16 * BOARD_SIZE * BOARD_SIZE   # 2704 plies ~= 1352 of our moves

# Opponent: "random" | "greedy" | "aggressive" | "defensive" | "corner" | "edge"
# Set ENTIRELY from the environment — the SLURM array (run_ppo_go_array.sh)
# exports OPPONENT per task, so every run names its opponent explicitly and no
# opponent is a special default. To run one manually, export it first:
#   OPPONENT=random python ppo_go.py
import os
OPPONENT        = os.environ["OPPONENT"]

# Opponent difficulty — STRICT (eps=0) for every bot: each plays purely by its
# own rules, no injected randomness. Same for PPO and feudal.
OPPONENT_EPSILON = 0.0

# ── Training ──────────────────────────────────────────────────
# TOTAL_TIMESTEPS is env-overridable so Phase-2 runs can use a shorter
# budget (2.3M) without touching this file. Default = Phase-1's 5M.
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 5_000_000))
SAVE_EVERY      = 200_000

# ── PHASE 2 (RQ3 shift/recovery experiments) — ALL default OFF ────
# With these unset a run behaves EXACTLY like Phase 1 (fresh network,
# fixed opponent, no per-game CSV). run_phase2_ppo_array.sh sets them.
RESUME_FROM     = os.environ.get("RESUME_FROM", "")     # Phase-1 ckpt to load; "" = fresh
SHIFT_SCHEDULE  = os.environ.get("SHIFT_SCHEDULE", "")  # e.g. "edge@300000,corner@1800000"; "" = no shifts
RUN_TAG         = os.environ.get("RUN_TAG", "")         # e.g. "phase2-low-f1"; suffixes wandb name + dirs
W_RECOVERY      = 100   # the ONE rolling window (games) used in Phase 2 for the
                        # win rate, the baseline AND recovery detection —
                        # consistent across both agents, all conditions, shifts

# Per-opponent dirs so parallel array jobs never overwrite each other's
# checkpoints. Phase-2 runs get their OWN dirs (RUN_TAG suffix) so they can
# never overwrite the Phase-1 checkpoints they resume from.
_DIR_KEY        = f"{OPPONENT}-{RUN_TAG}" if RUN_TAG else OPPONENT
_BOARD          = f"{BOARD_SIZE}x{BOARD_SIZE}"
# Board size in the path so 9x9 and 13x13 runs NEVER overwrite each other.
# Weights save to phase2_checkpoints/ for Phase-2 runs (RUN_TAG set),
# phase1_checkpoints/ for Phase-1. Both gitignored.
SAVE_DIR        = (f"./phase2_checkpoints/ppo_go/{_BOARD}/{_DIR_KEY}/" if RUN_TAG
                   else f"./phase1_checkpoints/ppo_go/{_BOARD}/{_DIR_KEY}/")
LOG_DIR         = f"./logs/ppo_go/{_BOARD}/{_DIR_KEY}/"   # local dir wandb writes its run files to
SEED            = int(os.environ.get("SEED", 0))   # env-overridable for re-seeding (e.g. SEED=1 to re-sample a run); default 0 keeps Phase-1 reproducible

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
