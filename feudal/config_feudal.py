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
BOARD_SIZE      = 13
KOMI            = 7.5
N_ACTIONS       = BOARD_SIZE * BOARD_SIZE + 1     # 170 on 13x13

# Max plies before a game is force-ended (see config_ppo_go.py for the full
# reasoning). Counts PLIES; per-player cap is MAX_MOVES/2. Set above natural
# game length so games aren't sliced off before the agent can win. MUST match
# the PPO setting for a fair comparison.
MAX_MOVES       = 16 * BOARD_SIZE * BOARD_SIZE   # 2704 plies ~= 1352 of our moves

# Opponent: "random" | "greedy" | "aggressive" | "defensive" | "corner" | "edge"
# Set ENTIRELY from the environment — the SLURM array (run_feudal_array.sh)
# exports OPPONENT per task, so every run names its opponent explicitly and no
# opponent is a special default. To run one manually, export it first:
#   OPPONENT=random python feudalAgent.py
import os
OPPONENT        = os.environ["OPPONENT"]

# Opponent difficulty — STRICT (eps=0) for every bot. MUST match PPO.
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
# TOTAL_TIMESTEPS is env-overridable so Phase-2 runs can use a shorter
# budget (2.3M) without touching this file. Default = Phase-1's 5M.
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 5_000_000))
SAVE_EVERY      = 200_000
SEED            = 0
WANDB_PROJECT   = "honours-rl-go"
WINDOW          = 200     # rolling window (games) for the win-rate metric

# ── PHASE 2 (RQ3 shift/recovery experiments) — ALL default OFF ────
# Identical to config_ppo_go.py so both agents are measured the same way.
# With these unset a run behaves EXACTLY like Phase 1.
RESUME_FROM     = os.environ.get("RESUME_FROM", "")     # Phase-1 ckpt to load; "" = fresh
SHIFT_SCHEDULE  = os.environ.get("SHIFT_SCHEDULE", "")  # e.g. "edge@300000,corner@1800000"; "" = no shifts
RUN_TAG         = os.environ.get("RUN_TAG", "")         # e.g. "phase2-low-f1"; suffixes wandb name + dirs
W_RECOVERY      = 100   # the ONE rolling window (games) — matches PPO's

# Per-opponent dirs so different opponent runs never overwrite each other.
# Phase-2 runs get their OWN dirs (RUN_TAG suffix) so they can never
# overwrite the Phase-1 checkpoints they resume from.
_DIR_KEY        = f"{OPPONENT}-{RUN_TAG}" if RUN_TAG else OPPONENT
SAVE_DIR        = f"./models/feudal/{_DIR_KEY}/"
LOG_DIR         = f"./logs/feudal/{_DIR_KEY}/"

# ══════════════════════════════════════════════════════════════
# FuN hyperparameters (architecture-specific — NOT shared with PPO)
# ══════════════════════════════════════════════════════════════

# learning_rate: Adam, 2e-4. Bumped from 1e-4 to learn faster within the tight
# 5M-step budget (small for FuN). Kept BELOW the earlier 3e-4 that made the win
# rate decline, and below the reference's RMSprop 5e-4 (not comparable — we use
# Adam + a single worker, so gradients are noisier). If a run COLLAPSES (win
# rate crashes to 0 and stays, or loss -> NaN), drop back to 1e-4.
LEARNING_RATE   = 2e-4

# hidden_dim_manager (d): manager/perception latent dimension.
HIDDEN_DIM_M    = 256

# hidden_dim_worker (k): worker embedding dimension. Worker LSTM is
# LSTMCell(d, k*n_actions); on 13x13 that is LSTMCell(256, 16*170=2720).
HIDDEN_DIM_W    = 16

# time_horizon (c): how many steps the manager commits to a goal.
# 15 for 13x13 (games are longer than 9x9). Try 10 <-> 20 if
# manager/cosines stay flat near 0.
TIME_HORIZON    = 15

# dilation (r): dilated LSTM radius. Keep in sync with TIME_HORIZON.
DILATION        = 15

# eps: probability of a random goal (manager exploration).
EPS             = 0.1

# alpha: intrinsic-reward mixing weight in the worker advantage. 0.5 matches the
# reference (lweitkamp) — was 0.7, which over-weighted the intrinsic goal-
# following signal relative to actually winning.
ALPHA           = 0.5

# gamma_manager / gamma_worker: separate discount factors, aligned to the
# reference (lweitkamp: gamma_m=0.999, gamma_w=0.99). We use a slightly lower
# manager gamma (0.995) because we train with ONE worker (the reference uses 16
# parallel workers, which averages out the higher-variance long-horizon return);
# 0.995 is a safer horizon for our single-worker, noisier gradient estimate.
GAMMA_M         = 0.995
GAMMA_W         = 0.99

# gae_lambda: GAE(lambda) trace decay (matches PPO's 0.95). This is the credit-
# assignment mechanism PPO uses to break through vs aggressive (~70%); feudal's
# old Monte-Carlo returns were too high-variance and left it at 0. GAE is gated
# by the per-step terminal flag `nt` (see feudalAgent) so it does not leak value
# across episode boundaries — the bug that made a previous GAE attempt oscillate.
GAE_LAMBDA      = 0.95

# entropy_coef: worker action-distribution entropy bonus. 0.01 matches the
# reference (lweitkamp) and PPO. (An earlier 0.005 was a speculative tweak;
# reverted to the proven value.)
ENTROPY_COEF    = 0.01

# num_steps (K): steps collected per rollout / update. Lowered 600->400 now
# that BPTT flows through the whole rollout (per-step repackage removed): this
# bounds how deep the retained LSTM graph gets (memory/OOM safety) and gives
# more frequent updates. If a run OOMs on the GPU, drop this further (300/256).
NUM_STEPS       = 400

# grad_clip: max gradient norm.
GRAD_CLIP       = 0.5
