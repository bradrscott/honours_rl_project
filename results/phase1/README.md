# Phase-1 results

Phase-1 measures each agent's per-opponent win-rate baseline (train PPO and FuN
from scratch against each of the five fixed opponents, on 9x9 and 13x13).

## What's here

- `phase1_win_rates.csv` — final settled win rate (%) per agent x opponent x board.
  This is the Phase-1 baseline that Phase-2 recovery is measured against.

## Raw training data

Phase-1 raw training curves (win rate, losses, entropy, etc. over training steps)
are **not stored locally** — they were logged to Weights & Biases during training:

- Entity:  `bradrscott4-university-of-cape-town`
- Project: `honours-rl-go`

The training-curve and internals figures in the paper are pulled directly from
W&B by the scripts in `paperMaterials/materialScripts/` (`make_phase1_curves.py`,
`make_phase1_internals.py`, `make_phase1_results.py`).
