# Phase-1 results

Phase-1 measures each agent's per-opponent win-rate baseline (train PPO and FuN
from scratch against each of the five fixed opponents, on 9x9 and 13x13).

## What's here

- `phase1_win_rates.csv` — final settled win rate (%) per agent x opponent x board.
  This is the Phase-1 baseline that Phase-2 recovery is measured against.

## Raw training data

Phase-1 raw training logs (win rate, losses, entropy, etc. over training steps) are not stored in this repo — they were logged to Weights & Biases during training. The complete set of curves for every run is in the W&B report below:

https://api.wandb.ai/links/bradrscott4-university-of-cape-town/fct92j4k
