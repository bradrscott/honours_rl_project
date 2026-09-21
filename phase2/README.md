# Phase 2

The runtime machinery for Phase 2  - triggers an opponent switch mid-training on a fixed schedule, tracks the rolling win
rate, and logs every game. 

## Files

- `phase2.py` — `parse_schedule` reads the `SHIFT_SCHEDULE` env string, `ShiftManager` - starts the opponent switch at the right step, `RecoveryTracker` rolling win rate, baseline and recovery per shift, `GameLog` writes one row per game to `games.csv`.
- `__init__.py` — package marker

## How to approach this folder

Start at `parse_schedule` to see how a schedule string is read then `ShiftManager` for when a shift
start. `RecoveryTracker` is the rolling-window win rate and recovery logic used live during training  `GameLog` is just the per-game CSV writer.

