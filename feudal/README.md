# Feudal

The hierarchical baseline agent - a manager-worker network (FuN)

## Files

- `feudalAgent.py` — Trains the agent against its configured opponent and writes checkpoints + W&B logs.
- `feudalNetwork.py` — the manager, worker and full FeudalNetwork model plus the FuN loss.
- `dilated_lstm.py` — the manager's dilated LSTM (its long-horizon memory).
- `config_feudal.py` — all settings for a run (board, opponent, hyperparameters).
- `__init__.py` — package marker.

## How to approach this folder

Start at `config_feudal.py` to see what a run is configured with then
`feudalNetwork.py` to understand the architecture (perception, manager,
worker), then `feudalAgent.py`'s `train()` for the actual training loop.
`dilated_lstm.py` is only worth reading if you want the manager's memory
mechanism in detail.
