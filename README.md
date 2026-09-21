# Does Hierarchical Feudal Reinforcement Learning Recover Faster Than Flat PPO After Abrupt Opponent Strategy Shifts?

Honours research project comparing a hierarchical Feudal Network (FuN) against
a flat PPO agent, testing how quickly each recovers after an opponent
abruptly changes strategy in the board game Go.

## Folders

- `ppo/` — the flat baseline agent (PPO).
- `feudal/` — the hierarchical agent (FuN).
- `opponents/` — the five fixed heuristic opponents the agents train against.
- `gnugo/` — the GNU Go external-reference baseline.
- `phase2/` — the runtime machinery for Phase 2 (opponent shift + recovery tracking during training).
- `results/` — all experiment data - the GNU Go baselines, Phase-1 win rates and Phase-2 per-game logs and recovery summaries.
- `runScripts/` — batch scripts to run every experiment on the UCT and CHPC clusters.
- `paper/` — the final submitted paper.
- `requirements/` — the Python dependencies.

Trained model weights are not stored in this repo (available on request). The complete training and evaluation curves for every run are on Weights & Biases: <https://api.wandb.ai/links/bradrscott4-university-of-cape-town/fct92j4k>
