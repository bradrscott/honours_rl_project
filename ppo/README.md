# PPO

The flat baseline agent - a single actor-critic network trained with Proximal Policy Optimization (PPO)

## Files

- `ppo_go.py` — the whole agent - the go_v5 environment wrapper, the actor-critic network (shared CNN trunk with separate policy/value heads) and the `train()` loop
- `config_ppo_go.py` — all settings for a run.
- `__init__.py` — package marker.

## How to approach this folder

Start at `config_ppo_go.py` to see what a run is configured with then `ppo_go.py` - which has the `GoEnv` wrapper for how the agent plays go_v5, the `ActorCritic` class for the network, and `train()` for the rollout.