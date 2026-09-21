# Opponents

The five fixed heuristic opponents both agents are trained against plus the
shared tactics engine and the factory that builds them.

## Files

- `factory.py` — builds an opponent by name.
- `tactics.py` — the shared tactics engine (board analysis, move evaluation, influence maps) used by the four strategic bots below.
- `greedyOpponent.py` — Greedy - plays the largest open point each move (territory focus)
- `defensiveOpponent.py` - Defensive - connects and defends its own groups, values liberties, avoids needless contact.
- `cornerOpponent.py` — Corner - reasonable contact play with a positional bias toward the corner.
- `edgeOpponent.py` — Edge - the same contact play with a positional bias toward the nearest edge.
- `randomOpponent.py` — Random - a uniformly random legal move each turn — the zero-strategy baseline.
- `__init__.py` — package marker.

## How to approach this folder

Start at `factory.py` to see the five names and how one is built. The four strategic bots (`greedy`, `defensive`, `corner`, `edge`) all lean on `tactics.py` for board analysis and move scoring, so read that next to understand how they choose moves. `randomOpponent.py` has no strategy — it is the baseline.