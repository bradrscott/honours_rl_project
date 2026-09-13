# ── Opponent factory (shared by ppo_go.py and feudalAgent.py) ─────
#
# Single place that maps an opponent NAME to a constructed opponent
# instance. Replaces the duplicated import-time if/elif switch in both
# training scripts, and — crucially for Phase 2 — allows the opponent
# to be swapped at RUNTIME mid-training (GoEnv.set_opponent), which an
# import-time switch cannot do.
#
# NO opponent behaviour is changed here: same classes, same EPSILON
# mechanism (module-level attribute, exactly what the old importlib
# block in train() did). The bots themselves stay frozen.

import importlib

_OPPONENTS = {
    "greedy":     ("greedyOpponent",     "GreedyOpponent"),
    "defensive":  ("defensiveOpponent",  "DefensiveOpponent"),
    "corner":     ("cornerOpponent",     "CornerOpponent"),
    "edge":       ("edgeOpponent",       "EdgeOpponent"),
    "random":     ("randomOpponent",     "RandomOpponent"),
}

OPPONENT_NAMES = list(_OPPONENTS.keys())


def make_opponent(name, board_size, epsilon=None):
    """Construct an opponent by name.

    epsilon: if given, sets the module-level EPSILON of the strategic
    bots (identical mechanism/effect to the old importlib block).
    "random" has no EPSILON knob — it is the zero-strategy baseline.
    """
    if name not in _OPPONENTS:
        raise ValueError(f"Unknown opponent '{name}'. Valid: {OPPONENT_NAMES}")
    module_name, class_name = _OPPONENTS[name]
    module = importlib.import_module(f"opponents.{module_name}")
    if epsilon is not None and name != "random":
        module.EPSILON = epsilon
    cls = getattr(module, class_name)
    return cls(board_size=board_size)
