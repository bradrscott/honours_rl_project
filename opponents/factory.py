# Opponent factory 
# One place that maps an opponent name to a constructed opponent instance.
# Phase 2, it lets the opponent be swapped at runtime mid-training (GoEnv.set_opponent)

# importlib lets us load each opponent's module by name
import importlib

# name - (module file, class name) for the five fixed opponents
_OPPONENTS = {
    "greedy": ("greedyOpponent", "GreedyOpponent"),
    "defensive": ("defensiveOpponent", "DefensiveOpponent"),
    "corner": ("cornerOpponent", "CornerOpponent"),
    "edge": ("edgeOpponent", "EdgeOpponent"),
    "random": ("randomOpponent", "RandomOpponent"),
}

# the valid opponent names (used for validation)
OPPONENT_NAMES = list(_OPPONENTS.keys())

# construct an opponent by name 
def make_opponent(name, board_size, epsilon=None):

    # reject unknown names up front
    if name not in _OPPONENTS:
        raise ValueError(f"Unknown opponent '{name}'. Valid: {OPPONENT_NAMES}")
    
    # look up and import the requested opponent's module
    module_name, class_name = _OPPONENTS[name]
    module = importlib.import_module(f"opponents.{module_name}")

    # apply the difficulty knob to the strategic bots (not the random baseline)
    if epsilon is not None and name != "random":
        module.EPSILON = epsilon
        
    # build and return the opponent instance for this board size
    cls = getattr(module, class_name)
    return cls(board_size=board_size)
