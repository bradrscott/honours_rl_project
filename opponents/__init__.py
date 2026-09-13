"""Opponents package: the 5 Go heuristic bots, the shared tactics engine,
and the runtime factory. Import the factory directly from the package:

    from opponents import make_opponent
"""
from opponents.factory import make_opponent, OPPONENT_NAMES

__all__ = ["make_opponent", "OPPONENT_NAMES"]
