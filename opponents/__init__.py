# Opponents package - the 5 Go heuristic bots, the shared tactics engine and
# the runtime factory.
from opponents.factory import make_opponent, OPPONENT_NAMES

# names exported when a caller does `from opponents import *`
__all__ = ["make_opponent", "OPPONENT_NAMES"]
