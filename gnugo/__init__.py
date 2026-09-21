# GNU Go external-baseline package
# GNU Go (the GPL C engine, spoken to over GTP) plays the black seat against
# each heuristic opponent, with go_v5 as referee to establish a fixed
# strength baseline for the trained agents to be compared against.
from gnugo.engine import GnuGo

# names exported when a caller does from gnugo import *
__all__ = ["GnuGo"]
