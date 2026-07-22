"""GNU Go external-baseline package (RQ3).

GNU Go (the GPL C engine, spoken to over GTP) plays the black seat against
each heuristic opponent, with go_v5 as referee, to establish a fixed
strength baseline for the trained agents to be compared against.

    from gnugo.engine import GnuGo      # GTP-backed engine wrapper
    # run:  python gnugo/run_baseline.py --games 20 --level 10
"""
from gnugo.engine import GnuGo

__all__ = ["GnuGo"]
