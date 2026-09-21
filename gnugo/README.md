# GNU Go

The external-reference baseline - GNU Go, an established rule-based Go engine
that plays without learning. It is played against each fixed opponent to show
how a competent engine scores on the same task the trained agents face.

## Files

- `run_baseline.py` - plays GNU Go (black) against one heuristic opponent, referees the game with go_v5, and writes the per-opponent win-rate CSV + W&B log.
- `engine.py` - the `GnuGo` wrapper that lets GNU Go play inside a go_v5 game, converting moves between go_v5's action numbers and GTP's column-letter/row vertex names.
- `gtp.py` - a minimal Go Text Protocol (GTP) client that launches GNU Go as a subprocess and exchanges commands with it (`GTPEngine`, and the `GTPError` type).
- `__init__.py` — package marker.

## How to approach this folder

Start at `run_baseline.py`'s `main()` for the overall loop. `engine.py` is the
go_v5 GNU Go bridge. `gtp.py` is the low-level protocol layer underneath it, only worth reading if you want the raw GTP details.

