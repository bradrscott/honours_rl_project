# ══════════════════════════════════════════════════════════════
# PHASE 2 — abrupt opponent-shift & recovery-time machinery (RQ3)
#
# Shared by ppo_go.py and feudalAgent.py so BOTH agents are measured
# with byte-identical logic. Three pieces:
#
#   ShiftManager    — parses a shift schedule ("edge@300000,corner@800000")
#                     and says when to swap the opponent. Checked at game
#                     boundaries only, so the opponent is constant within
#                     a game. Empty schedule => inert (Phase-1 mode).
#
#   RecoveryTracker — ONE continuous rolling window of W completed games
#                     (never reset; same window is the win-rate signal,
#                     the baseline and the recovery detector — consistent
#                     across agents/conditions/shifts, per the design).
#                     recovery time (games) = completed games after a
#                     shift until the rolling win rate is back >= 80% of
#                     the baseline (its value at the moment of the shift),
#                     requiring the window to have fully turned over
#                     (games_since >= W) so pre-shift games can't fake it.
#
#   GameLog         — one CSV row per completed game. The durable source
#                     for the paper's recovery plots (x-axis = games since
#                     shift), independent of wandb's step axis.
# ══════════════════════════════════════════════════════════════

import csv
import json
import os
from collections import deque

RECOVERY_FRAC = 0.8   # "within 80% of the Phase-1 baseline" (proposal 5.3.5)


def parse_schedule(s):
    """'edge@300000,corner@800000' -> [(300000, 'edge'), (800000, 'corner')].
    Empty/blank string -> [] (no shifts; Phase-1 behaviour)."""
    if not s or not s.strip():
        return []
    out = []
    for item in s.split(","):
        name, at = item.strip().split("@")
        out.append((int(at), name.strip()))
    out.sort(key=lambda x: x[0])
    return out


class ShiftManager:
    def __init__(self, schedule_str):
        self.schedule = parse_schedule(schedule_str)
        self._idx = 0

    @property
    def active(self):
        return bool(self.schedule)

    def check(self, global_step):
        """Return the new opponent name if a shift boundary has been
        crossed, else None. Call at episode end."""
        if self._idx < len(self.schedule) and global_step >= self.schedule[self._idx][0]:
            name = self.schedule[self._idx][1]
            self._idx += 1
            return name
        return None


class RecoveryTracker:
    def __init__(self, window):
        self.W = window
        self.window = deque(maxlen=window)   # the ONE window — never reset
        self.game_idx = 0
        self.shifts = []                     # one record per shift event

    def rolling(self):
        return float(sum(self.window) / len(self.window)) if self.window else 0.0

    def on_shift(self, global_step, new_opponent):
        """Snapshot the baseline = current rolling value. Window continues."""
        self.shifts.append({
            "shift_idx":      len(self.shifts),
            "step":           int(global_step),
            "to_opponent":    new_opponent,
            "game_at_shift":  self.game_idx,
            "baseline":       self.rolling(),
            "min_rolling":    self.rolling(),
            "recovery_games": None,
        })

    def on_game(self, win):
        """Record a completed game. Returns the rolling win rate."""
        self.game_idx += 1
        self.window.append(1.0 if win else 0.0)
        r = self.rolling()
        if self.shifts:
            s = self.shifts[-1]
            if s["recovery_games"] is None:
                s["min_rolling"] = min(s["min_rolling"], r)
                games_since = self.game_idx - s["game_at_shift"]
                # full window turnover required, so lingering pre-shift wins
                # in the continuous window can't count as "recovered"
                if games_since >= self.W and r >= RECOVERY_FRAC * s["baseline"]:
                    s["recovery_games"] = games_since
        return r

    def games_since_shift(self):
        return (self.game_idx - self.shifts[-1]["game_at_shift"]) if self.shifts else 0

    def summary(self):
        """Per-shift results incl. performance drop, for end-of-run dump."""
        out = []
        for s in self.shifts:
            out.append({**s, "perf_drop": s["baseline"] - s["min_rolling"]})
        return out

    def save_summary(self, path):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)


class GameLog:
    """Appends one row per completed game to <save_dir>/games.csv."""

    HEADER = ["game_idx", "global_step", "opponent", "win",
              "rolling", "shift_idx", "games_since_shift"]

    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, "games.csv")
        new = not os.path.exists(self.path)
        self._f = open(self.path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(self.HEADER)

    def log(self, game_idx, global_step, opponent, win, rolling,
            shift_idx, games_since_shift):
        self._w.writerow([game_idx, global_step, opponent, int(win),
                          f"{rolling:.4f}", shift_idx, games_since_shift])
        self._f.flush()   # survive walltime kills

    def close(self):
        self._f.close()
