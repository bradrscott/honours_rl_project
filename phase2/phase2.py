# Phase 2 - the opponent-shift and recovery-tracking logic shared by
# ppo_go.py and feudalAgent.py, so both agents are measured the same way.
# ShiftManager decides when to swap the opponent, RecoveryTracker measures
# how the win rate responds and GameLog writes the per-game record both
# of those are computed from.


# standard librarys
import csv
import os
from collections import deque

# within 80% of the Phase-1 baseline 
RECOVERY_FRAC = 0.8  


# parse a schedule string into a sorted list of (step, opponent) pairs 
# an empty/blank string means no shifts at all (Phase-1 behaviour)
def parse_schedule(s):
    if not s or not s.strip():
        return []
    out = []
    for item in s.split(","):
        name, at = item.strip().split("@")
        out.append((int(at), name.strip()))
    out.sort(key=lambda x: x[0])
    return out


# Decides when the opponent should change during Phase 2 by walking through
# a fixed schedule of (step, opponent) pairs. Does nothing at all in Phase 1
class ShiftManager:

    # parse the schedule and start at the first entry
    def __init__(self, schedule_str):
        self.schedule = parse_schedule(schedule_str)
        self._idx = 0

    # @property lets every caller read this as shifter.active instead of shifter.active()
    # that's what all the call sites in ppo_go.py/feudalAgent.py expect. 
    # True if this run has any shifts scheduled at all (False = Phase 1).
    @property
    def active(self):
        return bool(self.schedule)

    # call at the end of each game - returns the new opponent's name if
    # global_step has just crossed the next shift boundary, else None
    def check(self, global_step):
        if self._idx < len(self.schedule) and global_step >= self.schedule[self._idx][0]:
            name = self.schedule[self._idx][1]
            self._idx += 1
            return name
        return None


# Tracks the rolling win rate over one continuous window of W games (the
# same window used as the win-rate signal, the pre-shift baseline and the
# recovery detector) and records how the agent behaves around each shift.
class RecoveryTracker:

    # set up the rolling window and the per-shift record list
    def __init__(self, window):
        self.W = window
        self.window = deque(maxlen=window)   
        self.game_idx = 0
        self.shifts = []                     

    # current rolling win rate (0 if no games recorded yet)
    def rolling(self):
        return float(sum(self.window) / len(self.window)) if self.window else 0.0

    # call when the opponent changes - snapshot the current rolling win rate
    # as this shift's baseline. The window itself is not reset.
    def on_shift(self, global_step, new_opponent):
        self.shifts.append({
            "shift_idx": len(self.shifts),
            "step": int(global_step),
            "to_opponent": new_opponent,
            "game_at_shift": self.game_idx,
            "baseline": self.rolling(),
            "min_rolling": self.rolling(),
            "recovery_games": None,
        })

    # call after every completed game - updates the rolling window and if a
    # shift is in progress it checks whether the agent has now recovered
    def on_game(self, win):
        self.game_idx += 1
        self.window.append(1.0 if win else 0.0)
        r = self.rolling()

        # only the most recent shift can still be in progress
        if self.shifts:
            s = self.shifts[-1]

            # once recovery_games is set for this shift, leave it alone
            if s["recovery_games"] is None:
                s["min_rolling"] = min(s["min_rolling"], r)
                games_since = self.game_idx - s["game_at_shift"]

                # full window turnover required, so lingering pre-shift wins
                # in the continuous window can't count as recovered
                if games_since >= self.W and r >= RECOVERY_FRAC * s["baseline"]:
                    s["recovery_games"] = games_since
        return r

    # games played since the most recent shift (0 if no shift has happened)
    def games_since_shift(self):
        return (self.game_idx - self.shifts[-1]["game_at_shift"]) if self.shifts else 0


# Appends one row per completed game to <save_dir>/games.csv 
class GameLog:
    HEADER = ["game_idx", "global_step", "opponent", "win",
              "rolling", "shift_idx", "games_since_shift"]

    # Open games.csv in append mode so it keeps adding to the existing file instead
    # of erasing it. The header row must only be written once, so a check
    # whether the file already exists before writing it.
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, "games.csv")
        new = not os.path.exists(self.path)
        self._f = open(self.path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(self.HEADER)

    # append one row for a completed game and flush immediately (so data
    # survives a walltime kill)
    def log(self, game_idx, global_step, opponent, win, rolling,
            shift_idx, games_since_shift):
        self._w.writerow([game_idx, global_step, opponent, int(win),
                          f"{rolling:.4f}", shift_idx, games_since_shift])
        self._f.flush()

    # close the underlying file
    def close(self):
        self._f.close()
