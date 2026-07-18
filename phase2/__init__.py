"""Phase 2 (RQ3) package: opponent-shift scheduling + recovery-time tracking.
Import the machinery directly from the package:

    from phase2 import ShiftManager, RecoveryTracker, GameLog
"""
from phase2.phase2 import (
    ShiftManager, RecoveryTracker, GameLog, parse_schedule, RECOVERY_FRAC,
)

__all__ = [
    "ShiftManager", "RecoveryTracker", "GameLog", "parse_schedule",
    "RECOVERY_FRAC",
]
