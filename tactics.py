# ── Shared tactical helper for heuristic opponents ────────────
#
# Lightweight group/liberty awareness so positional bots (corner, edge,
# defensive) can also capture opponent stones in atari and defend their
# own groups in atari — making them competitive while keeping their
# distinct positional style.
#
# Observation planes (PettingZoo go_v5, current player's perspective):
#   plane 0 = OPPONENT stones, plane 1 = OWN stones.
#
# Usage in an opponent's select_action():
#   opp_map, own_map = liberty_maps(opp_stones, own_stones)
#   cap, def = capture_defense(opp_map, own_map, opp_stones, own_stones, r, c, N)
#   score = positional_score + cap * W_CAP + def * W_DEF

import numpy as np

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _group_map(stones, empty):
    """Flood-fill connected same-colour groups. `stones` is the binary
    plane for this colour; `empty` is the binary plane of truly-empty
    points. Returns dict (r,c) -> (group_liberties, group_size).
    A liberty is an adjacent EMPTY point (not an opponent stone)."""
    n = stones.shape[0]
    visited = np.zeros((n, n), dtype=bool)
    cell_info = {}
    for r in range(n):
        for c in range(n):
            if stones[r, c] == 1 and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                group, libs = [], set()
                while stack:
                    cr, cc = stack.pop()
                    group.append((cr, cc))
                    for dr, dc in _DIRS:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            if empty[nr, nc]:
                                libs.add((nr, nc))          # empty = liberty
                            elif stones[nr, nc] == 1 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                info = (len(libs), len(group))
                for cell in group:
                    cell_info[cell] = info
    return cell_info


def liberty_maps(opp_stones, own_stones):
    """Compute group-liberty maps for both colours once per move.
    Liberties are counted only against truly-empty points (occupied by
    neither colour)."""
    empty = (opp_stones == 0) & (own_stones == 0)
    return _group_map(opp_stones, empty), _group_map(own_stones, empty)


def capture_defense(opp_map, own_map, opp_stones, own_stones, r, c, n):
    """For a candidate empty point (r,c): how many opponent stones it would
    capture (adjacent opponent group with exactly 1 liberty), and how many
    of our own stones are in atari next to it (worth defending/extending)."""
    capture = 0.0
    defense = 0.0
    for dr, dc in _DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            if opp_stones[nr, nc] == 1:
                libs, size = opp_map.get((nr, nc), (99, 0))
                if libs <= 1:                 # this move captures the group
                    capture += size
            elif own_stones[nr, nc] == 1:
                libs, size = own_map.get((nr, nc), (99, 0))
                if libs <= 1:                 # own group in atari — defend it
                    defense += size
    return capture, defense
