# ── Shared tactical layer for the heuristic opponents ─────────────
#
# Real group / liberty awareness so every opponent (greedy, aggressive,
# defensive, corner, edge) plays SOUND tactical Go and only differs in
# STYLE (via per-bot weights). The agent can no longer win by exploiting
# blunders, because every bot now:
#   - never plays self-atari (a move leaving its own group on <=1 liberty)
#   - never fills its own eyes
#   - prefers moves that keep liberties (solid, living shape)
#   - captures opponent groups in atari and defends its own
#
# Observation planes (PettingZoo go_v5, current player's perspective):
#   plane 0 = OPPONENT stones, plane 1 = OWN stones.
#
# Usage in an opponent's select_action():
#   A = tactics.analyze(opp_stones, own_stones)
#   ev = tactics.evaluate_move(A, opp_stones, own_stones, r, c, N)
#   score = ev.captures*W_CAP + ev.saves*W_DEF + ... - ev.self_atari*W_SA ...

import numpy as np
from collections import namedtuple, deque

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def influence_maps(opp_stones, own_stones):
    """Multi-source BFS: Manhattan distance from every board point to the
    nearest OWN stone and the nearest OPPONENT stone.

    Used for TERRITORY awareness. A point that is far from your own stones
    but not far from the opponent's is a big, contestable area — playing
    there stakes new territory instead of piling onto ground you already
    control. This is what stops the bots from ceding the board (the reason
    a territorial agent beats them). Returns (dist_own, dist_opp), (n, n).
    A colour with no stones yields all-BIG (nothing owned yet)."""
    n = opp_stones.shape[0]
    BIG = np.int16(2 * n)

    def bfs(stones):
        dist = np.full((n, n), BIG, dtype=np.int16)
        dq = deque()
        ys, xs = np.where(stones == 1)
        for r, c in zip(ys.tolist(), xs.tolist()):
            dist[r, c] = 0
            dq.append((r, c))
        while dq:
            r, c = dq.popleft()
            d1 = dist[r, c] + 1
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and dist[nr, nc] > d1:
                    dist[nr, nc] = d1
                    dq.append((nr, nc))
        return dist

    return bfs(own_stones), bfs(opp_stones)


def openness(dist_own, dist_opp, r, c, cap=5):
    """Territory value of playing at (r,c), NORMALISED to [0, 1] so it can
    never override a real capture/defence (those use raw weights). High when
    the point is far from our OWN stones (new ground) yet still within reach
    of the opponent's (worth contesting). Distances beyond `cap` all count as
    "very open" (1.0). Rewards spreading out / contesting over clustering."""
    o = min(int(dist_own[r, c]), int(dist_opp[r, c]))
    return min(o, cap) / float(cap)

# Per-candidate-move evaluation result.
MoveEval = namedtuple("MoveEval",
    ["captures", "saves", "self_atari", "libs_after", "own_eye",
     "adj_opp", "adj_own"])


def _label(stones, empty):
    """Flood-fill connected same-colour groups. Returns:
        gid:   (n,n) int array, group id per stone (-1 if empty/other)
        libs:  dict gid -> set of liberty points (r,c)
        size:  dict gid -> number of stones
        cells: dict gid -> list of stone points (r,c)
    A liberty is an adjacent truly-empty point."""
    n = stones.shape[0]
    gid = -np.ones((n, n), dtype=int)
    libs, size, cells = {}, {}, {}
    g = 0
    for r in range(n):
        for c in range(n):
            if stones[r, c] == 1 and gid[r, c] == -1:
                stack = [(r, c)]
                gid[r, c] = g
                grp_cells, grp_libs = [], set()
                while stack:
                    cr, cc = stack.pop()
                    grp_cells.append((cr, cc))
                    for dr, dc in _DIRS:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            if empty[nr, nc]:
                                grp_libs.add((nr, nc))
                            elif stones[nr, nc] == 1 and gid[nr, nc] == -1:
                                gid[nr, nc] = g
                                stack.append((nr, nc))
                libs[g], size[g], cells[g] = grp_libs, len(grp_cells), grp_cells
                g += 1
    return gid, libs, size, cells


def analyze(opp_stones, own_stones):
    """Flood-fill BOTH colours once per move. Returns a dict of group info
    used by evaluate_move(). Liberties counted only against truly-empty
    points (occupied by neither colour)."""
    empty = (opp_stones == 0) & (own_stones == 0)
    own_gid, own_libs, own_size, own_cells = _label(own_stones, empty)
    opp_gid, opp_libs, opp_size, opp_cells = _label(opp_stones, empty)
    return {
        "empty":    empty,
        "own_gid":  own_gid, "own_libs": own_libs, "own_size": own_size, "own_cells": own_cells,
        "opp_gid":  opp_gid, "opp_libs": opp_libs, "opp_size": opp_size, "opp_cells": opp_cells,
    }


def evaluate_move(A, opp_stones, own_stones, r, c, n):
    """Single-ply evaluation of playing OUR stone at empty point (r,c).

    Returns a MoveEval with:
      captures   : number of opponent stones captured by this move
      saves      : size of own adjacent groups in atari that this move rescues
      self_atari : True if our resulting group would have <=1 liberty and
                   the move captures nothing (a blunder — never do this)
      libs_after : liberties of our merged group after the move (safety)
      own_eye    : True if (r,c) is surrounded only by our own stones
                   (filling our own eye — wasteful, sometimes suicidal)
      adj_opp    : number of orthogonally adjacent opponent stones (contact)
      adj_own    : number of orthogonally adjacent own stones (connection)
    """
    empty   = A["empty"]
    own_gid = A["own_gid"]; own_libs = A["own_libs"]; own_size = A["own_size"]
    opp_gid = A["opp_gid"]; opp_libs = A["opp_libs"]; opp_size = A["opp_size"]

    captures = 0
    saves    = 0
    adj_opp  = 0
    adj_own  = 0
    on_board_neighbours = 0
    own_neighbours      = 0

    merged_libs     = set()          # liberties of our group after the move
    captured_points = set()          # opponent points freed by capture
    seen_own_grp    = set()
    seen_opp_grp    = set()

    for dr, dc in _DIRS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n and 0 <= nc < n):
            continue
        on_board_neighbours += 1

        if empty[nr, nc]:
            merged_libs.add((nr, nc))                 # a fresh liberty
        elif own_stones[nr, nc] == 1:
            adj_own += 1
            own_neighbours += 1
            g = own_gid[nr, nc]
            if g not in seen_own_grp:
                seen_own_grp.add(g)
                merged_libs |= own_libs[g]            # inherit its liberties
                if len(own_libs[g]) <= 1:             # it was in atari...
                    saves += own_size[g]              # ...and we connect to it
        elif opp_stones[nr, nc] == 1:
            adj_opp += 1
            g = opp_gid[nr, nc]
            if g not in seen_opp_grp:
                seen_opp_grp.add(g)
                if len(opp_libs[g] - {(r, c)}) == 0:  # this move captures it
                    captures += opp_size[g]
                    captured_points |= set(A["opp_cells"][g])

    merged_libs.discard((r, c))           # the point we just filled
    merged_libs |= captured_points        # freed enemy points become liberties
    libs_after = len(merged_libs)

    self_atari = (libs_after <= 1) and (captures == 0)
    own_eye    = (on_board_neighbours > 0
                  and own_neighbours == on_board_neighbours
                  and adj_opp == 0)

    return MoveEval(captures, saves, self_atari, libs_after, own_eye,
                    adj_opp, adj_own)
