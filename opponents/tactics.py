# Shared tactical layer for the heuristic opponents.
# Real group, liberty awareness so every opponent plays sound tactical Go and
# differs only in style (via per-bot weights) - capture groups in atari, defend
# own groups, keep liberties, avoid self-atari and eye-filling.

# numpy for the board arrays, namedtuple/deque for the results + BFS queue
import numpy as np
from collections import namedtuple, deque

# the four orthogonal neighbours
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# Multi-source BFS - Manhattan distance from every point to the nearest own
# stone and the nearest opponent stone. Used for territory awareness — a point
# far from our stones but near the opponent's is contestable ground worth taking.
def influence_maps(opp_stones, own_stones):
    n = opp_stones.shape[0]

    # lookout distance
    BIG = np.int16(2 * n)

    # BFS outward from all stones of one colour at once
    def bfs(stones):
        dist = np.full((n, n), BIG, dtype=np.int16)
        dq = deque()

        # seed the queue with every stone of this colour at distance 0
        ys, xs = np.where(stones == 1)
        for r, c in zip(ys.tolist(), xs.tolist()):
            dist[r, c] = 0
            dq.append((r, c))

        # relax neighbours until the queue drains
        while dq:
            r, c = dq.popleft()
            d1 = dist[r, c] + 1
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc

                # update a neighbour only if this path reaches it sooner
                if 0 <= nr < n and 0 <= nc < n and dist[nr, nc] > d1:
                    dist[nr, nc] = d1
                    dq.append((nr, nc))
        return dist

    return bfs(own_stones), bfs(opp_stones)


# Territory value of playing at board points (row, column), normalised to [0,1] so it can never
# override a real capture/defence. High when the point is far from our own
# stones (new ground) yet still within reach of the opponent's (worth
# contesting). Distances beyond cap all count as fully open (1.0).
def openness(dist_own, dist_opp, r, c, cap=5):
    o = min(int(dist_own[r, c]), int(dist_opp[r, c]))
    return min(o, cap) / float(cap)

# per candidate move evaluation result
MoveEval = namedtuple("MoveEval",
    ["captures", "saves", "self_atari", "libs_after", "own_eye",
     "adj_opp", "adj_own"])


# finding group of stones - returns
# gid (n,n) group id per stone (-1 if empty/other colour)
# libs dict gid -> set of liberty points (adjacent truly-empty points)
# size dict gid -> number of stones
# cells dict gid -> list of stone points (row, column)
def _label(stones, empty):
    n = stones.shape[0]
    gid = -np.ones((n, n), dtype=int)
    libs, size, cells = {}, {}, {}
    g = 0

    # scan the board - each unlabelled stone starts a new search for group of stones
    for r in range(n):
        for c in range(n):
            if stones[r, c] == 1 and gid[r, c] == -1:

                # DFS stack seeded with this stone
                stack = [(r, c)]
                gid[r, c] = g
                grp_cells, grp_libs = [], set()
                while stack:
                    cr, cc = stack.pop()
                    grp_cells.append((cr, cc))

                    # visit the four neighbours
                    for dr, dc in _DIRS:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < n and 0 <= nc < n:

                            # empty neighbour = a liberty of this group
                            if empty[nr, nc]:
                                grp_libs.add((nr, nc))

                            # same-colour unlabelled neighbour = same group
                            elif stones[nr, nc] == 1 and gid[nr, nc] == -1:
                                gid[nr, nc] = g
                                stack.append((nr, nc))

                # record this group's liberties, size and cells
                libs[g], size[g], cells[g] = grp_libs, len(grp_cells), grp_cells
                g += 1
    return gid, libs, size, cells


# search group of stones for both colours once per move. Returns a dict of group info used by
# evaluate_move(). Liberties count only truly-empty points (neither colour).
def analyze(opp_stones, own_stones):
    empty = (opp_stones == 0) & (own_stones == 0)
    own_gid, own_libs, own_size, own_cells = _label(own_stones, empty)
    opp_gid, opp_libs, opp_size, opp_cells = _label(opp_stones, empty)
    return {
        "empty": empty,
        "own_gid": own_gid, "own_libs": own_libs, "own_size": own_size, "own_cells": own_cells,
        "opp_gid": opp_gid, "opp_libs": opp_libs, "opp_size": opp_size, "opp_cells": opp_cells,
    }


# Single-ply evaluation of playing our stone at empty point (row, column). Returns a MoveEval
#   captures - opponent stones captured by this move
#   saves - size of own adjacent groups in atari that this move rescues
#   self_atari - True if our group would have <=1 liberty and captures nothing
#   libs_after - liberties of our merged group after the move (safety)
#   own_eye - True if (row, column) is surrounded only by our own stones (eye-fill)
#   adj_opp - orthogonally adjacent opponent stones (contact)
#   adj_own - orthogonally adjacent own stones (connection)
def evaluate_move(A, opp_stones, own_stones, r, c, n):

    # pull the precomputed group info out of analyze()'s result
    empty = A["empty"]
    own_gid = A["own_gid"]; own_libs = A["own_libs"]; own_size = A["own_size"]
    opp_gid = A["opp_gid"]; opp_libs = A["opp_libs"]; opp_size = A["opp_size"]

    # counters accumulated over the four neighbours
    captures = 0
    saves = 0
    adj_opp = 0
    adj_own = 0
    on_board_neighbours = 0
    own_neighbours = 0

    merged_libs = set()          
    captured_points = set()          
    seen_own_grp = set()
    seen_opp_grp = set()

    # inspect each orthogonal neighbour of (r,c)
    for dr, dc in _DIRS:
        nr, nc = r + dr, c + dc

        # skip off-board neighbours
        if not (0 <= nr < n and 0 <= nc < n):
            continue
        on_board_neighbours += 1

        if empty[nr, nc]:
            merged_libs.add((nr, nc))                 
        elif own_stones[nr, nc] == 1:

            # connecting to one of our own groups
            adj_own += 1
            own_neighbours += 1
            g = own_gid[nr, nc]
            if g not in seen_own_grp:
                seen_own_grp.add(g)
                merged_libs |= own_libs[g]            
                if len(own_libs[g]) <= 1:             
                    saves += own_size[g]              
        elif opp_stones[nr, nc] == 1:

            # contacting an opponent group (maybe capturing it)
            adj_opp += 1
            g = opp_gid[nr, nc]
            if g not in seen_opp_grp:
                seen_opp_grp.add(g)
                if len(opp_libs[g] - {(r, c)}) == 0:  
                    captures += opp_size[g]
                    captured_points |= set(A["opp_cells"][g])

    # finalise our merged group's liberty set
    merged_libs.discard((r, c))           
    merged_libs |= captured_points        
    libs_after = len(merged_libs)

    # a self-atari is a move leaving us on <=1 liberty that captures nothing
    self_atari = (libs_after <= 1) and (captures == 0)

    # an eye-fill has only own stones around it and no opponent contact
    own_eye    = (on_board_neighbours > 0
                  and own_neighbours == on_board_neighbours
                  and adj_opp == 0)

    return MoveEval(captures, saves, self_atari, libs_after, own_eye,
                    adj_opp, adj_own)
