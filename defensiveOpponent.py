# ── Defensive Opponent ────────────────────────────────────────
#
# Competitive heuristic Go opponent focused on BUILDING its own groups.
#   - Top priority: defend own groups in atari, connect stones, keep
#     liberties (solid, living shape).
#   - Captures when handed an easy capture, but does not seek fights.
#   - Avoids unnecessary contact with the opponent.
#
# Built on the shared tactical layer (tactics.py).
#
# Observation planes (go_v5, current player's perspective):
#   plane 0 = OPPONENT stones, plane 1 = OWN stones.

import numpy as np
import tactics

EPSILON        = 0.0   # 0.0 = STRICT (plays purely by its rules, no random moves)
PASS_THRESHOLD = 0.5
W_DEFEND  = 5.0   # saving own groups is the priority
W_CONNECT = 3.0   # build / connect own stones
W_LIBERTY = 1.5   # value empty neighbours (liberties = safety)
W_CAPTURE = 2.0   # take a free capture if offered
W_CONTACT = -0.5  # mild aversion to picking fights


class DefensiveOpponent:

    def __init__(self, board_size=19):
        self.board_size  = board_size
        self.n_actions   = board_size * board_size + 1
        self.pass_action = board_size * board_size
        self.name        = "defensive"

    def select_action(self, obs):
        board       = obs['observation']
        action_mask = obs['action_mask']
        opp_stones  = board[:, :, 0]
        own_stones  = board[:, :, 1]

        legal_moves = np.where(action_mask == 1)[0]
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available — environment error.")
        if np.random.random() < EPSILON:
            return int(np.random.choice(legal_moves))

        opp_map, own_map = tactics.liberty_maps(opp_stones, own_stones)

        best_score, best_moves = -np.inf, []
        for action in legal_moves:
            if action == self.pass_action:
                continue
            r, c = action // self.board_size, action % self.board_size

            adj_opp = adj_own = empty = 0.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if   own_stones[nr, nc] == 1: adj_own += 1.0
                    elif opp_stones[nr, nc] == 1: adj_opp += 1.0
                    else:                          empty   += 1.0

            cap, dfd = tactics.capture_defense(
                opp_map, own_map, opp_stones, own_stones, r, c, self.board_size)

            score = (dfd * W_DEFEND + adj_own * W_CONNECT + empty * W_LIBERTY
                     + cap * W_CAPTURE + adj_opp * W_CONTACT)

            if score > best_score:
                best_score, best_moves = score, [action]
            elif score == best_score:
                best_moves.append(action)

        if best_score < PASS_THRESHOLD and self.pass_action in legal_moves:
            return self.pass_action
        if not best_moves:
            return int(np.random.choice(legal_moves))
        return int(np.random.choice(best_moves))
