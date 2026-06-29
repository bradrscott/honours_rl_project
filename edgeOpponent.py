# ── Edge-Focused Opponent ─────────────────────────────────────
#
# Competitive heuristic Go opponent that builds AND attacks, but
# concentrates its play along the PERIMETER (the board edges). It plays
# sound contact Go (capture / defend / build), with a strong spatial
# pull toward the edges — so its stones cluster on the sides while still
# fighting well.
#
# Distinct from the corner opponent: corner play clusters at one corner,
# edge play spreads along the whole perimeter.
#
# Built on the shared tactical layer (tactics.py).
#
# Observation planes (go_v5, current player's perspective):
#   plane 0 = OPPONENT stones, plane 1 = OWN stones.

import numpy as np
import tactics

EPSILON   = 0.0   # 0.0 = STRICT (plays purely by its rules, no random moves)
W_CAPTURE = 5.0
W_DEFEND  = 3.0
W_ATTACK  = 2.0
W_CONNECT = 1.5
W_BIAS    = 6.0   # perimeter spatial pull (competitive + visible lean;
                  # higher concentrates more but weakens play — see notes)


class EdgeOpponent:

    def __init__(self, board_size=19):
        self.board_size  = board_size
        self.n_actions   = board_size * board_size + 1
        self.pass_action = board_size * board_size
        self.name        = "edge"

        # spatial pull toward the nearest board edge
        n = board_size
        self._bias = np.zeros((n, n), dtype=np.float32)
        for r in range(n):
            for c in range(n):
                edge_dist = min(r, c, n - 1 - r, n - 1 - c)
                self._bias[r, c] = 1.0 - edge_dist / (n / 2.0)

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

            adj_opp = adj_own = 0.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    adj_opp += opp_stones[nr, nc]
                    adj_own += own_stones[nr, nc]

            cap, dfd = tactics.capture_defense(
                opp_map, own_map, opp_stones, own_stones, r, c, self.board_size)

            score = (cap * W_CAPTURE + dfd * W_DEFEND
                     + adj_opp * W_ATTACK + adj_own * W_CONNECT
                     + self._bias[r, c] * W_BIAS)

            if score > best_score:
                best_score, best_moves = score, [action]
            elif score == best_score:
                best_moves.append(action)

        if not best_moves:
            return self.pass_action if self.pass_action in legal_moves \
                else int(np.random.choice(legal_moves))
        return int(np.random.choice(best_moves))
