# ── Greedy Opponent ───────────────────────────────────────────
#
# Heuristic Go opponent operating on the (N, N, 17) board state.
#
# Strategy: maximise immediate local gain each turn.
#   - Strongly prefers moves adjacent to opponent stones (pressure).
#   - Rewards connecting to own stones and central positions.
#   - Passes when no move scores above a minimum threshold.
#
# Observation planes (PettingZoo go_v5, current player's perspective):
#   plane 0 = OPPONENT stones
#   plane 1 = OWN (current player) stones
#   (Confirmed empirically. The previous version read plane 0 as own
#    and plane 8 as opponent — both wrong; plane 8 is always empty.)
#
# Board-size aware: all action/pass indices derive from board_size,
# so it works on 7x7, 9x9, 13x13, 19x19 — not just 19x19.

import numpy as np

PASS_THRESHOLD = 0.3   # pass if best move scores below this
EPSILON        = 0.2   # default fraction of random legal moves


class GreedyOpponent:

    def __init__(self, board_size=19):
        self.board_size  = board_size
        self.n_actions   = board_size * board_size + 1
        self.pass_action = board_size * board_size
        self.name        = "greedy"

        centre = board_size / 2.0
        self._centre_w = np.zeros((board_size, board_size), dtype=np.float32)
        for r in range(board_size):
            for c in range(board_size):
                dist = abs(r - centre) + abs(c - centre)
                self._centre_w[r, c] = 1.0 - dist / board_size

    def select_action(self, obs):
        board       = obs['observation']
        action_mask = obs['action_mask']

        opp_stones  = board[:, :, 0]   # opponent
        own_stones  = board[:, :, 1]   # current player (us)

        legal_moves = np.where(action_mask == 1)[0]
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available — environment error.")

        # epsilon-fraction random play
        if np.random.random() < EPSILON:
            return int(np.random.choice(legal_moves))

        # only pass legal
        if len(legal_moves) == 1 and legal_moves[0] == self.pass_action:
            return self.pass_action

        best_score = -np.inf
        best_moves = []
        for action in legal_moves:
            if action == self.pass_action:
                continue
            r = action // self.board_size
            c = action % self.board_size

            adj_opp = 0.0
            adj_own = 0.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    adj_opp += opp_stones[nr, nc]
                    adj_own += own_stones[nr, nc]

            score = adj_opp * 3.0 + adj_own * 1.0 + self._centre_w[r, c] * 0.5

            if score > best_score:
                best_score = score
                best_moves = [action]
            elif score == best_score:
                best_moves.append(action)

        # pass if nothing meaningful and pass is legal
        if best_score < PASS_THRESHOLD and self.pass_action in legal_moves:
            return self.pass_action

        return int(np.random.choice(best_moves))
