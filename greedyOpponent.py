# ── Greedy Opponent ───────────────────────────────────────────
#
# Heuristic Go opponent operating on the (19,19,17) board state.
#
# Strategy: maximise immediate local gain each turn.
#   - Strongly prefers moves adjacent to opponent stones
#     (threatens captures and reduces opponent liberties).
#   - Also rewards connecting to own stones (building groups).
#   - Adds a small centre bonus — central positions have higher
#     strategic value in Go (more liberties, more territory).
#   - Avoids passing unless no legal move scores above 0.
#
# Usage:
#   opponent = GreedyOpponent()
#   action   = opponent.select_action(obs)

import numpy as np

BOARD_SIZE = 19
N_ACTIONS  = BOARD_SIZE * BOARD_SIZE + 1   # 362


class GreedyOpponent:
    """
    Greedy heuristic opponent for Go.

    Scores every legal move by:
        adjacent_opponent  * 3.0   — capture / reduce opponent liberties
        adjacent_own       * 1.0   — reinforce own groups
        centre_weight      * 0.5   — strategic value of the position

    Picks the highest-scoring legal move with random tiebreaking.
    Pass (action 361) is given score -1 and only selected if forced.

    Observation planes used (current player's perspective):
        Plane 0: current player stones
        Plane 8: opponent stones
    """

    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.name       = "greedy"

        # Precompute centre distance weights (normalised 0-1)
        centre         = board_size / 2.0
        self._centre_w = np.zeros((board_size, board_size), dtype=np.float32)
        for r in range(board_size):
            for c in range(board_size):
                dist = abs(r - centre) + abs(c - centre)
                self._centre_w[r, c] = 1.0 - dist / board_size

    def select_action(self, obs):
        """
        Parameters:
            obs (dict): PettingZoo observation with keys
                'observation'  — (19, 19, 17) board state
                'action_mask'  — (362,) binary legal-move mask

        Returns:
            int: selected legal action index
        """
        board       = obs['observation']
        action_mask = obs['action_mask']

        own_stones  = board[:, :, 0]
        opp_stones  = board[:, :, 8]

        legal_moves = np.where(action_mask == 1)[0]

        if len(legal_moves) == 0:
            raise ValueError("No legal moves available — environment error.")

        scores = np.full(N_ACTIONS, -np.inf)

        for action in legal_moves:
            if action == N_ACTIONS - 1:           # pass
                scores[action] = -1.0
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

            scores[action] = (adj_opp * 3.0
                              + adj_own * 1.0
                              + self._centre_w[r, c] * 0.5)

        best_score = np.max(scores[legal_moves])
        best_moves = legal_moves[scores[legal_moves] == best_score]
        return int(np.random.choice(best_moves))
