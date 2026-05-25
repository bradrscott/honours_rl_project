# ── Aggressive Opponent ───────────────────────────────────────
#
# Heuristic Go opponent operating on the (19,19,17) board state.
#
# Strategy: actively attack opponent stones every turn.
#   - Strongly prefers moves adjacent to the most opponent stones
#     (maximises pressure on opponent groups).
#   - Adds a bonus for moves adjacent to opponent stones that
#     have few liberties (targets already-weakened groups).
#   - Ignores own connectivity and centre value — purely offensive.
#   - Never passes voluntarily.
#
# Usage:
#   opponent = AggressiveOpponent()
#   action   = opponent.select_action(obs)

import numpy as np

BOARD_SIZE = 19
N_ACTIONS  = BOARD_SIZE * BOARD_SIZE + 1   # 362


class AggressiveOpponent:
    """
    Aggressive heuristic opponent for Go.

    Scores every legal move by:
        adjacent_opponent  * 5.0   — direct attack on opponent stones
        threat_bonus       * 3.0   — extra weight if adjacent opponent
                                     stones have <= 2 liberties (weak group)

    Picks the highest-scoring legal move with random tiebreaking.
    Pass (action 361) is given score -10 and never selected voluntarily.

    Observation planes used (current player's perspective):
        Plane 0: current player stones
        Plane 8: opponent stones
    """

    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.name       = "aggressive"

    def _liberty_count(self, r, c, board):
        """
        Count empty adjacent intersections (liberties) around (r, c).
        Used to identify opponent stones under threat (low liberties).
        """
        liberties = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if board[nr, nc, 0] == 0 and board[nr, nc, 8] == 0:
                    liberties += 1
        return liberties

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

        opp_stones  = board[:, :, 8]

        legal_moves = np.where(action_mask == 1)[0]

        if len(legal_moves) == 0:
            raise ValueError("No legal moves available — environment error.")

        scores = np.full(N_ACTIONS, -np.inf)

        for action in legal_moves:
            if action == N_ACTIONS - 1:           # pass — avoid at all costs
                scores[action] = -10.0
                continue

            r = action // self.board_size
            c = action % self.board_size

            adj_opp      = 0.0
            threat_bonus = 0.0

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if opp_stones[nr, nc] == 1:
                        adj_opp += 1.0
                        libs = self._liberty_count(nr, nc, board)
                        if libs <= 2:
                            threat_bonus += 1.0

            scores[action] = adj_opp * 5.0 + threat_bonus * 3.0

        best_score = np.max(scores[legal_moves])
        best_moves = legal_moves[scores[legal_moves] == best_score]
        return int(np.random.choice(best_moves))
