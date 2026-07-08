# ── Edge-Focused Opponent (blundering + edge lean) ────────────
#
# Sound-ish contact play (capture / defend / attack / connect) with a soft
# lean toward the perimeter (edges). Deliberately does NOT use self-atari /
# eye / territory avoidance, so it BLUNDERS — which keeps it beatable by a
# trained PPO. Its distinct trait is the edge lean. Strict eps=0.
#
# Observation planes (go_v5, current player's perspective):
#   plane 0 = OPPONENT stones, plane 1 = OWN stones.

import numpy as np
import tactics

EPSILON   = 0.0

W_CAPTURE = 5.0
W_DEFEND  = 3.0
W_ATTACK  = 2.0
W_CONNECT = 1.5
W_BIAS    = 3.0        # soft lean toward the nearest edge (its identity)


class EdgeOpponent:

    def __init__(self, board_size=19):
        self.board_size  = board_size
        self.n_actions   = board_size * board_size + 1
        self.pass_action = board_size * board_size
        self.name        = "edge"

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

        A = tactics.analyze(opp_stones, own_stones)

        best_score, best_moves = -np.inf, []
        for action in legal_moves:
            if action == self.pass_action:
                continue
            r, c = action // self.board_size, action % self.board_size
            ev = tactics.evaluate_move(A, opp_stones, own_stones, r, c, self.board_size)

            # Blundering scoring — no self-atari / eye / territory terms.
            score = (ev.captures * W_CAPTURE + ev.saves * W_DEFEND
                     + ev.adj_opp * W_ATTACK + ev.adj_own * W_CONNECT
                     + self._bias[r, c] * W_BIAS)

            if score > best_score:
                best_score, best_moves = score, [action]
            elif score == best_score:
                best_moves.append(action)

        if not best_moves:
            return self.pass_action if self.pass_action in legal_moves \
                else int(np.random.choice(legal_moves))
        return int(np.random.choice(best_moves))
