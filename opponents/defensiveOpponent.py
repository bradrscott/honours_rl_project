# ── Defensive Opponent (blundering + builder style) ───────────
#
# Builds its own groups: defends, connects, values liberties, avoids
# unnecessary contact, takes free captures. Deliberately does NOT use
# self-atari / eye / territory avoidance, so it BLUNDERS — which keeps it
# beatable by a trained PPO. Its distinct trait is solid, passive building.
# Strict eps=0.
#
# Observation planes (go_v5, current player's perspective):
#   plane 0 = OPPONENT stones, plane 1 = OWN stones.

import numpy as np
from opponents import tactics

EPSILON        = 0.0
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

        A = tactics.analyze(opp_stones, own_stones)

        best_score, best_moves = -np.inf, []
        for action in legal_moves:
            if action == self.pass_action:
                continue
            r, c = action // self.board_size, action % self.board_size
            ev = tactics.evaluate_move(A, opp_stones, own_stones, r, c, self.board_size)

            # Blundering scoring — no self-atari / eye / territory terms.
            # ev.libs_after used as a mild "keep liberties" (solid) preference.
            score = (ev.saves * W_DEFEND + ev.adj_own * W_CONNECT
                     + ev.libs_after * W_LIBERTY + ev.captures * W_CAPTURE
                     + ev.adj_opp * W_CONTACT)

            if score > best_score:
                best_score, best_moves = score, [action]
            elif score == best_score:
                best_moves.append(action)

        if best_score < PASS_THRESHOLD and self.pass_action in legal_moves:
            return self.pass_action
        if not best_moves:
            return int(np.random.choice(legal_moves))
        return int(np.random.choice(best_moves))
