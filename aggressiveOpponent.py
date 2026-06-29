# ── Aggressive Opponent ───────────────────────────────────────
#
# Competitive heuristic Go opponent focused on CAPTURING pieces.
#   - Top priority: capture opponent groups (atari) and pressure them.
#   - Also defends its own groups in atari (so it stays competitive).
#   - Plays contact moves against the opponent.
#
# Built on the shared tactical layer (tactics.py): real group/liberty
# awareness, so it actually captures rather than just touching stones.
#
# Observation planes (go_v5, current player's perspective):
#   plane 0 = OPPONENT stones, plane 1 = OWN stones.

import numpy as np
import tactics

EPSILON   = 0.0   # 0.0 = STRICT (plays purely by its rules, no random moves)
W_CAPTURE = 6.0   # capturing is the whole point
W_ATTACK  = 3.0   # pressure: play in contact with opponent stones
W_DEFEND  = 1.5   # don't lose own groups for free
W_CONNECT = 0.5


class AggressiveOpponent:

    def __init__(self, board_size=19):
        self.board_size  = board_size
        self.n_actions   = board_size * board_size + 1
        self.pass_action = board_size * board_size
        self.name        = "aggressive"

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

            score = (cap * W_CAPTURE + adj_opp * W_ATTACK
                     + dfd * W_DEFEND + adj_own * W_CONNECT)

            if score > best_score:
                best_score, best_moves = score, [action]
            elif score == best_score:
                best_moves.append(action)

        if not best_moves:
            return self.pass_action if self.pass_action in legal_moves \
                else int(np.random.choice(legal_moves))
        return int(np.random.choice(best_moves))
