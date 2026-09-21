# Edge-Focused Opponent
# Sound contact play (capture, defend, attack, connect) with a soft lean toward
# the perimeter (edges). Its distinct trait is the edge lean.

# import for the board arrays, tactics - the shared move-evaluation engine
import numpy as np
from opponents import tactics

# move-scoring weights
EPSILON = 0.0
W_CAPTURE = 5.0   # take a free capture if offered
W_DEFEND = 3.0   # save own stones in atari
W_ATTACK = 2.0   # play next to opponent stones (pressure)
W_CONNECT = 1.5   # play next to own stones (build/connect)
W_BIAS = 3.0   # soft lean toward the nearest edge (its identity)


class EdgeOpponent:

    # build the bot - board_size is supplied by make_opponent (9 or 13 in this study)
    def __init__(self, board_size=9):

        # board geometry and the pass action index
        self.board_size = board_size
        self.n_actions = board_size * board_size + 1
        self.pass_action = board_size * board_size
        self.name = "edge"

        # precompute a bias map that peaks on the perimeter and falls off toward
        # the centre (edge_dist = distance to the nearest edge) — its identity
        n = board_size
        self._bias = np.zeros((n, n), dtype=np.float32)
        for r in range(n):
            for c in range(n):
                edge_dist = min(r, c, n - 1 - r, n - 1 - c)
                self._bias[r, c] = 1.0 - edge_dist / (n / 2.0)

    # pick this bot's move for the current position - score every legal move
    # (tactics + edge bias) and return the best, breaking ties at random
    def select_action(self, obs):

        # unpack the board planes and the legal-move mask
        board = obs['observation']
        action_mask = obs['action_mask']
        opp_stones = board[:, :, 0]
        own_stones = board[:, :, 1]

        # must have at least one legal move
        legal_moves = np.where(action_mask == 1)[0]
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available — environment error.")
        
        # optional difficulty knob
        if np.random.random() < EPSILON:
            return int(np.random.choice(legal_moves))

        # tactical analysis of the current position (groups, liberties, etc.)
        A = tactics.analyze(opp_stones, own_stones)

        # score every legal placement and keep the best (ties collected)
        best_score, best_moves = -np.inf, []
        for action in legal_moves:

            # skip passing while real moves remain
            if action == self.pass_action:
                continue
            r, c = action // self.board_size, action % self.board_size
            ev = tactics.evaluate_move(A, opp_stones, own_stones, r, c, self.board_size)

            # Blundering scoring — no self-atari, eye, territory terms.
            score = (ev.captures * W_CAPTURE + ev.saves * W_DEFEND
                     + ev.adj_opp * W_ATTACK + ev.adj_own * W_CONNECT
                     + self._bias[r, c] * W_BIAS)

            # track the best score, collecting ties for a random tie-break
            if score > best_score:
                best_score, best_moves = score, [action]
            elif score == best_score:
                best_moves.append(action)

        # fallback: pass if allowed, else a random legal move
        if not best_moves:
            return self.pass_action if self.pass_action in legal_moves \
                else int(np.random.choice(legal_moves))
        
        # random pick among the tied-best moves
        return int(np.random.choice(best_moves))
