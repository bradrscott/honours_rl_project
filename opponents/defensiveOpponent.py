# Defensive Opponent
# Builds its own groups - defends, connects, values liberties, avoids
# unnecessary contact, takes free captures. Its distinct trait is solid, passive building.
# Observation planes (go_v5, current player's perspective) -
# plane 0 = opponent stones, plane 1 = own stones.

# import for the board arrays, tactics - the shared move-evaluation engine
import numpy as np
from opponents import tactics

# move-scoring weights 
EPSILON = 0.0
PASS_THRESHOLD = 0.5
W_DEFEND  = 5.0   # saving own groups is the priority
W_CONNECT = 3.0   # build, connect own stones
W_LIBERTY = 1.5   # value empty neighbours 
W_CAPTURE = 2.0   # take a free capture if offered
W_CONTACT = -0.5  # mild avoidance to picking fights

class DefensiveOpponent:

    # build the bot - board_size is supplied by make_opponent (9 or 13 in this study)
    def __init__(self, board_size=9):
        # board geometry and the pass action index
        self.board_size = board_size
        self.n_actions = board_size * board_size + 1
        self.pass_action = board_size * board_size
        self.name = "defensive"

    # pick this bot's move - score every legal move (defend/connect/liberties)
    # and return the best, breaking ties at random
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

            # no self-atari, eye, territory terms.
            # ev.libs_after used as a mild keep liberties preference.
            score = (ev.saves * W_DEFEND + ev.adj_own * W_CONNECT
                     + ev.libs_after * W_LIBERTY + ev.captures * W_CAPTURE
                     + ev.adj_opp * W_CONTACT)

            # track the best score, collecting ties for a random tie-break
            if score > best_score:
                best_score, best_moves = score, [action]
            elif score == best_score:
                best_moves.append(action)

        # if no move is clearly worthwhile, pass rather than play a weak move
        if best_score < PASS_THRESHOLD and self.pass_action in legal_moves:
            return self.pass_action
        
        # fallback - a random legal move if nothing was scored
        if not best_moves:
            return int(np.random.choice(legal_moves))
        
        # random pick among the tied-best moves
        return int(np.random.choice(best_moves))
