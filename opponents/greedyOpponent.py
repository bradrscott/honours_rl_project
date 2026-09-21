# Greedy Opponent 
# A greedy Go player - grabs the biggest available open point (most territory)
# every move and takes a capture only if one is handed to it. It does not hunt
# the opponent — that's its weakness - acquisitive and thin so the agent's
# groups survive and can out-play it.

# import for the board arrays, tactics - the shared move-evaluation engine
import numpy as np
from opponents import tactics

# move-scoring weights 
EPSILON = 0.0
PASS_THRESHOLD = 0.2
W_TERRITORY = 6.0   # play the biggest open point (grab territory)
W_CAPTURE = 2.0   # take a free capture if offered (not the focus)
W_DEFEND = 2.0   # basic defence of own groups
W_CONNECT = 1.0   # basic connection of own stones
W_ATTACK = 1.0   # slight pressure so it can chase a capture

class GreedyOpponent:

    # build the bot - board_size is supplied by make_opponent (9 or 13 in this study)
    def __init__(self, board_size=9):

        # board geometry and the pass action index
        self.board_size = board_size
        self.n_actions = board_size * board_size + 1
        self.pass_action = board_size * board_size
        self.name = "greedy"

    # pick this bot's move - score every legal move (territory + light tactics)
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

        # tactical analysis + influence maps 
        A = tactics.analyze(opp_stones, own_stones)
        dist_own, dist_opp = tactics.influence_maps(opp_stones, own_stones)

        # score every legal placement and keep the best (ties collected)
        best_score, best_moves = -np.inf, []
        for action in legal_moves:

            # skip passing while real moves remain
            if action == self.pass_action:
                continue
            r, c = action // self.board_size, action % self.board_size
            ev = tactics.evaluate_move(A, opp_stones, own_stones, r, c, self.board_size)

            # focus - biggest open point, take a free capture, basic defend.
            # minor attack term so it does a little hunting (chase captures)
            # without becoming an attacker — territory still dominates.
            score = (tactics.openness(dist_own, dist_opp, r, c) * W_TERRITORY
                     + ev.captures * W_CAPTURE
                     + ev.saves * W_DEFEND
                     + ev.adj_own * W_CONNECT
                     + ev.adj_opp * W_ATTACK)

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
