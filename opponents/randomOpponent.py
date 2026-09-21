# Random Opponent (zero-strategy baseline)
# Plays a uniformly random legal move every turn — no tactics, no scoring.
# Serves as the weakest, strategy-free reference opponent. 

# import for the board arrays / legal-move mask
import numpy as np


class RandomOpponent:

    # build the bot
    def __init__(self, board_size=None):
        self.board_size = board_size
        self.name = "random"

    # pick a uniformly random legal move from the action mask
    def select_action(self, obs):
        
        # the binary legal-move mask (1 = legal), including pass at index N*N
        action_mask = obs['action_mask']
        legal_moves  = np.where(action_mask == 1)[0]

        # must have at least one legal move
        if len(legal_moves) == 0:
            raise ValueError("No legal moves available — environment error.")

        # uniform random choice among the legal moves
        return int(np.random.choice(legal_moves))
