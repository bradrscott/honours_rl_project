import numpy as np

class RandomOpponent:
    """
    Random heuristic opponent for Go.

    Selects a uniformly random legal move from the action mask
    provided by PettingZoo on every turn.

    Legal moves are strictly enforced — only actions where
    action_mask == 1 are ever selected. This includes the pass
    move (action N^2) if it is legal.

    Usage:
        opponent = RandomOpponent()
        action = opponent.select_action(obs)
    """

    def __init__(self):
        self.name = "random"

    def select_action(self, obs):
        """
        Selects a random legal move.

        Parameters:
            obs (dict): PettingZoo observation containing:
                - 'observation': (N, N, 17) board state array
                - 'action_mask': (N*N + 1,) binary array of legal moves

        Returns:
            int: index of the selected legal action
        """
        action_mask = obs['action_mask']
        legal_moves  = np.where(action_mask == 1)[0]

        if len(legal_moves) == 0:
            raise ValueError("No legal moves available — environment error.")

        return int(np.random.choice(legal_moves))
