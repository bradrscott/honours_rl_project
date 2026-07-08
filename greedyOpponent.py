# ── Greedy Opponent (territory-grabbing / point-greedy) ───────
#
# A GREEDY Go player: grabs the biggest available point (most territory)
# every move, and takes a capture only if one is handed to it. It does NOT
# hunt the opponent (no attack/contact focus) — that's the real greedy
# weakness: it's acquisitive and thin, so the agent's groups survive and
# it can out-play a greedy point-grabber. Strict eps=0.
#
#   FOCUS:      biggest open point (territory / influence)   -> W_TERRITORY
#   secondary:  take a free capture, basic defend/connect    -> low weights
#   NOT present: attack / contact-seeking (it does not hunt)
#
# Observation planes (go_v5, current player's perspective):
#   plane 0 = OPPONENT stones, plane 1 = OWN stones.

import numpy as np
import tactics

EPSILON        = 0.0
PASS_THRESHOLD = 0.2

W_TERRITORY = 6.0   # FOCUS: play the biggest open point (grab territory)
W_CAPTURE   = 2.0   # take a free capture if offered (not the focus)
W_DEFEND    = 2.0   # basic defence of own groups
W_CONNECT   = 1.0   # basic connection
W_ATTACK    = 1.0   # MINOR hunt: slight contact/pressure so it can chase a
                    # capture. Kept small so territory stays the focus — this
                    # only nudges it toward the opponent, it is not an attacker.


class GreedyOpponent:

    def __init__(self, board_size=19):
        self.board_size  = board_size
        self.n_actions   = board_size * board_size + 1
        self.pass_action = board_size * board_size
        self.name        = "greedy"

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
        dist_own, dist_opp = tactics.influence_maps(opp_stones, own_stones)

        best_score, best_moves = -np.inf, []
        for action in legal_moves:
            if action == self.pass_action:
                continue
            r, c = action // self.board_size, action % self.board_size
            ev = tactics.evaluate_move(A, opp_stones, own_stones, r, c, self.board_size)

            # FOCUS = biggest open point; take a free capture / basic defend.
            # MINOR attack term so it does a *little* hunting (chase captures)
            # without becoming an attacker — territory still dominates.
            score = (tactics.openness(dist_own, dist_opp, r, c) * W_TERRITORY
                     + ev.captures * W_CAPTURE
                     + ev.saves * W_DEFEND
                     + ev.adj_own * W_CONNECT
                     + ev.adj_opp * W_ATTACK)

            if score > best_score:
                best_score, best_moves = score, [action]
            elif score == best_score:
                best_moves.append(action)

        if best_score < PASS_THRESHOLD and self.pass_action in legal_moves:
            return self.pass_action
        if not best_moves:
            return int(np.random.choice(legal_moves))
        return int(np.random.choice(best_moves))
