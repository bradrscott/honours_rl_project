# ── Aggressive Opponent (DUMB all-out attacker) ───────────────
#
# A DUMB aggressive Go player: it charges at the opponent. Its ONLY real
# driver is contact/attack (adj_opp) — it plays right up against the
# opponent's stones. It is deliberately tactically STUPID:
#   * NO defence of its own groups            -> overextends
#   * NO self-atari / liberty awareness       -> plays into capture
#   * NO territory / connection               -> never builds a base
#   * only a TINY capture term                -> it does not hunt/snipe ataris
#
# Why dumb? Our other bots use tactics.evaluate_move competently, and a
# COMPETENT eps=0 bot never blunders, so it tactically walls a from-scratch
# agent at ~0% forever (every capture-hungry config — 8/5, 5/5, 5/2, 3/3,
# +territory — stayed stuck). A dumb attacker still has the highest attack
# component of any bot, but it throws weak, capturable stones all over the
# board with no self-preservation, so the agent can punish it and win. That
# makes it beatable while keeping a clearly AGGRESSIVE personality. Strict eps=0.
#
# Observation planes (go_v5, current player's perspective):
#   plane 0 = OPPONENT stones, plane 1 = OWN stones.

import numpy as np
from opponents import tactics

EPSILON   = 0.0

W_ATTACK  = 3.0   # ONLY real driver: play in contact / charge the opponent.
                  # Highest attack weight of any bot -> the most aggressive.
W_CAPTURE = 1.0   # TINY: takes a capture only if it happens to be the contact
                  # move. It does NOT hunt or snipe the agent's ataris (that
                  # competence is exactly what deadlocked the agent at ~0%).
# NO defend / self-atari / territory / connect -> it is tactically dumb and
# overextends, handing the agent capturable weak groups everywhere.

# THE key to beatability (see below). go_v5 only ends on two CONSECUTIVE passes
# and area-scores (every stone on the board counts unless captured). A bot that
# NEVER passes forces games to saturation: it fills the agent's territory with
# stones that then count for it, and with komi the agent loses ~95% REGARDLESS
# of move quality (that is why smart/even/territory/dumb all gave the same flat
# 5%). A pure attacker naturally stops when it can't reach the enemy: once the
# agent's stones are all sealed off, NO legal move is adjacent to them, so the
# best attack score is 0 and the bot PASSES -> the game ends with the agent's
# territory counted. This threshold sits between 0 (nothing to attack) and the
# score of any real attack (adj_opp>=1 -> >=3), so it only fires when contained.
PASS_THRESHOLD = 0.5


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

        A = tactics.analyze(opp_stones, own_stones)

        best_score, best_moves = -np.inf, []
        for action in legal_moves:
            if action == self.pass_action:
                continue
            r, c = action // self.board_size, action % self.board_size
            ev = tactics.evaluate_move(A, opp_stones, own_stones, r, c, self.board_size)

            # DUMB attacker: charge at the opponent (adj_opp), take an
            # incidental capture. NO defence / self-atari / territory -> it
            # overextends and leaves weak groups the agent can punish.
            score = ev.adj_opp * W_ATTACK + ev.captures * W_CAPTURE

            if score > best_score:
                best_score, best_moves = score, [action]
            elif score == best_score:
                best_moves.append(action)

        # Nothing worth attacking (agent sealed off, no capture) -> pass, so the
        # game can actually end and the agent's territory is scored. Without
        # this the bot fills the board forever and the agent can't win (komi +
        # area scoring). This is what makes a pure attacker beatable.
        if best_score < PASS_THRESHOLD and self.pass_action in legal_moves:
            return self.pass_action
        if not best_moves:
            return int(np.random.choice(legal_moves))
        return int(np.random.choice(best_moves))
