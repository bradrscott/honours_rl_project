# ══════════════════════════════════════════════════════════════
# EVAL: trained agent (PPO or feudal) vs GNU Go — absolute skill anchor
#
# Loads a trained checkpoint and plays it (BLACK, the seat it trained in)
# against GNU Go (WHITE) at a given level, refereed + area-scored by go_v5.
# Reports win rate over N games. Runs ENTIRELY on your Mac (CPU) — needs only
# the checkpoint + the `gnugo` binary (already installed for your baseline).
#
# NOTE: the agents were trained ONLY vs heuristic bots, never vs GNU Go, so
# this measures TRANSFER to a strong unseen opponent — expect low win rates
# (GNU Go is far stronger). That is the intended "skill anchor" framing.
#
# Usage (one config; loop levels/boards/agents outside or via the helper):
#   python gnugo/eval_agent_vs_gnugo.py --agent feudal \
#       --checkpoint phase1_checkpoints/feudal/9x9/greedy/feudal_go_final.pt \
#       --board-size 9 --komi 5.5 --level 10 --games 100 --tag greedy
# ══════════════════════════════════════════════════════════════

import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The training configs read OPPONENT/BOARD_SIZE from the env at import time.
# Eval never calls make_opponent (GNU Go is the opponent), so a dummy is fine —
# set BEFORE importing so the imports don't KeyError.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--board-size", type=int, default=9)
_pre.add_argument("--komi", type=float, default=5.5)
_a, _ = _pre.parse_known_args()
os.environ.setdefault("OPPONENT", "eval")
os.environ["BOARD_SIZE"] = str(_a.board_size)
os.environ["KOMI"] = str(_a.komi)

import numpy as np
import torch
from pettingzoo.classic import go_v5

from ppo.ppo_go import ActorCritic
from ppo.config_ppo_go import (N_CHANNELS, CNN_FILTERS, CNN_LAYERS, HIDDEN_DIM)
from feudal.feudalNetwork import FeudalNetwork, init_hidden
from feudal.config_feudal import (HIDDEN_DIM_M, HIDDEN_DIM_W, TIME_HORIZON, DILATION)
from gnugo.engine import GnuGo
from gnugo.gtp import GTPError

BLACK, WHITE = "black_0", "white_0"


def obs_chw(obs_dict):
    o = obs_dict["observation"].astype(np.float32)
    return np.transpose(o, (2, 0, 1)).copy()


class AgentPolicy:
    """Greedy (argmax) policy wrapper around a loaded PPO or feudal net."""
    def __init__(self, agent_type, ckpt, board_size, device):
        self.type = agent_type
        self.n_actions = board_size * board_size + 1
        self.device = device
        if agent_type == "ppo":
            self.net = ActorCritic(board_size, N_CHANNELS, self.n_actions,
                                   CNN_FILTERS, CNN_LAYERS, HIDDEN_DIM)
        elif agent_type == "feudal":
            self.net = FeudalNetwork(
                board_size=board_size, n_channels=N_CHANNELS,
                n_actions=self.n_actions, hidden_dim_manager=HIDDEN_DIM_M,
                hidden_dim_worker=HIDDEN_DIM_W, time_horizon=TIME_HORIZON,
                dilation=DILATION, eps=0.0,              # eps=0 => no random goals in eval
                num_workers=1, device=str(device),
                n_filters=CNN_FILTERS, n_layers=CNN_LAYERS)
        else:
            raise ValueError(agent_type)
        self.net.load_state_dict(torch.load(ckpt, map_location=device))
        self.net.to(device).eval()

    def reset(self):
        """Per-GAME reset so games are independent (feudal is stateful)."""
        if self.type == "feudal":
            self.goals, self.states, _ = self.net.init_obj()
            self.net.hidden_m = init_hidden(1, self.net.r * self.net.d, device=self.device)
            self.net.hidden_w = init_hidden(1, self.net.k * self.n_actions, device=self.device)
            self.net.manager.Mrnn.dilation = 0

    @torch.no_grad()
    def act(self, obs_dict):
        x = torch.as_tensor(obs_chw(obs_dict), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        mask = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool,
                              device=self.device).unsqueeze(0)
        if self.type == "ppo":
            logits = self.net.actor(self.net._features(x))
            logits = logits.masked_fill(~mask, -1e8)
            return int(logits.argmax(dim=-1).item())
        else:
            mask_t = torch.ones(1, 1, device=self.device)   # not done mid-game
            dist, self.goals, self.states, _, _ = self.net(
                x, self.goals, self.states, mask_t, mask, save=True)
            return int(dist.probs.argmax(dim=-1).item())


def play_game(policy, gnugo, n, komi, seed, max_moves):
    """Agent (black) vs GNU Go (white). Returns (won: 0/1, plies)."""
    env = go_v5.env(board_size=n, komi=komi); env.reset(seed=seed)
    gnugo.reset(); policy.reset()
    black_reward, moves = 0.0, 0
    while env.agents:
        sel = env.agent_selection
        obs, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None); black_reward += float(env.rewards.get(BLACK, 0.0)); continue
        if sel == BLACK:                              # our trained agent
            a = policy.act(obs)
            if obs["action_mask"][a] == 0:            # safety (shouldn't happen)
                a = n * n                             # fall back to pass
            gnugo.play_opponent(a)                    # tell GNU Go our move
            env.step(a)
        else:                                         # white = GNU Go
            a = gnugo.genmove()
            if a == "resign":
                return 1, moves                       # GNU Go resigns -> agent wins
            if obs["action_mask"][a] == 0:
                raise RuntimeError(f"GNU Go move illegal in go_v5: {a}")
            env.step(a)
        black_reward += float(env.rewards.get(BLACK, 0.0))
        moves += 1
        if moves >= max_moves:                        # force area-scoring
            pa, g = n * n, 0
            while env.agents and g < 6:
                c = env.agent_selection
                env.step(None if (env.terminations.get(c, False)
                                  or env.truncations.get(c, False)) else pa)
                g += 1
            break
    return (1 if black_reward > 0 else 0), moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["ppo", "feudal"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--komi", type=float, default=5.5)
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--tag", default="", help="opponent the agent was trained on (for naming)")
    ap.add_argument("--binary", default="gnugo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n = args.board_size
    max_moves = 16 * n * n
    device = torch.device("cpu")
    policy = AgentPolicy(args.agent, args.checkpoint, n, device)

    tag = f"-{args.tag}" if args.tag else ""
    name = f"eval-{args.agent}{tag}-{n}x{n}-vs-gnugo-L{args.level}"
    out = args.out or f"results/{name}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project="honours-rl-go", name=name,
                         config={"agent": args.agent, "trained_vs": args.tag,
                                 "board_size": n, "komi": args.komi,
                                 "gnugo_level": args.level, "games": args.games})

    err_log = f"logs/gnugo_eval/{name}_stderr.log"
    os.makedirs(os.path.dirname(err_log), exist_ok=True)
    gnugo = GnuGo(n, args.komi, level=args.level, color="white",
                  binary=args.binary, stderr_log=err_log)

    print(f"EVAL {args.agent}{tag} ({args.checkpoint}) vs GNU Go L{args.level} "
          f"on {n}x{n}, {args.games} games\n")
    wins = done = skipped = 0
    try:
        while done < args.games:
            try:
                won, plies = play_game(policy, gnugo, n, args.komi,
                                       seed=args.seed + done + skipped, max_moves=max_moves)
            except GTPError as e:
                skipped += 1
                print(f"  ⚠ game skipped (GNU Go crash: {e}); restarting", flush=True)
                gnugo.restart(); continue
            done += 1; wins += won
            wr = wins / done
            if run is not None:
                run.log({"custom/win_rate": wr, "custom/total_episodes": done,
                         "rollout/ep_len_mean": float(plies)}, step=done)
            print(f"  game {done:>4}/{args.games} | {'WIN ' if won else 'loss'} "
                  f"| win_rate {wr:5.1%} | plies {plies}", flush=True)
    finally:
        gnugo.close()

    final = wins / done if done else 0.0
    print(f"\n  FINAL: {args.agent}{tag} vs GNU Go L{args.level} ({n}x{n}): "
          f"{final:.1%} ({wins}/{done})")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "trained_vs", "board_size", "gnugo_level",
                    "games", "wins", "win_rate"])
        w.writerow([args.agent, args.tag, n, args.level, done, wins, round(final, 4)])
    print(f"  wrote {out}")
    if run is not None:
        run.summary["win_rate"] = final; run.finish()


if __name__ == "__main__":
    main()
