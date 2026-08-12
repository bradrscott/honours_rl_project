# ══════════════════════════════════════════════════════════════
# LIVE DEMO: watch a trained agent play a full game of Go
#
# Pops a pygame window (go_v5 render_mode="human") and plays a trained
# PPO or feudal agent (BLACK, the seat it trained in) against one of the
# frozen heuristic bots (WHITE), one move at a time with a delay so an
# audience can follow the stones going down. Prints a running move log +
# the final area score / winner to the terminal.
#
# Runs entirely on your Mac (CPU) — needs only the local checkpoint.
#
# Examples:
#   # PPO that trained vs greedy, playing greedy, ~0.7s per move:
#   python demo_live_game.py --agent ppo --trained-vs greedy --delay 0.7
#
#   # feudal that trained vs defensive, playing a DIFFERENT bot (corner):
#   python demo_live_game.py --agent feudal --trained-vs defensive --opponent corner
#
#   # pick the checkpoint explicitly:
#   python demo_live_game.py --agent ppo \
#       --checkpoint models/ppo_go/9x9/greedy/ppo_go_final.pt --opponent greedy
# ══════════════════════════════════════════════════════════════

import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The training configs read OPPONENT / BOARD_SIZE / KOMI from the env at import
# time, so set them BEFORE importing anything that pulls those configs in.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--board-size", type=int, default=9)
_pre.add_argument("--komi", type=float, default=5.5)
_pre.add_argument("--opponent", default=None)
_pre.add_argument("--trained-vs", default=None)
_a, _ = _pre.parse_known_args()
os.environ.setdefault("OPPONENT", _a.opponent or _a.trained_vs or "greedy")
os.environ["BOARD_SIZE"] = str(_a.board_size)
os.environ["KOMI"] = str(_a.komi)

import numpy as np
import torch
from pettingzoo.classic import go_v5

from opponents.factory import make_opponent, OPPONENT_NAMES
from ppo.ppo_go import ActorCritic
from ppo.config_ppo_go import N_CHANNELS, CNN_FILTERS, CNN_LAYERS, HIDDEN_DIM
from feudal.feudalNetwork import FeudalNetwork, init_hidden
from feudal.config_feudal import HIDDEN_DIM_M, HIDDEN_DIM_W, TIME_HORIZON, DILATION

BLACK, WHITE = "black_0", "white_0"


def obs_chw(obs_dict):
    o = obs_dict["observation"].astype(np.float32)
    return np.transpose(o, (2, 0, 1)).copy()


class AgentPolicy:
    """Greedy (argmax) policy wrapper around a loaded PPO or feudal net.
    Mirrors gnugo/eval_agent_vs_gnugo.py so demo play == eval play."""

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
                dilation=DILATION, eps=0.0,
                num_workers=1, device=str(device),
                n_filters=CNN_FILTERS, n_layers=CNN_LAYERS)
        else:
            raise ValueError(agent_type)
        self.net.load_state_dict(torch.load(ckpt, map_location=device))
        self.net.to(device).eval()

    def reset(self):
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
            mask_t = torch.ones(1, 1, device=self.device)
            dist, self.goals, self.states, _, _ = self.net(
                x, self.goals, self.states, mask_t, mask, save=True)
            return int(dist.probs.argmax(dim=-1).item())


def fmt_move(n, a):
    if a == n * n:
        return "PASS"
    r, c = a // n, a % n
    return f"{chr(ord('A') + c)}{n - r}"   # GTP-ish coordinate


def hold_result(title, subtitle, hold_seconds=0):
    """Draw a result banner over the final board and keep the window open.
    hold_seconds=0 waits for a key/click/close; >0 auto-closes after a timeout."""
    import pygame
    surf = pygame.display.get_surface()
    if surf is None:
        return
    W, H = surf.get_size()
    pygame.font.init()
    bh = max(96, H // 6)
    overlay = pygame.Surface((W, bh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    tfont = pygame.font.SysFont(None, max(34, W // 16))
    sfont = pygame.font.SysFont(None, max(20, W // 32))
    t = tfont.render(title, True, (255, 255, 255))
    s = sfont.render(subtitle, True, (220, 220, 220))
    overlay.blit(t, ((W - t.get_width()) // 2, bh // 2 - t.get_height()))
    overlay.blit(s, ((W - s.get_width()) // 2, bh // 2 + 6))
    surf.blit(overlay, (0, (H - bh) // 2))
    pygame.display.flip()

    clock = pygame.time.get_ticks()
    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
        if hold_seconds and (pygame.time.get_ticks() - clock) >= hold_seconds * 1000:
            waiting = False
        pygame.time.wait(40)


def default_ckpt(agent, trained_vs, n):
    root = "ppo_go" if agent == "ppo" else "feudal"
    fname = "ppo_go_final.pt" if agent == "ppo" else "feudal_go_final.pt"
    return f"models/{root}/{n}x{n}/{trained_vs}/{fname}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["ppo", "feudal"])
    ap.add_argument("--trained-vs", default=None,
                    help="which bot the checkpoint trained on (picks the default checkpoint)")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit checkpoint path (overrides --trained-vs)")
    ap.add_argument("--opponent", default=None,
                    help=f"live opponent to play now ({'|'.join(OPPONENT_NAMES)}); "
                         "defaults to --trained-vs")
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--komi", type=float, default=5.5)
    ap.add_argument("--delay", type=float, default=0.6,
                    help="seconds between moves (so the audience can follow)")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true",
                    help="skip the pygame window (terminal log only)")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="seconds to keep the final board up after each game "
                         "(0 = wait for a key/click before closing)")
    args = ap.parse_args()

    n = args.board_size
    if not args.trained_vs and not args.checkpoint:
        args.trained_vs = "greedy"
    ckpt = args.checkpoint or default_ckpt(args.agent, args.trained_vs, n)
    if not os.path.exists(ckpt):
        sys.exit(f"✗ checkpoint not found: {ckpt}")
    opp_name = args.opponent or args.trained_vs or "greedy"

    device = torch.device("cpu")
    policy = AgentPolicy(args.agent, ckpt, n, device)
    opponent = make_opponent(opp_name, n, epsilon=0.0)

    render_mode = None if args.no_render else "human"
    print(f"\n  {args.agent.upper()} (BLACK, trained vs {args.trained_vs or '?'})"
          f"  vs  {opp_name} bot (WHITE)   on {n}x{n}, komi {args.komi}")
    print(f"  checkpoint: {ckpt}\n")

    agent_wins = 0
    for g in range(args.games):
        env = go_v5.env(board_size=n, komi=args.komi, render_mode=render_mode)
        env.reset(seed=args.seed + g)
        policy.reset()
        black_reward, ply = 0.0, 0
        max_plies = 16 * n * n

        while env.agents:
            sel = env.agent_selection
            obs, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                black_reward += float(env.rewards.get(BLACK, 0.0))
                continue

            if sel == BLACK:
                a = policy.act(obs)
                if obs["action_mask"][a] == 0:
                    a = n * n
                who = f"{args.agent.upper()}"
            else:
                a = int(opponent.select_action(obs))
                if obs["action_mask"][a] == 0:
                    a = n * n
                who = f"{opp_name} bot"

            env.step(a)
            black_reward += float(env.rewards.get(BLACK, 0.0))
            ply += 1
            print(f"  move {ply:>4} | {who:<12} -> {fmt_move(n, a)}")
            if render_mode:
                time.sleep(args.delay)
            if ply >= max_plies:
                # force area-scoring by passing both out
                gpass = 0
                while env.agents and gpass < 6:
                    c = env.agent_selection
                    done_c = env.terminations.get(c, False) or env.truncations.get(c, False)
                    env.step(None if done_c else n * n)
                    gpass += 1
                break

        won = black_reward > 0
        drawish = black_reward == 0
        agent_wins += int(won)
        if won:
            title = f"{args.agent.upper()} (BLACK) WINS"
        elif drawish:
            title = "NO RESULT (move cap)"
        else:
            title = f"{opp_name} bot (WHITE) WINS"
        bar = "═" * 46
        print(f"\n  {bar}\n   GAME {g + 1}/{args.games}:  {title}"
              f"   (reward {black_reward:+.0f}, {ply} moves)\n  {bar}\n")
        if render_mode:
            hold_result(title, "press any key to close" if not args.hold
                        else f"next in {args.hold:.0f}s…", hold_seconds=args.hold)
        env.close()

    if args.games > 1:
        print(f"  AGENT record: {agent_wins}/{args.games} "
              f"({agent_wins / args.games:.0%})\n")


if __name__ == "__main__":
    main()
