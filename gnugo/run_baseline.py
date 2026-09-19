# ══════════════════════════════════════════════════════════════
# GNU Go BASELINE — external reference for RQ3 (proposal 5.3.1)
#
# GNU Go plays the BLACK seat (the seat PPO/feudal occupy) against ONE
# heuristic opponent (WHITE), with go_v5 refereeing + area-scoring. Logs to
# wandb with the SAME metric names as the agents, and maps the games across a
# 0->SPAN step axis so the baseline overlays the agents' 5M-step charts as a
# flat reference line. GNU Go does NOT learn — this is a fixed reference, so we
# play a feasible number of GAMES (not 5M literal plies, which would take weeks
# at level 10). One opponent per run, mirroring the agent pipeline.
#
# CPU-only. Needs the gnugo binary:  conda install -c conda-forge gnugo
#
# Usage (one opponent per run; OPPONENT via env like the agents, or --opponent):
#     OPPONENT=greedy python gnugo/run_baseline.py --games 300
#     python gnugo/run_baseline.py --opponent greedy --games 1 --verify   # smoke
# ══════════════════════════════════════════════════════════════

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import deque
from pettingzoo.classic import go_v5

from opponents import make_opponent, OPPONENT_NAMES
from gnugo.engine import GnuGo
from gnugo.gtp import GTPError

BLACK, WHITE = "black_0", "white_0"


def stones_from_obs(obs, agent, n):
    board = obs["observation"]
    own, opp = board[:, :, 1], board[:, :, 0]
    black_plane, white_plane = (own, opp) if agent == BLACK else (opp, own)

    def acts(plane):
        ys, xs = np.where(plane == 1)
        return {int(r) * n + int(c) for r, c in zip(ys.tolist(), xs.tolist())}

    return acts(black_plane), acts(white_plane)


def force_finish(env, n):
    pass_action, guard = n * n, 0
    while env.agents and guard < 6:
        a = env.agent_selection
        if env.terminations.get(a, False) or env.truncations.get(a, False):
            env.step(None)
        else:
            env.step(pass_action)
        guard += 1


def play_game(gnugo, bot, n, komi, seed, max_moves, verify):
    """One game: GNU Go (black) vs bot (white). Returns (won: 0/1, plies)."""
    env = go_v5.env(board_size=n, komi=komi)
    env.reset(seed=seed)
    gnugo.reset()
    moves = 0
    black_reward = 0.0        # ACCUMULATE — env.rewards is zeroed once the
                              # terminated agent is drained with step(None),
                              # so reading it after the loop misses the ±1.
    while env.agents:
        agent = env.agent_selection
        obs, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            black_reward += float(env.rewards.get(BLACK, 0.0))
            continue

        if agent == BLACK:                       # GNU Go
            a = gnugo.genmove()
            if a == "resign":
                return 0, moves                  # black resigns -> loss
            if obs["action_mask"][a] == 0:
                raise RuntimeError(
                    f"GNU Go move illegal in go_v5: {a} "
                    f"({gnugo.action_to_vertex(a)}) — coord/rules mismatch.")
            env.step(a)
        else:                                    # heuristic (white)
            a = bot.select_action(obs)
            gnugo.play_opponent(a)
            env.step(a)
        black_reward += float(env.rewards.get(BLACK, 0.0))

        moves += 1
        if verify and env.agents:
            nobs, _, nterm, _, _ = env.last()
            if not nterm:
                b, w = stones_from_obs(nobs, env.agent_selection, n)
                gnugo.verify_sync(b, w)
        if moves >= max_moves:
            force_finish(env, n)
            break

    return (1 if black_reward > 0 else 0), moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default=os.environ.get("OPPONENT", "greedy"),
                    choices=OPPONENT_NAMES)
    ap.add_argument("--games", type=int, default=300, help="games to play (a few hundred is plenty for a fixed baseline)")
    ap.add_argument("--level", type=int, default=10, help="GNU Go strength 0-10")
    ap.add_argument("--board-size", type=int, default=13)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--window", type=int, default=200, help="rolling win-rate window (games)")
    ap.add_argument("--span", type=int, default=5_000_000,
                    help="x-axis steps the games are mapped across, so the flat "
                         "baseline overlays the agents' 5M-step charts")
    ap.add_argument("--binary", default="gnugo")
    ap.add_argument("--verify", action="store_true", help="board-sync check every move")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n, opp = args.board_size, args.opponent
    max_moves = 16 * n * n
    out = args.out or f"results/gnugo_baseline_{n}x{n}_vs_{opp}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    run = None
    if not args.no_wandb:
        import wandb
        log_dir = f"./logs/gnugo/{opp}/"
        os.makedirs(log_dir, exist_ok=True)
        run = wandb.init(
            project="honours-rl-go",
            name=f"gnugo-{n}x{n}-vs-{opp}",
            dir=log_dir,
            config={"agent": "gnugo", "board_size": n, "komi": args.komi,
                    "opponent": opp, "gnugo_level": args.level,
                    "games": args.games, "window": args.window},
        )

    print(f"GNU Go (black, level {args.level}) vs {opp} on {n}x{n}, "
          f"{args.games} games\n")

    err_log = f"logs/gnugo/{opp}/gnugo_stderr.log"
    os.makedirs(os.path.dirname(err_log), exist_ok=True)
    gnugo = GnuGo(n, args.komi, level=args.level, color="black",
                  binary=args.binary, stderr_log=err_log)
    win_window = deque(maxlen=args.window)
    len_window = deque(maxlen=args.window)
    wins = completed = skipped = consec_fail = attempt = 0
    try:
        while completed < args.games:
            # GNU Go can crash on messy positions; skip+restart rather than die
            try:
                won, plies = play_game(gnugo, make_opponent(opp, n, epsilon=0.0),
                                       n, args.komi, seed=args.seed + attempt,
                                       max_moves=max_moves,
                                       verify=args.verify or completed == 0)
            except GTPError as e:
                skipped += 1
                consec_fail += 1
                attempt += 1
                print(f"  ⚠ game skipped — GNU Go crashed ({e}); see {err_log}. "
                      f"Restarting engine…", flush=True)
                if consec_fail >= 10:
                    print(f"  ✗ 10 consecutive crashes vs {opp} — aborting this "
                          f"opponent. Check {err_log}.", flush=True)
                    break
                gnugo.restart()
                continue

            consec_fail = 0
            attempt += 1
            completed += 1
            wins += won
            win_window.append(won)
            len_window.append(plies)
            win_rate = float(np.mean(win_window))
            if run is not None:
                step = int(completed / args.games * args.span)   # overlay 0..5M axis
                run.log({
                    "custom/win_rate":       win_rate,
                    "custom/total_episodes": completed,
                    "rollout/ep_rew_mean":   2.0 * win_rate - 1.0,   # +1 win / -1 loss
                    "rollout/ep_len_mean":   float(np.mean(len_window)),
                }, step=step)
            print(f"  game {completed:>4}/{args.games} | result "
                  f"{'WIN ' if won else 'loss'} | "
                  f"win_rate(last{args.window}) {win_rate:5.1%} | "
                  f"epLen {np.mean(len_window):5.1f}"
                  f"{f'  (skipped {skipped})' if skipped else ''}", flush=True)
    finally:
        gnugo.close()

    final_wr = wins / completed if completed else 0.0
    print(f"\n  FINAL: GNU Go vs {opp}: {final_wr:.1%}  ({wins}/{completed})"
          f"{f'  [{skipped} games skipped due to engine crashes]' if skipped else ''}")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "games", "gnugo_wins", "gnugo_win_rate",
                    "level", "board_size", "komi"])
        w.writerow([opp, completed, wins, round(final_wr, 4),
                    args.level, n, args.komi])
    print(f"  ✓ wrote {out}")
    if run is not None:
        run.summary["gnugo_win_rate"] = final_wr
        run.finish()


if __name__ == "__main__":
    main()
