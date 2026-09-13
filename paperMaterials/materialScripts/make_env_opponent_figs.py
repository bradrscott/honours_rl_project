# ══════════════════════════════════════════════════════════════
# Section 3.1 (Environment & Opponents) — paper figures.
#
# Generates, into paperMaterials/:
#   fig_boards.png            9x9 vs 13x13 boards side by side
#   fig_obs_encoding.png      the 17-plane observation stack (schematic)
#   fig_opponents_5panel.png  one representative board per bot style
#   fig_opponent_heatmaps.png move-density heatmap per bot (where it plays)
#
# Opponents shown (paper set, aggressive excluded): greedy, defensive, corner,
# edge, random. Run from the repo root:
#   OPPONENT=greedy BOARD_SIZE=9 KOMI=5.5 python paperMaterials/make_env_opponent_figs.py
# (OPPONENT is only needed to satisfy the config import; it is not used here.)
# ══════════════════════════════════════════════════════════════

import os
import sys
os.environ.setdefault("OPPONENT", "greedy")
os.environ.setdefault("BOARD_SIZE", "9")
os.environ.setdefault("KOMI", "5.5")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pettingzoo.classic import go_v5

from opponents.factory import make_opponent

OUT = os.path.join(ROOT, "paperMaterials", "restOfPaperSection")
OPPS = ["greedy", "defensive", "corner", "edge", "random"]
NICE = {"greedy": "Greedy", "defensive": "Defensive", "corner": "Corner",
        "edge": "Edge", "random": "Random"}


def play_and_record(bot_name, n, komi, n_games, max_plies, seed0=0):
    """Bot plays BLACK vs a random opponent; record the bot's (r,c) moves.
    Returns the density grid NxN (pass moves ignored)."""
    grid = np.zeros((n, n), dtype=float)
    bot = make_opponent(bot_name, n, epsilon=0.0)
    opp = make_opponent("random", n)
    for g in range(n_games):
        env = go_v5.env(board_size=n, komi=komi)
        env.reset(seed=seed0 + g)
        ply = 0
        while env.agents and ply < max_plies:
            sel = env.agent_selection
            obs, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None); continue
            if sel == "black_0":                       # the bot we profile
                a = int(bot.select_action(obs))
                if a != n * n:                         # ignore pass for density
                    grid[a // n, a % n] += 1
            else:
                a = int(opp.select_action(obs))
            if obs["action_mask"][a] == 0:
                a = n * n
            env.step(a); ply += 1
        env.close()
    return grid


def rep_board(bot_name, n, komi, plies, seed):
    """A SHORT game (bot=black vs random) rendered to show early placement style
    without the board saturating. Returns an rgb image."""
    bot = make_opponent(bot_name, n, epsilon=0.0)
    opp = make_opponent("random", n)
    env = go_v5.env(board_size=n, komi=komi, render_mode="rgb_array")
    env.reset(seed=seed)
    ply = 0
    while env.agents and ply < plies:
        sel = env.agent_selection
        obs, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None); continue
        a = int(bot.select_action(obs) if sel == "black_0" else opp.select_action(obs))
        if obs["action_mask"][a] == 0:
            a = n * n
        env.step(a); ply += 1
    img = np.asarray(env.render())
    env.close()
    return img


def random_board_image(n, komi, plies, seed=0):
    """A board with some random-legal stones, for the size-comparison figure."""
    env = go_v5.env(board_size=n, komi=komi, render_mode="rgb_array")
    env.reset(seed=seed)
    for _ in range(plies):
        if not env.agents:
            break
        obs, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None); continue
        legal = np.where(obs["action_mask"] == 1)[0]
        legal = legal[legal != n * n]                   # avoid pass so it fills
        a = int(np.random.choice(legal)) if len(legal) else n * n
        env.step(a)
    img = np.asarray(env.render())
    env.close()
    return img


# ---------------------------------------------------------------- fig: boards
def fig_boards():
    np.random.seed(0)
    img9 = random_board_image(9, 5.5, 40, seed=1)
    img13 = random_board_image(13, 7.5, 80, seed=1)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    for ax, img, ttl in [(axes[0], img9, "9×9"),
                         (axes[1], img13, "13×13")]:
        ax.imshow(img); ax.set_title(ttl, fontsize=16, pad=4); ax.axis("off")
    fig.subplots_adjust(wspace=0.03, top=0.94, bottom=0.01, left=0.01, right=0.99)
    p = os.path.join(OUT, "fig_boards.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------- fig: obs encoding
def fig_obs_encoding():
    """Isometric stack of the 17 feature planes. Muted palette; a small colour
    key in the empty corner — no arrows and no text over the planes."""
    from matplotlib.patches import Rectangle
    OPP, OWN, TURN = "#6b8ea3", "#c2a878", "#8ba888"       # muted steel / sand / sage
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 12); ax.axis("off")
    dx, dy, w, h = 0.34, 0.32, 4.3, 4.3
    for i in range(16, -1, -1):                            # back (16) to front (0)
        x, y = 1 + i * dx, 1 + i * dy
        c = TURN if i == 16 else (OWN if i % 2 == 1 else OPP)   # 0=opp,1=own,...
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="white", lw=1.3))
    # faint grid on the front plane so it reads as an N×N board
    fx, fy = 1, 1
    for k in range(1, 5):
        ax.plot([fx + k * w / 5]*2, [fy, fy + h], color="white", lw=0.5, alpha=0.45)
        ax.plot([fx, fx + w], [fy + k * h / 5]*2, color="white", lw=0.5, alpha=0.45)
    # colour key — placed in the empty upper-left, not over the planes
    lx, ly = 0.3, 11.4
    for col, lab in [(OPP, "opponent stones"), (OWN, "own stones"),
                     (TURN, "colour to play")]:
        ax.add_patch(Rectangle((lx, ly), 0.55, 0.55, facecolor=col,
                               edgecolor="white", lw=1))
        ax.text(lx + 0.75, ly + 0.28, lab, va="center", fontsize=10.5, color="#333")
        ly -= 0.85
    p = os.path.join(OUT, "fig_obs_encoding.png")
    fig.savefig(p, dpi=190, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


# --------------------------------------------- fig: opponents 5-panel + heatmaps
def fig_opponents_and_heatmaps(n=9, komi=5.5, n_games=40, max_plies=140):
    grids, imgs = {}, {}
    for name in OPPS:
        grids[name] = play_and_record(name, n, komi, n_games, max_plies, seed0=100)
        imgs[name] = rep_board(name, n, komi, plies=34, seed=7)   # short => shows style
        print(f"  {name}: recorded {int(grids[name].sum())} moves over {n_games} games")

    # 5-panel representative boards
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, name in zip(axes, OPPS):
        ax.imshow(imgs[name]); ax.set_title(NICE[name], fontsize=24); ax.axis("off")
    fig.tight_layout()
    p1 = os.path.join(OUT, "fig_opponents_5panel.png")
    fig.savefig(p1, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", p1)

    # move-density heatmaps
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, name in zip(axes, OPPS):
        d = grids[name]
        d = d / d.max() if d.max() > 0 else d
        im = ax.imshow(d, cmap="magma", vmin=0, vmax=1)
        ax.set_title(NICE[name], fontsize=24)
        ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cbar.set_label("relative move frequency", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    p2 = os.path.join(OUT, "fig_opponent_heatmaps.png")
    fig.savefig(p2, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", p2)


# ------------------------------------ fig: opponents + heatmaps, ONE figure
def fig_opponents_combined(n=9, komi=5.5, n_games=40, max_plies=140):
    grids, imgs = {}, {}
    for name in OPPS:
        grids[name] = play_and_record(name, n, komi, n_games, max_plies, seed0=100)
        imgs[name] = rep_board(name, n, komi, plies=34, seed=7)
        print(f"  {name}: recorded {int(grids[name].sum())} moves over {n_games} games")

    ncols = len(OPPS)
    fig = plt.figure(figsize=(15, 8.3))
    gs = fig.add_gridspec(2, ncols + 1, width_ratios=[1] * ncols + [0.09],
                           wspace=0.04, hspace=0.06)
    axes = np.empty((2, ncols), dtype=object)
    for col, name in enumerate(OPPS):
        ax = fig.add_subplot(gs[0, col])
        axes[0, col] = ax
        ax.imshow(imgs[name]); ax.set_aspect("equal")
        ax.set_title(NICE[name], fontsize=22)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        d = grids[name]
        d = d / d.max() if d.max() > 0 else d
        ax2 = fig.add_subplot(gs[1, col])
        axes[1, col] = ax2
        im = ax2.imshow(d, cmap="magma", vmin=0, vmax=1, aspect="equal")
        ax2.set_xticks([]); ax2.set_yticks([])

    axes[0, 0].set_ylabel("example position", fontsize=20)
    axes[1, 0].set_ylabel("move-density\nheatmap", fontsize=20)

    cax = fig.add_subplot(gs[1, ncols])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("relative move frequency", fontsize=18, labelpad=10)
    cbar.ax.tick_params(labelsize=15)

    p = os.path.join(OUT, "fig_opponents_combined.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", pad_inches=0.25); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "boards"):
        fig_boards()
    if which in ("all", "obs"):
        fig_obs_encoding()
    if which in ("all", "opp"):
        fig_opponents_and_heatmaps()
    if which == "combined":
        fig_opponents_combined()
    print("done.")
