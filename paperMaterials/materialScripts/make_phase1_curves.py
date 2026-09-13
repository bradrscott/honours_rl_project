# ══════════════════════════════════════════════════════════════
# Section 4.1 — Phase-1 training-curve figure, W&B-style.
#
# Rolling win rate (custom/win_rate) over training steps, PPO and FuN
# overlaid per opponent, faceted by opponent, one figure per board.
# Mimics a native W&B line panel: faint raw trace + bold EMA-smoothed
# line (debiased exponential smoothing, matching W&B's default).
#
# Pulls the CANONICAL (keep) Phase-1 run per (board, agent, opponent):
# the most-recent = the fixed-code 31-Jul runs, not the pre-fix dups.
#
#   python paperMaterials/make_phase1_curves.py
#
# Writes into paperMaterials/results/:
#   fig_phase1_curves_9x9.png
#   fig_phase1_curves_13x13.png
#   fig_phase1_curves.png          (both boards stacked, 2x5)
# ══════════════════════════════════════════════════════════════

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paperMaterials", "resultsSection")
os.makedirs(OUT, exist_ok=True)
ENTITY, PROJECT = "bradrscott4-university-of-cape-town", "honours-rl-go"

OPPS = ["greedy", "defensive", "corner", "edge", "random"]
NICE = {o: o.capitalize() for o in OPPS}
BOARDS = ["9x9", "13x13"]
PPO_C, FUN_C = "#6a1b9a", "#2ec4d6"   # deep purple (dark) / bright cyan (light) — grayscale-safe
SMOOTH = 0.9                          # W&B default-style smoothing weight
GNU_C = "#5f7d6a"
GNUGO = {"9x9": {o: 1.00 for o in OPPS},
         "13x13": {"greedy": 1.00, "defensive": 1.00, "corner": 0.99,
                   "edge": 1.00, "random": 1.00}}


def ema_debiased(y, w):
    """W&B-style exponential moving average with zero-debiasing."""
    y = np.asarray(y, float)
    out = np.empty_like(y)
    last, deb = 0.0, 0.0
    for i, v in enumerate(y):
        last = last * w + (1 - w) * v
        deb = deb * w + (1 - w)
        out[i] = last / deb if deb > 0 else v
    return out


def canonical_runs():
    """Most-recent Phase-1 run per (board, agent, opponent) = fixed-code keep-runs."""
    api = wandb.Api()
    best = {}
    for r in api.runs(f"{ENTITY}/{PROJECT}"):
        c = r.config
        if c.get("meta_phase") != "phase1":
            continue
        o = c.get("meta_opponent")
        if o not in OPPS:
            continue
        key = (c.get("meta_board"), c.get("meta_agent"), o)
        if r.summary.get("custom/win_rate") is None:
            continue
        if key not in best or r.created_at > best[key][0]:
            best[key] = (r.created_at, r)
    return best


def get_curve(run):
    h = run.history(keys=["custom/win_rate"], samples=2000)
    h = h.dropna(subset=["custom/win_rate"]).sort_values("_step")
    return h["_step"].to_numpy(), h["custom/win_rate"].to_numpy()


def draw_board(axes, board, best, show_row_label=True):
    for col, o in enumerate(OPPS):
        ax = axes[col]
        for agent, c in (("ppo", PPO_C), ("feudal", FUN_C)):
            v = best.get((board, agent, o))
            if not v:
                continue
            x, y = get_curve(v[1])
            if len(x) == 0:
                continue
            xm = x / 1e6
            ax.plot(xm, y, color=c, lw=0.8, alpha=0.18)              # raw
            ax.plot(xm, ema_debiased(y, SMOOTH), color=c, lw=1.7,
                    label={"ppo": "PPO", "feudal": "FuN"}[agent])    # smoothed
        ax.axhline(GNUGO[board][o], color=GNU_C, lw=1.0,
                   ls=(0, (3, 2)), alpha=0.7)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.grid(True, color="#e9ecef", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(NICE[o], fontsize=14)
        if col == 0 and show_row_label:
            ax.set_ylabel(f"{board}\nwin rate", fontsize=13)
        elif col == 0:
            ax.set_ylabel("win rate", fontsize=13)
        ax.set_xlabel("steps (M)", fontsize=12)
        ax.tick_params(labelsize=11)


def per_board(best):
    for board in BOARDS:
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
        draw_board(axes, board, best, show_row_label=False)
        axes[0].legend(fontsize=12, loc="lower right", frameon=False)
        fig.tight_layout()
        p = os.path.join(OUT, f"fig_phase1_curves_{board}.png")
        fig.savefig(p, dpi=190, bbox_inches="tight"); plt.close(fig)
        print("wrote", p)


def stacked(best):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=True)
    for row, board in enumerate(BOARDS):
        draw_board(axes[row], board, best, show_row_label=True)
        if row == 0:
            for ax in axes[row]:
                ax.set_xlabel("")
    axes[0, 0].legend(fontsize=9, loc="lower right", frameon=False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_phase1_curves.png")
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    best = canonical_runs()
    print(f"resolved {len(best)} canonical (board,agent,opponent) runs")
    for k in sorted(best):
        print("  ", k, "->", best[k][1].id, best[k][1].created_at)
    per_board(best)
    stacked(best)
    print("done.")
