# ══════════════════════════════════════════════════════════════
# Section 4.1 — board-transfer slope chart: 9x9 -> 13x13 win rate.
#
# Two vertical axes ("9x9" left, "13x13" right); one line per
# (agent, opponent) connecting its two baseline win rates. Downward slope =
# dropped on the bigger board, upward = improved. Reads as a mini story per
# opponent while keeping the absolute win rates.
#
# Reads paperMaterials/results/phase1_win_rates.csv.
#   python paperMaterials/make_phase1_scatter.py
# Writes paperMaterials/results/fig_phase1_scatter.png
# ══════════════════════════════════════════════════════════════

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PPO_C, FUN_C = "#2f6f8f", "#b9754a"     # matches the other 4.1 figures
NICE = {"greedy": "Greedy", "defensive": "Defensive", "corner": "Corner",
        "edge": "Edge", "random": "Random"}


def load():
    d = {}
    with open(os.path.join(OUT, "phase1_win_rates.csv")) as f:
        for r in csv.DictReader(f):
            d.setdefault(r["opponent"], {})[r["board"]] = (
                float(r["ppo_win_rate"]), float(r["fun_win_rate"]))
    return d


def spread(vals, gap):
    """Push apart overlapping label y-positions, preserving order & value."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    adj = list(vals)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if adj[j] - adj[i] < gap:
            adj[j] = adj[i] + gap
    return adj


def main():
    data = load()
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    xL, xR = 0.0, 1.0
    GAP = 0.021           # min vertical spacing between labels (win-rate units)
    LX = 0.055            # label horizontal offset from each axis

    # draw the slope lines + endpoint dots
    for agent, color, idx in (("ppo", PPO_C, 0), ("feudal", FUN_C, 1)):
        for opp in NICE:
            y9, y13 = data[opp]["9x9"][idx], data[opp]["13x13"][idx]
            ax.plot([xL, xR], [y9, y13], color=color, lw=1.8, alpha=0.85,
                    zorder=3, solid_capstyle="round")
            ax.scatter([xL, xR], [y9, y13], s=34, color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)

    # de-collided end labels, one ladder per (agent, side)
    for agent, color, idx in (("ppo", PPO_C, 0), ("feudal", FUN_C, 1)):
        opps = list(NICE)
        for side, xv, ha, sign in (("9x9", xL, "right", -1),
                                   ("13x13", xR, "left", 1)):
            ys = [data[o][side][idx] for o in opps]
            adj = spread(ys, GAP)
            for o, y, ya in zip(opps, ys, adj):
                txt = (f"{NICE[o]}  {y*100:.0f}" if side == "9x9"
                       else f"{y*100:.0f}  {NICE[o]}")
                ax.annotate(txt, (xv, y), xytext=(xv + sign * LX, ya),
                            ha=ha, va="center", fontsize=8.3, color=color,
                            zorder=5,
                            arrowprops=dict(arrowstyle="-", color=color,
                                            lw=0.6, alpha=0.5,
                                            shrinkA=1, shrinkB=3))

    # two vertical axes
    for xv in (xL, xR):
        ax.axvline(xv, color="#c7ccd1", lw=1.3, zorder=1)
    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(0.5, 1.03)
    ax.set_xticks([xL, xR])
    ax.set_xticklabels([r"$9\times9$", r"$13\times13$"], fontsize=13,
                       fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylabel("baseline win rate (%)", fontsize=11)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([50, 60, 70, 80, 90, 100], fontsize=9)
    ax.grid(axis="y", color="#eef0f2", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_visible(False)

    handles = [Line2D([0], [0], color=PPO_C, lw=2.4, marker="o",
                      markerfacecolor=PPO_C, markeredgecolor="white",
                      markersize=7, label="PPO"),
               Line2D([0], [0], color=FUN_C, lw=2.4, marker="o",
                      markerfacecolor=FUN_C, markeredgecolor="white",
                      markersize=7, label="FuN")]
    ax.legend(handles=handles, fontsize=10, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_phase1_scatter.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
