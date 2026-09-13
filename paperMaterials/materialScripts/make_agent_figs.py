# ══════════════════════════════════════════════════════════════
# Section 3.2 (Agents) — architecture figure + parameter counts.
#
# Writes paperMaterials/fig_agents.png: flat PPO vs Feudal (FuN), side by side,
# sharing the same CNN backbone so the flat-vs-hierarchical contrast is
# immediate. Also prints the parameter counts used in tab_paramcount.tex.
#
#   OPPONENT=greedy BOARD_SIZE=9 KOMI=5.5 python paperMaterials/make_agent_figs.py
# ══════════════════════════════════════════════════════════════

import os
import sys
os.environ.setdefault("OPPONENT", "greedy")
os.environ.setdefault("BOARD_SIZE", "9")
os.environ.setdefault("KOMI", "5.5")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(ROOT, "paperMaterials", "restOfPaperSection")

# muted palette (cohesive with the Section 3.1 figures)
SLATE, SLATE_F = "#6b8ea3", "#e7eef2"      # shared components
SAND,  SAND_F  = "#b9975b", "#f1e8d6"      # left branch  (policy / manager)
SAGE,  SAGE_F  = "#7f9e87", "#e6efe6"      # right branch (value / worker)
NEUT,  NEUT_F  = "#9aa0a6", "#f5f3ee"      # outputs
INK, MUTE      = "#243039", "#7c7c78"


def box(ax, cx, cy, w, h, text, fill, edge, fs=16, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor=fill, edgecolor=edge, linewidth=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=INK, weight=("bold" if bold else "normal"), zorder=5)


def arrow(ax, x1, y1, x2, y2, color=SLATE):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(13.8, 8.0))
    ax.set_xlim(0, 24); ax.set_ylim(2.4, 15.6); ax.axis("off")

    # panel identifiers
    ax.text(6, 15.0, "Flat agent (PPO)", ha="center", fontsize=22,
            weight="bold", color=INK)
    ax.text(17.6, 15.0, "Hierarchical agent (FuN)", ha="center", fontsize=22,
            weight="bold", color=INK)
    ax.plot([12, 12], [0.6, 14.4], ls=(0, (4, 4)), color="#c7ccd1", lw=1.2)

    # ---------------- Flat agent (PPO) ----------------
    px = 6
    box(ax, px, 13.5, 4.0, 0.95, "Board state", SLATE_F, SLATE, fs=18)
    box(ax, px, 11.4, 4.8, 0.95, "CNN encoder", SLATE_F, SLATE, fs=18)
    box(ax, 3.8, 8.9, 3.6, 1.0, "Policy head", SAND_F, SAND, fs=18)
    box(ax, 8.2, 8.9, 3.6, 1.0, "Value head", SAGE_F, SAGE, fs=18)
    box(ax, 3.8, 6.6, 3.8, 1.1, "Move\nprobabilities", NEUT_F, NEUT, fs=17)
    box(ax, 8.2, 6.6, 3.8, 1.1, "Position value", NEUT_F, NEUT, fs=17)
    arrow(ax, px, 13.02, px, 11.9)
    arrow(ax, px, 10.92, 3.8, 9.42); arrow(ax, px, 10.92, 8.2, 9.42)
    arrow(ax, 3.8, 8.38, 3.8, 7.17, SAND)
    arrow(ax, 8.2, 8.38, 8.2, 7.17, SAGE)

    # ---------------- Hierarchical agent (FuN) ----------------
    fx = 17.6
    box(ax, fx, 13.5, 4.0, 0.95, "Board state", SLATE_F, SLATE, fs=18)
    box(ax, fx, 11.4, 4.8, 0.95, "CNN encoder", SLATE_F, SLATE, fs=18)
    mx, wx = 15.0, 20.3
    box(ax, mx, 9.0, 4.3, 1.2, "Manager\n(dilated LSTM)", SAND_F, SAND, fs=18)
    box(ax, wx, 9.0, 4.3, 1.2, "Worker", SAGE_F, SAGE, fs=18)
    box(ax, mx, 6.7, 3.4, 0.95, "Goal", NEUT_F, SAND, fs=18)
    box(ax, wx, 6.65, 3.9, 1.1, "Move\nprobabilities", NEUT_F, SAGE, fs=17)
    box(ax, mx, 4.5, 3.7, 0.9, "Manager value", NEUT_F, NEUT, fs=17)
    box(ax, wx, 4.45, 3.7, 0.9, "Worker value", NEUT_F, NEUT, fs=17)
    arrow(ax, fx, 13.02, fx, 11.9)
    arrow(ax, fx, 10.92, mx, 9.62); arrow(ax, fx, 10.92, wx, 9.62)
    arrow(ax, mx, 8.4, mx, 7.19, SAND)                   # manager -> goal
    arrow(ax, mx + 1.75, 6.7, wx - 2.2, 8.6, SAND)       # goal -> worker
    arrow(ax, wx, 8.4, wx, 7.22, SAGE)                   # worker -> move probs
    arrow(ax, mx, 6.22, mx, 4.96, NEUT)                  # manager -> value
    arrow(ax, wx, 6.09, wx, 4.91, NEUT)                  # worker -> value
    # reward signals (short professional labels)
    ax.text(mx, 3.35, "extrinsic reward", ha="center", va="center",
            fontsize=15, style="italic", color=MUTE)
    ax.text(wx, 3.35, "intrinsic reward", ha="center", va="center",
            fontsize=15, style="italic", color=MUTE)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_agents.png")
    fig.savefig(p, dpi=190, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
