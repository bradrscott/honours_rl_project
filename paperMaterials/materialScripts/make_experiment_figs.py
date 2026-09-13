# ══════════════════════════════════════════════════════════════
# Section 3.3 (Experimental Design) — figures.
#   fig_shift_schedule.png   timeline of the f1/f2/f3 shift schedules (9x9)
#   fig_phase1_phase2_flow.png   Phase 1 -> Phase 2 pipeline (appendix)
#
#   python paperMaterials/make_experiment_figs.py [schedule|flow|all]
# ══════════════════════════════════════════════════════════════

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paperMaterials", "restOfPaperSection")

# muted palette (cohesive with the other Section 3 figures)
SLATE, SLATE_F = "#6b8ea3", "#e7eef2"      # opponent A (pre-shift / trained)
SAND,  SAND_F  = "#b9975b", "#f1e8d6"      # opponent B (post-shift / novel)
SAGE,  SAGE_F  = "#7f9e87", "#e6efe6"
NEUT,  NEUT_F  = "#9aa0a6", "#f5f3ee"
INK, MUTE      = "#243039", "#7c7c78"

# 9x9 Phase-2 schedule (budget 2.3M). Entry (opp, step) = switch to opp at step.
BUDGET = 2_300_000
SCHED = {
    "f1  (single shift)":  [("B", 300_000), ("A", 1_800_000)],
    "f2  (periodic)":      [("B", 300_000), ("A", 800_000),
                            ("B", 1_300_000), ("A", 1_800_000)],
    "f3  (frequent)":      [("B", 300_000), ("A", 500_000), ("B", 700_000),
                            ("A", 900_000), ("B", 1_100_000), ("A", 1_300_000),
                            ("B", 1_500_000), ("A", 1_700_000), ("B", 1_900_000),
                            ("A", 2_100_000)],
}


def segments(sched):
    """(opp, step) list -> list of (opp, start, end) segments; starts on A."""
    segs, cur, start = [], "A", 0
    for opp, step in sched:
        segs.append((cur, start, step)); cur, start = opp, step
    segs.append((cur, start, BUDGET))
    return segs


def fig_shift_schedule():
    # vivid pair for this figure (distinct from the pipeline's blue/amber/teal/purple)
    A_F, A_E = "#d8f2dd", "#2f9e44"   # seen trained opponent (pre-shift)  — green
    B_F, B_E = "#fbd6e4", "#d6336c"   # unseen post-shift opponent          — rose/pink
    fig, ax = plt.subplots(figsize=(11, 3.6))
    rows = list(SCHED.keys())
    h = 0.6
    for i, name in enumerate(rows):
        y = len(rows) - 1 - i
        for opp, s, e in segments(SCHED[name]):
            c = A_F if opp == "A" else B_F
            ec = A_E if opp == "A" else B_E
            ax.add_patch(Rectangle((s, y - h / 2), e - s, h, facecolor=c,
                         edgecolor=ec, linewidth=1.2))
        # shift markers
        for _, step in SCHED[name]:
            ax.plot([step, step], [y - h / 2, y + h / 2], color=INK, lw=0.8, alpha=0.35)
        ax.text(-90_000, y, name, ha="right", va="center", fontsize=10.5, color=INK)

    ax.axvline(0, color=NEUT, lw=1.1, ls=(0, (3, 3)))
    ax.text(0, len(rows) - 0.28, "  Phase 2 begins\n  (resume Phase-1 checkpoint)",
            ha="left", va="bottom", fontsize=8.8, style="italic", color=MUTE)
    ax.set_xlim(-250_000, BUDGET + 60_000)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5e6, 1e6, 1.5e6, 2e6])
    ax.set_xticklabels(["0", "0.5M", "1.0M", "1.5M", "2.0M"])
    ax.set_xlabel("training steps", fontsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    # legend
    ax.add_patch(Rectangle((0.55e6, -0.62), 0.09e6, 0.16, facecolor=A_F,
                 edgecolor=A_E, transform=ax.transData))
    ax.text(0.66e6, -0.54, "seen trained opponent", fontsize=8.8, va="center")
    ax.add_patch(Rectangle((1.35e6, -0.62), 0.09e6, 0.16, facecolor=B_F,
                 edgecolor=B_E, transform=ax.transData))
    ax.text(1.46e6, -0.54, "unseen post-shift opponent", fontsize=8.8, va="center")

    fig.tight_layout()
    p = os.path.join(OUT, "fig_shift_schedule.png")
    fig.savefig(p, dpi=190, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_phase1_phase2_flow():
    fig, ax = plt.subplots(figsize=(13.5, 3.0))
    ax.set_xlim(0, 25.4); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(cx, w, text, fill, edge, h=1.5):
        ax.add_patch(FancyBboxPatch((cx - w / 2, 2.15 - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     facecolor=fill, edgecolor=edge, linewidth=1.6))
        ax.text(cx, 2.15, text, ha="center", va="center", fontsize=9.8, color=INK)

    # vivid, distinct palette for this figure (fill, edge) per box
    BLUE_F,  BLUE_E  = "#d9e8fb", "#2f6fd0"
    AMBER_F, AMBER_E = "#fdecd2", "#e8952f"
    TEAL_F,  TEAL_E  = "#d4f2ea", "#17a589"
    PURP_F,  PURP_E  = "#ecdcf7", "#8e44c9"
    ARROW_C = "#5a6b73"

    def arrow(x1, x2, label=None):
        ax.annotate("", xy=(x2, 2.15), xytext=(x1, 2.15),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.8))
        if label:
            ax.text((x1 + x2) / 2, 2.5, label, ha="center", fontsize=8.5,
                    style="italic", color=MUTE)

    box(3.4, 5.6, "Phase 1\nTrain each agent vs each\nfixed opponent to a stable\nwin-rate baseline",
        BLUE_F, BLUE_E)
    box(10.2, 4.4, "Saved checkpoint\n(the pre-shift baseline\npolicy for each opponent)",
        AMBER_F, AMBER_E)
    box(16.8, 5.2, "Phase 2\nResume checkpoint, apply\nabrupt opponent shifts,\nlog per-game win rate",
        TEAL_F, TEAL_E)
    box(22.2, 3.6, "Metrics\nrecovery time, dip depth,\nadaptation cost,\ndisruption rate",
        PURP_F, PURP_E, h=1.95)
    arrow(6.25, 7.95); arrow(12.45, 14.15); arrow(19.45, 20.35)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_phase1_phase2_flow.png")
    fig.savefig(p, dpi=190, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "schedule"):
        fig_shift_schedule()
    if which in ("all", "flow"):
        fig_phase1_phase2_flow()
    print("done.")
