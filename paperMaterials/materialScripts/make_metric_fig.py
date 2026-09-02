# ══════════════════════════════════════════════════════════════
# Section 3.4 (Metrics) — annotated metric figure.
#
# One real rolling win-rate segment (feudal 9x9, MED-f1, first shift), with all
# four recovery measures drawn on it: pre-shift baseline + 0.8x band, recovery
# time (span below the band), dip depth (baseline -> lowest point), and
# adaptation cost (shaded area between baseline and the curve).
#
#   python paperMaterials/make_metric_fig.py
# ══════════════════════════════════════════════════════════════

import csv
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)
CSV = os.path.join(ROOT, "models/feudal/9x9/corner-phase2-med-f1/games.csv")
SHIFT_K = 0          # first shift (-> defensive)
W = 100              # rolling window (matches the recovery metric)

SLATE, SAND, SAGE = "#4f7891", "#b9975b", "#7f9e87"
INK, MUTE, PLUM = "#243039", "#6f6f6b", "#7d4f8a"


def rollw(wins, W):
    dq, out = deque(maxlen=W), []
    for w in wins:
        dq.append(w); out.append(sum(dq) / len(dq))
    return out


def main():
    rows = list(csv.DictReader(open(CSV)))
    wins = [int(r["win"]) for r in rows]
    rw = rollw(wins, W)
    sidx = [int(r["shift_idx"]) for r in rows]

    seg = [i for i, s in enumerate(sidx) if s == SHIFT_K]
    prev = [i for i, s in enumerate(sidx) if s == SHIFT_K - 1]
    shift_pos = seg[0]
    baseline = rw[prev[-1]]
    thr = 0.8 * baseline
    segr = [rw[i] for i in seg]
    below = [j for j, v in enumerate(segr) if v < thr]
    fb = below[0]
    ret = next((j for j in range(fb, len(segr)) if segr[j] >= thr), len(segr) - 1)
    # dip = lowest point DURING this disruption (not a later noise dip)
    win = segr[:ret + 1]
    mn = min(win); mn_rel = win.index(mn)

    xrel = np.arange(len(rw)) - shift_pos
    lo, hi = -180, ret + 200
    m = (xrel >= lo) & (xrel <= hi)
    X, Y = xrel[m], np.array(rw)[m]

    fig, ax = plt.subplots(figsize=(11, 4.7))

    # adaptation cost = area between baseline and the curve (post-shift deficit)
    post = X >= 0
    ax.fill_between(X, Y, baseline, where=post & (Y < baseline),
                    color=SAND, alpha=0.28, linewidth=0, zorder=1)

    ax.plot(X, Y, color=SLATE, lw=1.8, zorder=4)

    # baseline + 0.8x band
    ax.axhline(baseline, color=INK, lw=1.2, ls="--", zorder=3)
    ax.axhline(thr, color=SAGE, lw=1.4, ls="--", zorder=3)
    # shift line
    ax.axvline(0, color=MUTE, lw=1.3, ls=(0, (2, 2)), zorder=3)

    # recovery span (games below the band) — double arrow low, clear of the curve
    ry = 0.20
    ax.annotate("", xy=(ret, ry), xytext=(fb, ry),
                arrowprops=dict(arrowstyle="<|-|>", color=SAGE, lw=1.7))
    ax.plot([fb, fb], [ry - 0.03, ry + 0.03], color=SAGE, lw=1.0)
    ax.plot([ret, ret], [ry - 0.03, ry + 0.03], color=SAGE, lw=1.0)
    lblbox = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.5)
    ax.text((fb + ret) / 2, ry - 0.03, "recovery time  (games below the band)",
            ha="center", va="top", fontsize=15, color=SAGE, style="italic",
            bbox=lblbox)

    # disruption: the crossing event itself (recovery/dip/cost only apply once
    # this has happened) — marked where the curve first drops below the band
    ax.plot(fb, segr[fb], marker="o", color=PLUM, markersize=9, zorder=6,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate("disruption", xy=(fb, segr[fb]),
                xytext=(fb + 25, 0.32), ha="left", va="center",
                fontsize=15, color=PLUM, style="italic", bbox=lblbox,
                arrowprops=dict(arrowstyle="-", color=PLUM, lw=1.3))

    # dip depth (baseline -> lowest point), with a short leader pointing at the arrow
    ax.annotate("", xy=(mn_rel, mn), xytext=(mn_rel, baseline),
                arrowprops=dict(arrowstyle="<|-|>", color="#a15b4b", lw=1.7))
    ax.annotate("dip depth", xy=(mn_rel - 6, (baseline + mn) / 2),
                xytext=(mn_rel - 60, (baseline + mn) / 2),
                ha="right", va="center", fontsize=15, color="#a15b4b", style="italic",
                bbox=lblbox, arrowprops=dict(arrowstyle="-", color="#a15b4b", lw=1.3))

    # labels for the horizontal references + shift + cost
    ax.text(hi - 6, baseline + 0.015, "pre-shift baseline", ha="right", va="bottom",
            fontsize=15, color=INK, bbox=lblbox)
    # recovery band: leader points down to the actual dashed line
    ax.annotate(r"recovery band  ($0.8\times$ baseline)", xy=(420, thr),
                xytext=(130, thr + 0.095), ha="left", va="bottom", fontsize=15,
                color="#4f6b5c", weight="bold", bbox=lblbox,
                arrowprops=dict(arrowstyle="-", color="#4f6b5c", lw=1.3))
    ax.text(10, 0.955, "opponent shift", ha="left", va="top", fontsize=15,
            color=MUTE, bbox=lblbox)
    # adaptation cost: leader points into the shaded region itself
    cost_x = mn_rel + 260
    ax.annotate("adaptation cost\n(shaded area)", xy=(cost_x - 90, thr - 0.06),
                xytext=(cost_x, thr + 0.02), ha="center", va="bottom", fontsize=15,
                color="#8a6a33", style="italic", bbox=lblbox,
                arrowprops=dict(arrowstyle="-", color="#8a6a33", lw=1.3))

    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("games since the shift", fontsize=17)
    ax.set_ylabel(f"rolling win rate (window = {W})", fontsize=17)
    ax.tick_params(axis="both", labelsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_metrics_annotated.png")
    fig.savefig(p, dpi=190, bbox_inches="tight"); plt.close(fig)
    print("wrote", p,
          f"(baseline={baseline:.2f}, dip={baseline-mn:.2f}, recovery={ret-fb})")


if __name__ == "__main__":
    main()
