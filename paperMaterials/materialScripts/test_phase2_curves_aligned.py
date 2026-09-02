# ══════════════════════════════════════════════════════════════
# TEST ONLY — does not touch make_phase2_results.py or the real
# fig_phase2_curves_*.png files. Tries aligning each seed's curve to ITS
# OWN first-shift step (instead of absolute training step) before
# averaging, to check whether that sharpens FuN's dips on 13x13, which
# look smeared out in the current (absolute-step) version.
#
#   python paperMaterials/test_phase2_curves_aligned.py
# Writes paperMaterials/results/fig_phase2_curves_13x13_TEST_aligned.png
# ══════════════════════════════════════════════════════════════

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperMaterials.materialScripts.make_phase2_results as m

OUT = m.OUT
BOARD = "13x13"
AGENT = "feudal"


def _load_full(g):
    step, roll, opp, gsince, sidx = [], [], [], [], []
    with open(g) as f:
        for r in csv.DictReader(f):
            step.append(int(r["global_step"]))
            roll.append(float(r["rolling"]))
            opp.append(r["opponent"])
            gsince.append(int(r["games_since_shift"]))
            sidx.append(int(r["shift_idx"]))
    return (np.array(step), np.array(roll), np.array(opp),
            np.array(gsince), np.array(sidx))


def _aligned_curve(files, lo=-2.0e5, hi=1.2e6):
    """Shift each seed's step axis so its FIRST shift (shift_idx 0 -> 0
    boundary) sits at x=0, then interpolate onto a common relative grid."""
    curves = []
    for g in files:
        step, roll, opp, gsince, sidx = _load_full(g)
        first_shift_idx = np.where(sidx == 0)[0]
        if len(first_shift_idx) == 0:
            continue
        zero_step = step[first_shift_idx[0]]
        rel = step - zero_step
        mask = (rel >= lo) & (rel <= hi)
        if mask.sum() < 2:
            continue
        curves.append((rel[mask], roll[mask]))
    if not curves:
        return None, None
    grid = np.linspace(lo, hi, 700)
    stack = [np.interp(grid, rel, roll) for rel, roll in curves]
    return grid, np.mean(stack, axis=0)


def _aligned_b_interval(files, lo=-2.0e5, hi=1.2e6):
    """Opponent-B interval in the SAME relative coordinate, from one
    reference seed (first file)."""
    step, roll, opp, gsince, sidx = _load_full(files[0])
    first_shift_idx = np.where(sidx == 0)[0]
    if len(first_shift_idx) == 0:
        return []
    zero_step = step[first_shift_idx[0]]
    rel = step - zero_step
    B = None
    ivs, start = [], None
    for i in range(len(rel)):
        if rel[i] < lo or rel[i] > hi:
            continue
        if sidx[i] == 0:
            if B is None:
                B = opp[i]
            if start is None:
                start = rel[i]
        elif start is not None:
            ivs.append((start, rel[i])); start = None
    if start is not None:
        ivs.append((start, min(rel[-1], hi)))
    return ivs


def main():
    fig, axes = plt.subplots(len(m.MAGS), len(m.FREQS), figsize=(15, 5.6),
                             sharex=True, sharey=True)
    scale = 1e6
    for r, mag in enumerate(m.MAGS):
        for c, freq in enumerate(m.FREQS):
            ax = axes[r, c]
            files = m._condition_files(BOARD, AGENT, mag, freq)
            for iv in _aligned_b_interval(files):
                ax.axvspan(iv[0] / scale, iv[1] / scale, color=m.BAND,
                           zorder=0, lw=0)
            grid, mean = _aligned_curve(files)
            if grid is not None:
                ax.plot(grid / scale, mean, color=m.FUN_C, lw=1.5, zorder=3)
            ax.set_ylim(0, 1.02)
            ax.grid(True, color=m.GRID, lw=0.7, zorder=1)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=12)
            if r == 0:
                ax.set_title(f"{freq}", fontsize=17)
            if c == 0:
                A, B = m.AB[mag]
                ax.set_ylabel(f"{m.MAG_NICE[mag]}\n({A}→{B})\nwin rate",
                              fontsize=13)
            if r == len(m.MAGS) - 1:
                ax.set_xlabel("steps since first shift (M)", fontsize=14)
    legend = [Line2D([0], [0], color=m.FUN_C, lw=2, label="FuN"),
              Patch(facecolor=m.BAND, label="opponent B (post-shift)")]
    fig.legend(handles=legend, ncol=2, fontsize=16, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("TEST: FuN 13x13, aligned to each seed's own first shift")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = os.path.join(OUT, "fig_phase2_curves_13x13_TEST_aligned.png")
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
