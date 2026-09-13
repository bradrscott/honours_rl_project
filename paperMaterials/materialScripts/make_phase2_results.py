# ══════════════════════════════════════════════════════════════
# Section 4.2 — Phase-2 Recovery: table + figures (paper-ready).
#
# Reads the validated analysis output in results/phase2/ and the raw
# per-game logs in models/*/<board>/<run>/games.csv, and writes into
# paperMaterials/results/:
#
#   tab_phase2.tex                    recovery / dip / adapt-cost per
#                                     agent x magnitude x board (mean +- SE,
#                                     over 3 seeds x 3 frequencies).
#   fig_phase2_recovery_bars.png      grouped recovery time by magnitude,
#                                     per board, PPO vs FuN (SE bars).
#   fig_phase2_dip_adapt.png          dip depth (top) + adaptation cost
#                                     (bottom) across conditions, per board.
#   fig_phase2_curves_9x9.png         rolling win rate, PPO vs FuN overlaid,
#   fig_phase2_curves_13x13.png       one panel per condition (mag x freq),
#                                     shift points marked, opponent-B shaded.
#
#   python paperMaterials/make_phase2_results.py
# ══════════════════════════════════════════════════════════════

import csv
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# __file__ now lives at <repo>/paperMaterials/materialScripts/, so the repo
# root is three levels up (not two — this moved when the script was filed
# into materialScripts/ during the repo reorganisation).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "phase2")
OUT = os.path.join(ROOT, "paperMaterials", "resultsSection")
os.makedirs(OUT, exist_ok=True)

BOARDS = ["9x9", "13x13"]
MAGS = ["low", "med", "high"]
FREQS = ["f1", "f2", "f3"]
MAG_NICE = {"low": "Low", "med": "Med", "high": "High"}
BOARD_NICE = {"9x9": r"$9\times9$", "13x13": r"$13\times13$"}
AGENT_ROOT = {"ppo": "ppo_go", "feudal": "feudal"}
# magnitude -> (opponent A = trained baseline, opponent B = post-shift / unseen)
AB = {"low": ("corner", "edge"),
      "med": ("corner", "defensive"),
      "high": ("greedy", "defensive")}
# Phase-2 palette — slate blue / coral (distinct from the 4.1 blue/clay figures).
PPO_C, FUN_C = "#6a1b9a", "#2ec4d6"       # deep purple (dark) / bright cyan (light) — matches Phase-1 curves, grayscale-safe
PPO_E, FUN_E = "#35506b", "#b3583b"       # darker error-bar shades
BAND = "#ecebf3"                          # neutral lavender-grey shift shading
GRID = "#e9ecef"
INK = "#2b2b2b"


# ── read the validated per-condition summary ─────────────────────
def load_summary():
    rows = {}
    with open(os.path.join(RES, "summary_by_condition.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["board"], r["agent"], r["magnitude"], r["frequency"])
            rows[key] = r
    return rows


def agg_disruption(summary):
    """Pooled disruption rate per (board, agent, mag): sum(n_disrupted) /
    sum(n_shifts) across the 3 frequencies (a true pooled rate, not a mean
    of three per-frequency rates)."""
    out = {}
    for board in BOARDS:
        for agent in ("ppo", "feudal"):
            for mag in MAGS:
                nd, ns = 0, 0
                for freq in FREQS:
                    r = summary.get((board, agent, mag, freq))
                    if not r:
                        continue
                    nd += int(r["n_disrupted"]); ns += int(r["n_shifts"])
                out[(board, agent, mag)] = nd / ns if ns else 0.0
    return out


def agg_by_mag(summary):
    """Collapse the 3 frequencies -> one estimate per (board, agent, mag).
    Mean of the per-frequency seed-means; SE pooled across the 3 independent
    frequency means (se = sqrt(sum se_i^2)/3)."""
    out = {}
    for board in BOARDS:
        for agent in ("ppo", "feudal"):
            for mag in MAGS:
                rec, recse, dip, dipse, adp, adpse = [], [], [], [], [], []
                for freq in FREQS:
                    r = summary.get((board, agent, mag, freq))
                    if not r:
                        continue
                    rec.append(float(r["recovery_mean"]))
                    recse.append(float(r["recovery_se"]))
                    dip.append(float(r["dip_mean"]))
                    dipse.append(float(r["dip_se"]))
                    adp.append(float(r["adapt_cost_mean"]))
                    adpse.append(float(r["adapt_cost_se"]))
                n = len(rec)
                pool = lambda ses: (sum(s * s for s in ses) ** 0.5) / n if n else 0.0
                out[(board, agent, mag)] = dict(
                    recovery=np.mean(rec), recovery_se=pool(recse),
                    dip=np.mean(dip), dip_se=pool(dipse),
                    adapt=np.mean(adp), adapt_se=pool(adpse))
    return out


# ── (1) LaTeX table ──────────────────────────────────────────────
def write_table(agg):
    L = [
        r"% ── Phase-2 recovery: recovery time / dip depth / adaptation cost ──",
        r"% mean +- SE over 3 seeds x 3 shift frequencies. requires booktabs.",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  % Add your own \caption{...} here.",
        r"  \label{tab:phase2}",
        r"  \small",
        r"  \begin{tabular}{@{}ll ccc ccc@{}}",
        r"    \toprule",
        r"    & & \multicolumn{3}{c}{PPO} & \multicolumn{3}{c}{FuN} \\",
        r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"    Board & Mag. & Recovery & Dip & Adapt.\ cost "
        r"& Recovery & Dip & Adapt.\ cost \\",
        r"    & & (games) & (win rate) & (k steps) "
        r"& (games) & (win rate) & (k steps) \\",
        r"    \midrule",
    ]
    for board in BOARDS:
        for i, mag in enumerate(MAGS):
            p = agg[(board, "ppo", mag)]; f = agg[(board, "feudal", mag)]
            bcol = BOARD_NICE[board] if i == 0 else ""
            L.append(
                f"    {bcol} & {MAG_NICE[mag]} "
                f"& {p['recovery']:.0f}\\,$\\pm$\\,{p['recovery_se']:.0f} "
                f"& {p['dip']:.2f} & {p['adapt']:.1f} "
                f"& {f['recovery']:.0f}\\,$\\pm$\\,{f['recovery_se']:.0f} "
                f"& {f['dip']:.2f} & {f['adapt']:.1f} \\\\")
        if board != BOARDS[-1]:
            L.append(r"    \midrule")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    path = os.path.join(OUT, "tab_phase2.tex")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path)


# ── (2) recovery-time bars ───────────────────────────────────────
# NOTE: single-column target size (rendered at \linewidth in a one-column
# figure) — figsize kept small and fonts enlarged so labels stay legible
# after the shrink, matching fig_phase1_bars.png. No error bars (dropped
# per author preference; SE is still reported in tab_phase2.tex).
def _grouped_bars(ax, board, agg, metric, ppo_c=PPO_C, fun_c=FUN_C,
                  edge_c="white", edge_lw=0.8):
    x = np.arange(len(MAGS)); w = 0.38
    ppo = [agg[(board, "ppo", m)][metric] for m in MAGS]
    fun = [agg[(board, "feudal", m)][metric] for m in MAGS]
    b1 = ax.bar(x - w / 2, ppo, w, label="PPO", color=ppo_c,
                edgecolor=edge_c, linewidth=edge_lw, zorder=3)
    b2 = ax.bar(x + w / 2, fun, w, label="FuN", color=fun_c,
                edgecolor=edge_c, linewidth=edge_lw, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([MAG_NICE[m] for m in MAGS], fontsize=15)
    ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=14)
    return b1, b2, ppo, fun


def fig_recovery_bars(agg):
    # this figure only: burgundy (dark) / blush (light) with dark outlines —
    # grayscale-safe and distinct from the purple/cyan curve palette
    REC_PPO, REC_FUN = "#6a1030", "#f4c2c2"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), sharey=False)
    for ax, board in zip(axes, BOARDS):
        b1, b2, ppo, fun = _grouped_bars(ax, board, agg, "recovery",
                                         ppo_c=REC_PPO, fun_c=REC_FUN,
                                         edge_c="#3d0a1c", edge_lw=1.3)
        top = max(max(ppo), max(fun))
        for bars, vals in ((b1, ppo), (b2, fun)):
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.02,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=13, color="#4a4a4a", zorder=4)
        ax.set_ylim(0, top * 1.20)
        ax.set_title(BOARD_NICE[board], fontsize=19, fontweight="bold", pad=10)
        ax.set_xlabel("shift magnitude", fontsize=15)
    axes[0].set_ylabel("recovery time\n(games)", fontsize=16)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, fontsize=16, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), columnspacing=1.8, handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = os.path.join(OUT, "fig_phase2_recovery_bars.png")
    fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


# ── (2b) disruption rate — heatmap, not a bar/line chart ─────────
# NOTE: figsize's aspect ratio (~1.57:1) matches the LaTeX target box
# (width=\linewidth, height=5.4cm on a single-column ACM figure, ~8.46cm /
# 5.4cm), so forcing both dimensions there does not stretch/distort the
# image. Font sizes are trimmed to match the other Phase-2 figures
# (fig_dip_adapt, fig_recovery_bars) once both are scaled to \linewidth.
def fig_disruption_heatmap(disr):
    rows = [(b, m) for b in BOARDS for m in MAGS]   # 6 rows: board x magnitude
    cols = ["ppo", "feudal"]
    grid = np.array([[disr[(b, a, m)] * 100 for a in cols] for (b, m) in rows])

    fig, ax = plt.subplots(figsize=(6.5, 4.15))
    cmap = plt.get_cmap("YlOrBr")
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["PPO", "FuN"], fontsize=22)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{BOARD_NICE[b]}\n{MAG_NICE[m]}" for b, m in rows],
                        fontsize=18)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", length=0)

    for i, (b, m) in enumerate(rows):
        for j in range(2):
            v = grid[i, j]
            txt_c = "white" if v > 55 else "#2b2b2b"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=22, color=txt_c, fontweight="bold")
        if m == "high" and b != BOARDS[-1]:
            ax.axhline(i + 0.5, color="white", lw=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.12)
    cbar.set_label("shifts disrupted (%)", fontsize=17)
    cbar.ax.tick_params(labelsize=15, length=0)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_phase2_disruption_heatmap.png")
    fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


# ── (3) dip depth + adaptation cost ──────────────────────────────
def fig_dip_adapt(agg):
    # turquoise / deep navy, matching fig_phase2_recovery_bars
    DA_PPO, DA_FUN = "#3bbecd", "#1f3a5f"
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4), sharex=True,
                              gridspec_kw=dict(hspace=0.38))
    specs = [("dip", "dip depth\n(win-rate drop)"),
             ("adapt", "adapt.\ncost (k steps)")]
    for row, (metric, ylab) in enumerate(specs):
        for col, board in enumerate(BOARDS):
            ax = axes[row, col]
            b1, b2, ppo, fun = _grouped_bars(ax, board, agg, metric,
                                             ppo_c=DA_PPO, fun_c=DA_FUN)
            top = max(max(ppo), max(fun))
            for bars, vals in ((b1, ppo), (b2, fun)):
                for bar, v in zip(bars, vals):
                    fmt = f"{v:.2f}" if metric == "dip" else f"{v:.0f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.02,
                            fmt, ha="center", va="bottom", fontsize=10.5,
                            color="#4a4a4a", zorder=4)
            ax.set_ylim(0, top * 1.20)
            if row == 0:
                ax.set_title(BOARD_NICE[board], fontsize=17,
                             fontweight="bold", pad=8)
            if row == 1:
                ax.set_xlabel("shift magnitude", fontsize=13)
            if col == 0:
                ax.set_ylabel(ylab, fontsize=13)
            ax.tick_params(axis="both", length=0, labelsize=12)
            ax.set_xticklabels([MAG_NICE[m] for m in MAGS], fontsize=12)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, fontsize=14, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.03), columnspacing=1.8, handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(OUT, "fig_phase2_dip_adapt.png")
    fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


# ── (4) overlaid rolling curves with shift markers ───────────────
def _condition_files(board, agent, mag, freq):
    a = AB[mag][0]
    base = os.path.join(ROOT, "results", "phase2", AGENT_ROOT[agent], board)
    pats = [f"{a}-phase2-{mag}-{freq}", f"{a}-phase2-{mag}-{freq}-s*"]
    files = []
    for pat in pats:
        for d in glob.glob(os.path.join(base, pat)):
            g = os.path.join(d, "games.csv")
            if os.path.isfile(g) and os.path.basename(d) == os.path.basename(d).split("-s")[0] or "-s" in os.path.basename(d):
                if os.path.isfile(g):
                    files.append(g)
    # de-dup and keep exact condition (avoid f1 matching f1x etc.)
    keep = []
    for g in set(files):
        name = os.path.basename(os.path.dirname(g))
        tail = name.split("phase2-", 1)[1]      # e.g. low-f1 or low-f1-s2
        toks = tail.split("-")
        if toks[0] == mag and toks[1] == freq:
            keep.append(g)
    return sorted(keep)


def _load(g):
    step, roll, opp = [], [], []
    with open(g) as f:
        for r in csv.DictReader(f):
            step.append(int(r["global_step"]))
            roll.append(float(r["rolling"]))
            opp.append(r["opponent"])
    return np.array(step), np.array(roll), np.array(opp)


def _mean_curve(files):
    """Interpolate each seed's rolling(step) onto a common grid and average."""
    if not files:
        return None, None
    loaded = [_load(g) for g in files]
    gmax = min(s[-1] for s, _, _ in loaded)
    grid = np.linspace(0, gmax, 700)
    stack = [np.interp(grid, s, r) for s, r, _ in loaded]
    return grid, np.mean(stack, axis=0)


def _b_intervals(board, mag, freq):
    """Opponent-B step intervals (from a reference PPO seed-0 log)."""
    files = _condition_files(board, "ppo", mag, freq)
    if not files:
        return []
    s, _, opp = _load(files[0])
    B = AB[mag][1]
    ivs, start = [], None
    for i in range(len(s)):
        inB = (opp[i] == B)
        if inB and start is None:
            start = s[i]
        elif not inB and start is not None:
            ivs.append((start, s[i])); start = None
    if start is not None:
        ivs.append((start, s[-1]))
    return ivs


def fig_curves(board):
    fig, axes = plt.subplots(len(MAGS), len(FREQS), figsize=(15, 5.6),
                             sharex=True, sharey=True)
    scale = 1e6
    for r, mag in enumerate(MAGS):
        for c, freq in enumerate(FREQS):
            ax = axes[r, c]
            for iv in _b_intervals(board, mag, freq):
                ax.axvspan(iv[0] / scale, iv[1] / scale, color=BAND,
                           zorder=0, lw=0)
            for agent, col in (("ppo", PPO_C), ("feudal", FUN_C)):
                grid, mean = _mean_curve(_condition_files(board, agent, mag, freq))
                if grid is None:
                    continue
                ax.plot(grid / scale, mean, color=col, lw=1.5, zorder=3)
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, None)
            ax.grid(True, color=GRID, lw=0.7, zorder=1)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=12)
            if r == 0:
                ax.set_title(f"{freq}", fontsize=17)
            if c == 0:
                ax.set_ylabel(MAG_NICE[mag], fontsize=16)
                A, B = AB[mag]
                ax.text(0.03, 0.06, f"{A}→{B}", transform=ax.transAxes,
                        fontsize=10.5, color="#6f6f6b", ha="left", va="bottom")
            if r == len(MAGS) - 1:
                ax.set_xlabel("steps (M)", fontsize=14)
    legend = [Line2D([0], [0], color=PPO_C, lw=2, label="PPO"),
              Line2D([0], [0], color=FUN_C, lw=2, label="FuN"),
              Patch(facecolor=BAND, label="post-shift opponent")]
    fig.legend(handles=legend, ncol=3, fontsize=16, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.supylabel("win rate", fontsize=14, x=-0.005)
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(OUT, f"fig_phase2_curves_{board}.png")
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    summary = load_summary()
    agg = agg_by_mag(summary)
    write_table(agg)
    fig_recovery_bars(agg)
    fig_disruption_heatmap(agg_disruption(summary))
    fig_dip_adapt(agg)
    for b in BOARDS:
        fig_curves(b)
    print("done.")
