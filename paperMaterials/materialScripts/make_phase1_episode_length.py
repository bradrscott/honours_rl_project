# ══════════════════════════════════════════════════════════════
# Section 4.1 — average game length (env. steps) per agent x opponent x
# board, PPO vs FuN. Replaces fig_phase1_bars.png in main.tex: that figure
# just re-plotted the win rates already in tab:phase1, whereas this uses a
# different logged quantity (rollout/ep_len_mean) entirely, so it shows a
# different property (how long games run) rather than the same numbers
# again (how often they're won).
#
# Same single-column layout/sizing as the figure it replaces (two panels,
# one per board, large fonts for a half-page slot), new colour pair.
#
#   python paperMaterials/make_phase1_episode_length.py
#
# Writes into paperMaterials/results/:
#   tab_phase1_eplen.tex          avg. game length (env. steps), same shape
#                                 as tab_phase1.tex (for reference/appendix)
#   fig_phase1_eplen_bars.png     grouped bars per board, PPO vs FuN
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
# Grayscale-native pair: charcoal (solid) vs light grey (hatched) so the two
# series are distinguished by shade AND pattern, readable in grayscale.
PPO_C, FUN_C = "#3a3a3a", "#c9c9c9"   # charcoal / light grey


def canonical_runs():
    """Most-recent Phase-1 run per (board, agent, opponent), keyed on
    rollout/ep_len_mean (same run selection rule as make_phase1_results.py)."""
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
        el = r.summary.get("rollout/ep_len_mean")
        if el is None:
            continue
        if key not in best or r.created_at > best[key][0]:
            best[key] = (r.created_at, el, r)
    return best


def write_table(best):
    def cell(board, agent, o):
        v = best.get((board, agent, o))
        return f"{v[1]:.0f}" if v else "---"

    def pair(board, o):  # bold the SHORTER game (more decisive)
        p = best.get((board, "ppo", o)); f = best.get((board, "feudal", o))
        ps, fs = cell(board, "ppo", o), cell(board, "feudal", o)
        if p and f:
            if p[1] < f[1]:
                ps = r"\textbf{" + ps + "}"
            elif f[1] < p[1]:
                fs = r"\textbf{" + fs + "}"
        return ps, fs

    lines = [
        r"% ── Phase-1 average game length (environment steps) per agent x opponent x board ──",
        r"% requires \usepackage{booktabs}. Bold marks the shorter (more decisive) game.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  % Add your own \caption{...} here.",
        r"  \label{tab:phase1-eplen}",
        r"  \small",
        r"  \begin{tabular}{@{}l cc cc@{}}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{$9\times9$} & \multicolumn{2}{c}{$13\times13$} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"    Opponent & PPO & FuN & PPO & FuN \\",
        r"    \midrule",
    ]
    for o in OPPS:
        p9, f9 = pair("9x9", o); p13, f13 = pair("13x13", o)
        lines.append(f"    {NICE[o]} & {p9} & {f9} & {p13} & {f13} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    path = os.path.join(OUT, "tab_phase1_eplen.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


def fig_bars(best):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=False)
    x = np.arange(len(OPPS)); w = 0.40
    for ax, board in zip(axes, BOARDS):
        ppo = [best.get((board, "ppo", o), (0, np.nan, None))[1] for o in OPPS]
        fun = [best.get((board, "feudal", o), (0, np.nan, None))[1] for o in OPPS]
        b1 = ax.bar(x - w / 2, ppo, w, label="PPO", color=PPO_C,
                    edgecolor="white", linewidth=0.8, zorder=3)
        b2 = ax.bar(x + w / 2, fun, w, label="FuN", color=FUN_C,
                    edgecolor="#3a3a3a", linewidth=0.8, hatch="////", zorder=3)
        top = np.nanmax(ppo + fun)
        for bars, vals in ((b1, ppo), (b2, fun)):
            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.02,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=15, color="#4a4a4a", zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[o] for o in OPPS], fontsize=17)
        ax.set_ylim(0, top * 1.18)
        title = "$" + board.replace("x", r"\times") + "$"
        ax.set_title(title, fontsize=22, fontweight="bold", pad=10)
        ax.yaxis.grid(True, color="#e9ecef", lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=16)
    axes[0].set_ylabel("avg. game length\n(env. steps)", fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=17, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.03),
               handlelength=1.5, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "fig_phase1_eplen_bars.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    best = canonical_runs()
    print(f"resolved {len(best)} canonical (board,agent,opponent) runs")
    write_table(best)
    fig_bars(best)
    print("done.")
