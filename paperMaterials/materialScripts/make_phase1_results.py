# ══════════════════════════════════════════════════════════════
# Section 4.1 (Phase-1 Capability) — table + figures, from W&B.
#
# Writes into paperMaterials/results/:
#   tab_phase1.tex             win rate per agent x opponent x board (+ GNU Go)
#   fig_phase1_curves.png      rolling win-rate curves, PPO vs FuN, faceted by board
#   fig_phase1_bars.png        grouped bars per board, PPO vs FuN, GNU Go ref line
#
# Pulls the canonical Phase-1 run per (board, agent, opponent) — the most recent
# (the fixed-code runs that produced the Phase-2 resume checkpoints).
#
#   python paperMaterials/make_phase1_results.py
# ══════════════════════════════════════════════════════════════

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
ENTITY, PROJECT = "bradrscott4-university-of-cape-town", "honours-rl-go"

OPPS = ["greedy", "defensive", "corner", "edge", "random"]
NICE = {o: o.capitalize() for o in OPPS}
BOARDS = ["9x9", "13x13"]
PPO_C, FUN_C, GNU_C = "#2f6f8f", "#b9754a", "#5f7d6a"   # muted blue / clay / green

# GNU Go reference (level 10, 100 games) — from tab_gnugo
GNUGO = {"9x9": {o: 1.00 for o in OPPS},
         "13x13": {"greedy": 1.00, "defensive": 1.00, "corner": 0.99,
                   "edge": 1.00, "random": 1.00}}


def canonical_runs():
    """Most-recent Phase-1 run per (board, agent, opponent) for the 5 opponents."""
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
        wr = r.summary.get("custom/win_rate")
        if wr is None:
            continue
        if key not in best or r.created_at > best[key][0]:
            best[key] = (r.created_at, wr, r)
    return best


def write_table(best):
    def cell(board, agent, o):
        v = best.get((board, agent, o))
        return f"{v[1] * 100:.1f}" if v else "---"

    def pair(board, o):  # bold the stronger agent
        p = best.get((board, "ppo", o)); f = best.get((board, "feudal", o))
        ps, fs = cell(board, "ppo", o), cell(board, "feudal", o)
        if p and f:
            if p[1] > f[1]:
                ps = r"\textbf{" + ps + "}"
            elif f[1] > p[1]:
                fs = r"\textbf{" + fs + "}"
        return ps, fs

    lines = [
        r"% ── Phase-1 capability: win rate (%) per agent x opponent x board ──",
        r"% requires \usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  % Add your own \caption{...} here.",
        r"  \label{tab:phase1}",
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
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(OUT, "tab_phase1.tex")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


def write_csv(best):
    rows = ["board,opponent,ppo_win_rate,fun_win_rate"]
    for board in BOARDS:
        for o in OPPS:
            p = best.get((board, "ppo", o)); f = best.get((board, "feudal", o))
            pv = f"{p[1]:g}" if p else ""
            fv = f"{f[1]:g}" if f else ""
            rows.append(f"{board},{o},{pv},{fv}")
    path = os.path.join(OUT, "phase1_win_rates.csv")
    open(path, "w").write("\n".join(rows) + "\n")
    print("wrote", path)


def fig_curves(best):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey=True)
    for row, board in enumerate(BOARDS):
        for col, o in enumerate(OPPS):
            ax = axes[row, col]
            for agent, c in (("ppo", PPO_C), ("feudal", FUN_C)):
                v = best.get((board, agent, o))
                if not v:
                    continue
                h = v[2].history(keys=["custom/win_rate"], samples=400)
                h = h.dropna(subset=["custom/win_rate"])
                ax.plot(h["_step"] / 1e6, h["custom/win_rate"], color=c, lw=1.3,
                        label={"ppo": "PPO", "feudal": "FuN"}[agent])
            ax.axhline(GNUGO[board][o], color=GNU_C, lw=1.0, ls=(0, (3, 2)), alpha=0.7)
            ax.set_ylim(0, 1.02)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(NICE[o], fontsize=14)
            if col == 0:
                ax.set_ylabel(f"{board}\nwin rate", fontsize=13)
            ax.tick_params(labelsize=11)
    for col in range(5):
        axes[1, col].set_xlabel("steps (M)", fontsize=12)
    axes[0, 0].legend(fontsize=12, loc="lower right", frameon=False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_phase1_curves.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_bars(best):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    x = np.arange(len(OPPS)); w = 0.40
    for ax, board in zip(axes, BOARDS):
        ppo = [best.get((board, "ppo", o), (0, np.nan))[1] for o in OPPS]
        fun = [best.get((board, "feudal", o), (0, np.nan))[1] for o in OPPS]
        b1 = ax.bar(x - w / 2, ppo, w, label="PPO", color=PPO_C,
                    edgecolor="white", linewidth=0.8, zorder=3)
        b2 = ax.bar(x + w / 2, fun, w, label="FuN", color=FUN_C,
                    edgecolor="white", linewidth=0.8, zorder=3)
        # value labels above each bar
        for bars, vals in ((b1, ppo), (b2, fun)):
            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                        f"{v * 100:.0f}", ha="center", va="bottom",
                        fontsize=15, color="#4a4a4a", zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[o] for o in OPPS], fontsize=17)
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        title = "$" + board.replace("x", r"\times") + "$"
        ax.set_title(title, fontsize=22, fontweight="bold", pad=10)
        ax.yaxis.grid(True, color="#e9ecef", lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=16)
    axes[0].set_ylabel("final win rate", fontsize=18)
    axes[0].set_yticklabels([f"{t:.1f}" for t in np.arange(0, 1.01, 0.2)])
    # single shared legend, above the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=17, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.03),
               handlelength=1.2, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "fig_phase1_bars.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    best = canonical_runs()
    print(f"resolved {len(best)} canonical (board,agent,opponent) runs")
    write_table(best)
    write_csv(best)
    fig_bars(best)
    fig_curves(best)
    print("done.")
