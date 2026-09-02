# ══════════════════════════════════════════════════════════════
# Section 4.1 — "time to stable baseline" settling-step figure.
#
# For each (board, agent, opponent): target = that run's final win rate
# (the exact value in tab:phase1). Settling step = the FIRST training point
# after which the rolling win rate stays within a target*frac band for a
# sustained settling window (HOLD of the total budget) — the standard
# control-theory settling time with a tolerance band + hold window. This is
# robust to isolated late noise dips (which the naive "last dip, then stays
# forever" rule would let pin the settling step at the end of training).
#
# frac = 0.90 (10% band), HOLD = 0.15 (must hold for 15% of the budget).
# A run that never holds the band for a full window is CENSORED ("did not
# settle") and drawn as a hatched bar at the training budget, not a solid one.
#
# Reuses canonical_runs() from make_phase1_curves.py (same run mapping).
#
#   python paperMaterials/make_phase1_settle.py
#
# Writes paperMaterials/results/fig_phase1_settle_bars.png  (replaces the
# old fig_phase1_bars.png, which duplicated the table).
# ══════════════════════════════════════════════════════════════

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the exact same (board, agent, opponent) -> run resolution
from paperMaterials.materialScripts.make_phase1_curves import (canonical_runs, OPPS, NICE, BOARDS,
                                 PPO_C, FUN_C, OUT)

FRAC = 0.90   # tolerance band = 90% of the final win rate (a 10% band)
HOLD = 0.15   # must stay in-band for a window of 15% of the training budget


def settling_step(steps, wr, target, frac=FRAC, hold=HOLD):
    """First step after which wr stays within [target*frac, inf) for a window
    of `hold` of the run length. Returns (step, censored). Censored=True (and
    step = final step) when no such window exists — the run never settles."""
    steps = np.asarray(steps, float)
    wr = np.asarray(wr, float)
    band = target * frac
    n = len(steps)
    W = max(int(n * hold), 1)
    inband = wr >= band
    for i in range(max(n - W, 1)):
        if inband[i:i + W].all():
            return float(steps[i]), False
    return float(steps[-1]), True       # never held the band for a full window


def resolve_settles(best):
    """best: {(board, agent, opp): (created_at, run)}  (from canonical_runs).
    Returns {(board, agent, opp): (settle_M, censored)}."""
    settle = {}
    for (board, agent, opp), (_created, run) in best.items():
        target = run.summary.get("custom/win_rate")
        if target is None:
            continue
        h = run.history(keys=["custom/win_rate"], samples=2000)
        h = h.dropna(subset=["custom/win_rate"]).sort_values("_step")
        if h.empty:
            continue
        s, cens = settling_step(h["_step"].to_numpy(),
                                h["custom/win_rate"].to_numpy(), target)
        settle[(board, agent, opp)] = (s / 1e6, cens)   # millions of steps
    return settle


def _draw_group(ax, x, entries, color, budget, top):
    """entries: list of (value_M, censored) per opponent. Censored bars are
    drawn hatched at the training budget and labelled 'n/s' (did not settle)."""
    bars = []
    for xi, e in zip(x, entries):
        if e is None:
            bars.append(None); continue
        v, cens = e
        if cens:
            b = ax.bar(xi, budget, ax_w, color=color, alpha=0.32,
                       edgecolor=color, linewidth=1.1, hatch="///", zorder=3)
            ax.text(xi, budget + top * 0.02, "n/s", ha="center", va="bottom",
                    fontsize=8, color=color, zorder=4, fontweight="bold")
        else:
            b = ax.bar(xi, v, ax_w, color=color, edgecolor="white",
                       linewidth=0.8, zorder=3)
            ax.text(xi, v + top * 0.02, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color="#4a4a4a", zorder=4)
        bars.append(b)
    return bars


ax_w = 0.40


def fig_settle_bars(settle):
    # two panels, one per board; independent y-axes so each board's spread
    # (9x9 up to ~3M, 13x13 up to ~5M) is legible.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(OPPS))
    for ax, board in zip(axes, BOARDS):
        ppo = [settle.get((board, "ppo", o)) for o in OPPS]
        fun = [settle.get((board, "feudal", o)) for o in OPPS]
        # board training budget = the censored (full-run) value, else max seen
        vals = [e[0] for e in (ppo + fun) if e]
        budget = max(vals) if vals else 1.0
        top = budget
        _draw_group(ax, x - ax_w / 2, ppo, PPO_C, budget, top)
        _draw_group(ax, x + ax_w / 2, fun, FUN_C, budget, top)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[o] for o in OPPS], fontsize=10)
        ax.set_ylim(0, budget * 1.15)
        title = "$" + board.replace("x", r"\times") + "$"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.yaxis.grid(True, color="#e9ecef", lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", length=0)
        ax.set_ylabel("steps to stable baseline (M)", fontsize=10.5)
    # legend: agents + the "did not settle" hatch marker
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=PPO_C, label="PPO"),
               Patch(facecolor=FUN_C, label="FuN"),
               Patch(facecolor="#b0b0b0", alpha=0.32, hatch="///",
                     edgecolor="#808080", label="did not settle (n/s)")]
    fig.legend(handles=handles, ncol=3, fontsize=10.5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.04),
               handlelength=1.4, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(OUT, "fig_phase1_settle_bars.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    best = canonical_runs()
    print(f"resolved {len(best)} canonical runs")
    settle = resolve_settles(best)
    print(f"\nsettling step (M steps), band={FRAC}, hold={HOLD} "
          f"(n/s = did not settle):")
    for board in BOARDS:
        for agent in ("ppo", "feudal"):
            cells = []
            for o in OPPS:
                e = settle.get((board, agent, o))
                cells.append(f"{o}=n/s" if (e and e[1])
                             else (f"{o}={e[0]:.2f}" if e else f"{o}=NA"))
            print(f"  {board:5s} {agent:6s}  " + "  ".join(cells))
    fig_settle_bars(settle)
    print("done.")
