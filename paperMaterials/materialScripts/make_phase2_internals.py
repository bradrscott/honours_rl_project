# ══════════════════════════════════════════════════════════════
# Section 4.2 appendix — ALL Phase-2 training internals, one figure per
# (board, agent). Includes the generic training internals AND the
# Phase-2-specific recovery metrics (phase2/rolling, phase2/baseline,
# phase2/recovery_games_last_shift). Curves are the seed-0 run of each
# condition, coloured by shift MAGNITUDE, line style by FREQUENCY.
#
#   python paperMaterials/make_phase2_internals.py
#
# Writes into paperMaterials/results/:
#   fig_phase2_internals_ppo_9x9.png
#   fig_phase2_internals_feudal_9x9.png
#   fig_phase2_internals_ppo_13x13.png
#   fig_phase2_internals_feudal_13x13.png
# ══════════════════════════════════════════════════════════════

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paperMaterials", "resultsSection")
os.makedirs(OUT, exist_ok=True)
ENTITY, PROJECT = "bradrscott4-university-of-cape-town", "honours-rl-go"

MAGS = ["low", "med", "high"]
FREQS = ["f1", "f2", "f3"]
NICE_M = {"low": "Low", "med": "Med", "high": "High"}
MAG_C = {"low": "#2f6f8f", "med": "#b9754a", "high": "#c0504d"}   # blue/clay/red
FREQ_LS = {"f1": "-", "f2": "--", "f3": ":"}
SMOOTH = 0.9

# Only the meaningful Phase-2 recovery metrics (same set for both agents).
P2 = [
    ("Rolling win rate (W=100)", "phase2/rolling"),
    ("Pre-shift baseline", "phase2/baseline"),
    ("Recovery games (last shift)", "phase2/recovery_games_last_shift"),
]
METRICS = {"ppo": P2, "feudal": P2}


def ema(y, w):
    y = np.asarray(y, float)
    out = np.empty_like(y); last = deb = 0.0
    for i, v in enumerate(y):
        last = last * w + (1 - w) * v
        deb = deb * w + (1 - w)
        out[i] = last / deb if deb else v
    return out


def seed0_runs(api, board, agent):
    """seed-0 Phase-2 run per (magnitude, frequency)."""
    best = {}
    for r in api.runs(f"{ENTITY}/{PROJECT}"):
        c = r.config
        if (c.get("meta_phase") == "phase2" and c.get("meta_board") == board
                and c.get("meta_agent") == agent and c.get("meta_seed") == "0"):
            key = (c.get("meta_magnitude"), c.get("meta_frequency"))
            best[key] = r
    return best


def fig_for(api, board, agent):
    metrics = METRICS[agent]
    runs = seed0_runs(api, board, agent)
    keys = [k for _, k in metrics]
    hist = {}
    for k2, run in runs.items():
        hist[k2] = run.history(keys=keys, samples=1500)

    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.9 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for ax in axes[len(metrics):]:
        ax.axis("off")

    for ax, (title, key) in zip(axes, metrics):
        for mag in MAGS:
            for fq in FREQS:
                h = hist.get((mag, fq))
                if h is None or key not in h.columns:
                    continue
                hh = h.dropna(subset=[key]).sort_values("_step")
                if hh.empty:
                    continue
                x = hh["_step"].to_numpy() / 1e6
                y = hh[key].to_numpy()
                ax.plot(x, ema(y, SMOOTH), color=MAG_C[mag],
                        ls=FREQ_LS[fq], lw=1.2, alpha=0.9)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("steps (M)", fontsize=8.5)
        ax.grid(True, color="#eef0f2", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        if key in ("custom/win_rate", "phase2/rolling", "phase2/baseline"):
            ax.set_ylim(0, 1.02)

    mag_handles = [Line2D([0], [0], color=MAG_C[m], lw=2.4, label=NICE_M[m])
                   for m in MAGS]
    freq_handles = [Line2D([0], [0], color="#555555", lw=1.6, ls=FREQ_LS[f],
                           label=f) for f in FREQS]
    leg1 = fig.legend(handles=mag_handles, ncol=3, fontsize=10,
                      frameon=False, loc="upper center",
                      bbox_to_anchor=(0.38, 1.005), title="magnitude")
    fig.legend(handles=freq_handles, ncol=3, fontsize=10, frameon=False,
               loc="upper center", bbox_to_anchor=(0.68, 1.005),
               title="frequency")
    fig.add_artist(leg1)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = os.path.join(OUT, f"fig_phase2_internals_{agent}_{board}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    api = wandb.Api()
    for board in ["9x9", "13x13"]:
        for agent in ["ppo", "feudal"]:
            fig_for(api, board, agent)
    print("done.")
