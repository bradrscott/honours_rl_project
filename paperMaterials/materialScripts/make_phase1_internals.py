# ══════════════════════════════════════════════════════════════
# Section 4.1 appendix — ALL Phase-1 training internals, one figure per
# (board, agent): every logged metric as a subplot, curves coloured by
# opponent (the paper's five; aggressive excluded). Smoothed line over a
# faint raw trace (W&B-style).
#
#   python paperMaterials/make_phase1_internals.py
#
# Writes into paperMaterials/results/:
#   fig_phase1_internals_ppo_9x9.png
#   fig_phase1_internals_feudal_9x9.png
#   fig_phase1_internals_ppo_13x13.png
#   fig_phase1_internals_feudal_13x13.png
# ══════════════════════════════════════════════════════════════

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
ENTITY, PROJECT = "bradrscott4-university-of-cape-town", "honours-rl-go"

OPPS = ["greedy", "defensive", "corner", "edge", "random"]
NICE_O = {o: o.capitalize() for o in OPPS}
# distinct, print-friendly colour per opponent
OPP_C = {"greedy": "#2f6f8f", "defensive": "#b9754a", "corner": "#5f7d6a",
         "edge": "#8e5a9e", "random": "#c0504d"}
SMOOTH = 0.9

PPO_METRICS = [
    ("Win rate", "custom/win_rate"),
    ("Episode reward (mean)", "rollout/ep_rew_mean"),
    ("Episode length (mean)", "rollout/ep_len_mean"),
    ("Total episodes", "custom/total_episodes"),
    ("Policy loss", "train/policy_loss"),
    ("Value loss", "train/value_loss"),
    ("Entropy", "train/entropy"),
    ("Approx. KL", "train/approx_kl"),
    ("Clip fraction", "train/clip_fraction"),
    ("Explained variance", "train/explained_var"),
]
FUN_METRICS = [
    ("Win rate", "custom/win_rate"),
    ("Episode reward (mean)", "rollout/ep_rew_mean"),
    ("Episode length (mean)", "rollout/ep_len_mean"),
    ("Total episodes", "custom/total_episodes"),
    ("Total loss", "loss/total"),
    ("Manager loss", "loss/manager"),
    ("Worker loss", "loss/worker"),
    ("Manager value loss", "loss/value_manager"),
    ("Worker value loss", "loss/value_worker"),
    ("Manager advantage", "manager/advantage"),
    ("Manager cosine", "manager/cosines"),
    ("Worker advantage", "worker/advantage"),
    ("Worker entropy", "worker/entropy"),
    ("Worker intrinsic reward", "worker/intrinsic_reward"),
]
METRICS = {"ppo": PPO_METRICS, "feudal": FUN_METRICS}


def ema(y, w):
    y = np.asarray(y, float)
    out = np.empty_like(y); last = deb = 0.0
    for i, v in enumerate(y):
        last = last * w + (1 - w) * v
        deb = deb * w + (1 - w)
        out[i] = last / deb if deb else v
    return out


def canonical(api, board, agent):
    """most-recent Phase-1 run per opponent (the 5 paper opponents)."""
    best = {}
    for r in api.runs(f"{ENTITY}/{PROJECT}"):
        c = r.config
        if (c.get("meta_phase") == "phase1" and c.get("meta_board") == board
                and c.get("meta_agent") == agent):
            o = c.get("meta_opponent")
            if o in OPPS and (o not in best or r.created_at > best[o][0]):
                best[o] = (r.created_at, r)
    return {o: v[1] for o, v in best.items()}


def fig_for(api, board, agent):
    metrics = METRICS[agent]
    runs = canonical(api, board, agent)
    keys = [k for _, k in metrics]
    # fetch history once per run
    hist = {}
    for o, run in runs.items():
        h = run.history(keys=keys, samples=1500)
        hist[o] = h

    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.9 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for ax in axes[len(metrics):]:
        ax.axis("off")

    for ax, (title, key) in zip(axes, metrics):
        for o in OPPS:
            h = hist.get(o)
            if h is None or key not in h.columns:
                continue
            hh = h.dropna(subset=[key]).sort_values("_step")
            if hh.empty:
                continue
            x = hh["_step"].to_numpy() / 1e6
            y = hh[key].to_numpy()
            ax.plot(x, y, color=OPP_C[o], lw=0.7, alpha=0.15)
            ax.plot(x, ema(y, SMOOTH), color=OPP_C[o], lw=1.4)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("steps (M)", fontsize=8.5)
        ax.grid(True, color="#eef0f2", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        if key == "custom/win_rate":
            ax.set_ylim(0, 1.02)

    handles = [Line2D([0], [0], color=OPP_C[o], lw=2, label=NICE_O[o])
               for o in OPPS]
    fig.legend(handles=handles, ncol=5, fontsize=10.5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.005))
    board_t = board.replace("x", r"$\times$")
    agent_t = "PPO" if agent == "ppo" else "Feudal (FuN)"
    fig.suptitle(f"{agent_t} — Phase-1 training internals ({board_t})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = os.path.join(OUT, f"fig_phase1_internals_{agent}_{board}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    api = wandb.Api()
    for board in ["9x9", "13x13"]:
        for agent in ["ppo", "feudal"]:
            fig_for(api, board, agent)
    print("done.")
