# ══════════════════════════════════════════════════════════════
# Build a W&B Report of RAW training curves (Phase 1 + Phase 2) so the
# panels can be downloaded (PNG/SVG) straight from W&B for the appendix.
#
#   python paperMaterials/make_wandb_report.py
# prints the report URL. Open it, switch W&B to light mode, then download
# each panel via its  ⋯ → Download → SVG/PNG.
# ══════════════════════════════════════════════════════════════

import wandb_workspaces.reports.v2 as wr

E, P = "bradrscott4-university-of-cape-town", "honours-rl-go"
BOARDS = ["9x9", "13x13"]
MAGS = ["low", "med", "high"]
FREQS = ["f1", "f2", "f3"]
NICE_B = {"9x9": "9x9", "13x13": "13x13"}
NICE_M = {"low": "Low", "med": "Med", "high": "High"}


def f(*conds):
    return " and ".join(conds)


def C(k, v):
    return f"Config('{k}') = '{v}'"


def runset(name, *conds):
    return wr.Runset(entity=E, project=P, name=name, filters=f(*conds))


def by_opponent(title, ykey):
    """Per-run raw curves for a single agent, coloured by opponent."""
    return wr.LinePlot(title=title, x="Step", y=[ykey],
                       legend_fields=["meta_opponent"],
                       layout=wr.Layout(w=8, h=6))


# every metric logged, per agent (title, key)
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
    ("Early stopped", "train/early_stopped"),
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


def agent_grouped(title, ykey):
    """PPO vs FuN: mean ± stderr grouped by agent (valid within one condition)."""
    return wr.LinePlot(title=title, x="Step", y=[ykey],
                       groupby="meta_agent", groupby_aggfunc="mean",
                       groupby_rangefunc="stderr",
                       layout=wr.Layout(w=8, h=6))


blocks = [
    wr.H1("Raw Training Curves (Weights & Biases)"),
    wr.P("Raw per-run and per-condition curves exported directly from W&B, "
         "for the appendix. Switch to light mode before downloading each panel."),

    wr.H1("Phase 1 - Capability"),
]

# Phase 1: PPO and Feudal kept in SEPARATE sections (each panel = one agent,
# coloured by opponent). Aggressive opponent excluded (not one of the paper's five).
NOT_AGGR = "Config('meta_opponent') != 'aggressive'"
for agent, nice, metrics in (("ppo", "PPO", PPO_METRICS),
                             ("feudal", "Feudal (FuN)", FUN_METRICS)):
    blocks.append(wr.H2(f"Phase 1 · {nice}"))
    for b in BOARDS:
        rs = runset(f"{nice} · {NICE_B[b]}", C("meta_phase", "phase1"),
                    C("meta_agent", agent), C("meta_board", b), NOT_AGGR)
        blocks += [
            wr.H3(f"{nice} · {NICE_B[b]}"),
            wr.PanelGrid(runsets=[rs],
                         panels=[by_opponent(t, k) for t, k in metrics]),
        ]

# Phase 2: per condition (board × magnitude × frequency), PPO vs FuN mean±stderr
blocks += [wr.H1("Phase 2 - Recovery")]
for b in BOARDS:
    blocks += [wr.H2(f"Phase 2 · {NICE_B[b]}")]
    for m in MAGS:
        grids = []
        panels = []
        for fq in FREQS:
            rs = runset(f"{NICE_B[b]} · {NICE_M[m]} · {fq}",
                        C("meta_phase", "phase2"), C("meta_board", b),
                        C("meta_magnitude", m), C("meta_frequency", fq))
            # each condition needs its own runset → its own PanelGrid
            grids.append(wr.PanelGrid(runsets=[rs], panels=[
                agent_grouped(f"{NICE_M[m]} · {fq} · win rate", "custom/win_rate")
            ]))
        blocks += grids

# Update the existing report in place if EXISTING_URL is set, else create new.
EXISTING_URL = ("https://wandb.ai/bradrscott4-university-of-cape-town/"
                "honours-rl-go/reports/Appendix---Raw-Training-Curves-"
                "(Phase-1-&-2)--VmlldzoxNzg0Mjc5MQ==")
if EXISTING_URL:
    report = wr.Report.from_url(EXISTING_URL)
    report.blocks = blocks
    report.width = "fluid"
else:
    report = wr.Report(entity=E, project=P,
                       title="Appendix - Raw Training Curves (Phase 1 & 2)",
                       width="fluid", blocks=blocks)
report.save()
print("REPORT URL:", report.url)
