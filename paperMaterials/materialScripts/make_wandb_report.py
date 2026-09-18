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


def by_frequency(title, ykey):
    """Per-run Phase-2 curves for one agent+board+magnitude, coloured by
    shift frequency (f1/f2/f3)."""
    return wr.LinePlot(title=title, x="Step", y=[ykey],
                       legend_fields=["meta_frequency"],
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
    wr.H1("Honours RL Go — All Training Runs (W&B)"),
    wr.P("Complete set of raw training and evaluation curves for the study, "
         "straight from Weights & Biases: the GNU Go external-reference baseline "
         "(10 runs), Phase-1 capability training (20 runs), and Phase-2 recovery "
         "under opponent shifts (108 runs)."),

    wr.H1("GNU Go Baseline (external reference)"),
]

# GNU Go baseline: win rate per opponent, per board (agent = gnugo, 10 runs).
for b in BOARDS:
    rs = runset(f"GNU Go · {NICE_B[b]}", C("meta_phase", "baseline"), C("meta_board", b))
    blocks += [
        wr.H3(f"GNU Go · {NICE_B[b]}"),
        wr.PanelGrid(runsets=[rs], panels=[
            by_opponent("Win rate", "custom/win_rate"),
        ]),
    ]

blocks += [wr.H1("Phase 1 - Capability")]

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

# Phase 2: per agent, per board — ALL logged metrics, runs coloured by shift
# condition (magnitude × frequency). Mirrors the Phase-1 layout, plus the
# Phase-2-specific recovery metrics.
blocks += [wr.H1("Phase 2 - Recovery")]
PHASE2_EXTRA = [
    ("Rolling win rate", "phase2/rolling"),
    ("Recovery games (last shift)", "phase2/recovery_games_last_shift"),
    ("Pre-shift baseline", "phase2/baseline"),
]
for agent, nice, metrics in (("ppo", "PPO", PPO_METRICS + PHASE2_EXTRA),
                             ("feudal", "Feudal (FuN)", FUN_METRICS + PHASE2_EXTRA)):
    blocks.append(wr.H2(f"Phase 2 · {nice}"))
    for b in BOARDS:
        for m in MAGS:
            rs = runset(f"{nice} · {NICE_B[b]} · {NICE_M[m]}",
                        C("meta_phase", "phase2"), C("meta_agent", agent),
                        C("meta_board", b), C("meta_magnitude", m))
            blocks += [
                wr.H3(f"{nice} · {NICE_B[b]} · {NICE_M[m]}"),
                wr.PanelGrid(runsets=[rs],
                             panels=[by_frequency(t, k) for t, k in metrics]),
            ]

# Update the existing report in place if EXISTING_URL is set, else create new.
EXISTING_URL = ("https://wandb.ai/bradrscott4-university-of-cape-town/"
                "honours-rl-go/reports/Appendix---Raw-Training-Curves-"
                "(Phase-1-&-2)--VmlldzoxNzg0Mjc5MQ==")
if EXISTING_URL:
    report = wr.Report.from_url(EXISTING_URL)
    report.blocks = blocks
    report.width = "fluid"
    report.title = "Honours RL Go — All Training Runs (GNU Go, Phase 1 & 2)"
else:
    report = wr.Report(entity=E, project=P,
                       title="Appendix - Raw Training Curves (Phase 1 & 2)",
                       width="fluid", blocks=blocks)
report.save()
print("REPORT URL:", report.url)
