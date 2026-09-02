# ══════════════════════════════════════════════════════════════
# PHASE 2 — supervisor-facing results report (self-contained HTML)
#
# Reads phase2_results/summary_by_condition.csv (+ the raw games.csv for the
# two overlay plots) and writes phase2_results/RESULTS_REPORT.html with every
# figure embedded as base64, so the single file opens anywhere with no assets.
#
#   python3 phase2/make_report.py
# ══════════════════════════════════════════════════════════════

import base64
import csv
import io
import os
from collections import deque, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "phase2_results"
PPO_C, FUN_C = "#2563eb", "#e11d48"          # blue = PPO, red = FuN
MAGS = ["low", "med", "high"]
MAG_LABEL = {"low": "LOW\n(corner→edge)", "med": "MED\n(corner→defensive)",
             "high": "HIGH\n(greedy→defensive)"}


def load_summary():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "summary_by_condition.csv"))))
    for r in rows:
        for k in ("n_shifts", "n_disrupted", "n_recovered", "n_censored"):
            r[k] = int(r[k])
        for k in ("mean_recovery_disrupted", "mean_dip", "worst_dip",
                  "mean_adapt_cost_ksteps"):
            r[k] = float(r[k]) if r[k] != "" else None
    return rows


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def agg(rows, board, agent, field, reduce_="mean"):
    """Aggregate a field across frequencies per magnitude for one board+agent."""
    out = {}
    for m in MAGS:
        vals = [r[field] for r in rows
                if r["board"] == board and r["agent"] == agent
                and r["magnitude"] == m and r[field] is not None]
        out[m] = (sum(vals) / len(vals)) if vals else None
    return out


# ---- Figure 1: disruption rate, board x agent -----------------------------
def fig_disruption(rows):
    boards = ["9x9", "13x13"]
    rate = {}
    for b in boards:
        for a in ("ppo", "feudal"):
            nd = sum(r["n_disrupted"] for r in rows if r["board"] == b and r["agent"] == a)
            ns = sum(r["n_shifts"]    for r in rows if r["board"] == b and r["agent"] == a)
            rate[(b, a)] = 100 * nd / ns
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = range(len(boards)); w = 0.36
    ax.bar([i - w/2 for i in x], [rate[(b, "ppo")] for b in boards], w,
           label="PPO (flat)", color=PPO_C)
    ax.bar([i + w/2 for i in x], [rate[(b, "feudal")] for b in boards], w,
           label="FuN (hierarchical)", color=FUN_C)
    for i, b in enumerate(boards):
        ax.text(i - w/2, rate[(b, "ppo")] + 1, f"{rate[(b,'ppo')]:.0f}%", ha="center", fontsize=9)
        ax.text(i + w/2, rate[(b, "feudal")] + 1, f"{rate[(b,'feudal')]:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(boards)
    ax.set_ylabel("shifts disrupted (%)"); ax.set_ylim(0, 100)
    ax.set_title("Disruption rate: how often each agent's win rate\nfell below 80% of its pre-shift baseline")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    return fig_to_b64(fig)


# ---- Figure 2: recovery duration by magnitude (9x9) -----------------------
def fig_recovery(rows, board="9x9"):
    ppo = agg(rows, board, "ppo", "mean_recovery_disrupted")
    fun = agg(rows, board, "feudal", "mean_recovery_disrupted")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = range(len(MAGS)); w = 0.36
    ax.bar([i - w/2 for i in x], [ppo[m] or 0 for m in MAGS], w, label="PPO", color=PPO_C)
    ax.bar([i + w/2 for i in x], [fun[m] or 0 for m in MAGS], w, label="FuN", color=FUN_C)
    for i, m in enumerate(MAGS):
        if ppo[m] is not None: ax.text(i - w/2, ppo[m] + 20, f"{ppo[m]:.0f}", ha="center", fontsize=9)
        if fun[m] is not None: ax.text(i + w/2, fun[m] + 20, f"{fun[m]:.0f}", ha="center", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels([MAG_LABEL[m] for m in MAGS], fontsize=8)
    ax.set_ylabel("recovery time (games below the bar)")
    ax.set_title(f"Recovery duration when disrupted — {board}\n(mean over frequencies; lower = recovers faster)")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    return fig_to_b64(fig)


# ---- Figure 3: adaptation cost by magnitude, both boards (log) ------------
def fig_cost(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, board in zip(axes, ["9x9", "13x13"]):
        ppo = agg(rows, board, "ppo", "mean_adapt_cost_ksteps")
        fun = agg(rows, board, "feudal", "mean_adapt_cost_ksteps")
        x = range(len(MAGS)); w = 0.36
        ax.bar([i - w/2 for i in x], [ppo[m] or 0 for m in MAGS], w, label="PPO", color=PPO_C)
        ax.bar([i + w/2 for i in x], [fun[m] or 0 for m in MAGS], w, label="FuN", color=FUN_C)
        ax.set_yscale("log")
        ax.set_xticks(list(x)); ax.set_xticklabels([m.upper() for m in MAGS])
        ax.set_title(board); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("adaptation cost\n(win-rate · ksteps, log)")
    axes[0].legend()
    fig.suptitle("Adaptation cost — area under the win-rate dip (lower = less disrupted)", y=1.02)
    return fig_to_b64(fig)


# ---- Figure 4: rolling win-rate overlay for one condition -----------------
def rolling(wins, W=100):
    dq, o = deque(maxlen=W), []
    for w in wins:
        dq.append(w); o.append(sum(dq) / len(dq))
    return o


def fig_overlay(board, cond, k=0, maxg=4000):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for agent, root, c in [("PPO", "ppo_go", PPO_C), ("FuN", "feudal", FUN_C)]:
        p = f"models/{root}/{board}/{cond}/games.csv"
        if not os.path.exists(p):
            continue
        rr = list(csv.DictReader(open(p)))
        rv = rolling([int(r["win"]) for r in rr])
        seg = [(int(r["games_since_shift"]), rv[i]) for i, r in enumerate(rr)
               if int(r["shift_idx"]) == k]
        seg = [(g, v) for g, v in seg if g <= maxg]
        if seg:
            ax.plot([g for g, _ in seg], [v for _, v in seg], lw=1.4, color=c, label=agent)
    ax.axvline(0, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("completed games since the shift")
    ax.set_ylabel("rolling win rate (W=100)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Win rate after a single HIGH-magnitude shift (greedy→defensive) — {board}")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    return fig_to_b64(fig)


# ---- HTML table from summary rows -----------------------------------------
def cell(v, fmt="{:.0f}", dash="—"):
    return dash if v is None else fmt.format(v)


def summary_table(rows):
    order = {"low": 0, "med": 1, "high": 2}
    rows = sorted(rows, key=lambda r: (r["board"], r["agent"], order[r["magnitude"]], r["frequency"]))
    head = ("<tr><th>Board</th><th>Agent</th><th>Condition</th><th>Disrupted</th>"
            "<th>Recovered</th><th>Recovery (games)</th><th>Worst dip</th>"
            "<th>Adapt. cost (k)</th></tr>")
    body = []
    for r in rows:
        disr = f"{r['n_disrupted']}/{r['n_shifts']}"
        rec  = f"{r['n_recovered']}/{r['n_disrupted']}" if r["n_disrupted"] else "—"
        agent = "FuN" if r["agent"] == "feudal" else "PPO"
        cls = ' class="fun"' if r["agent"] == "feudal" else ""
        body.append(
            f"<tr{cls}><td>{r['board']}</td><td>{agent}</td>"
            f"<td>{r['magnitude']}-{r['frequency']}</td><td>{disr}</td><td>{rec}</td>"
            f"<td>{cell(r['mean_recovery_disrupted'])}</td>"
            f"<td>{cell(r['worst_dip'], '{:.2f}')}</td>"
            f"<td>{cell(r['mean_adapt_cost_ksteps'], '{:.1f}')}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


# ---- Policy reuse: novel-to-B vs return-to-A ------------------------------
def load_reuse():
    p = os.path.join(RESULTS, "policy_reuse.csv")
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ("mean_dip", "mean_recovery", "mean_cost_ksteps"):
            r[k] = float(r[k])
        for k in ("n_shifts", "n_disrupted"):
            r[k] = int(r[k])
    return rows


def fig_reuse(reuse):
    NB, RA = "#94a3b8", "#334155"                 # novel-B (light) / return-A (dark)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)

    def dip(board, agent, tgt):
        for r in reuse:
            if r["board"] == board and r["agent"] == agent and r["shift_type"] == tgt:
                return r["mean_dip"]
        return 0

    for ax, board in zip(axes, ["9x9", "13x13"]):
        agents = ["ppo", "feudal"]; x = range(len(agents)); w = 0.36
        ax.bar([i - w/2 for i in x], [dip(board, a, "novel-B") for a in agents], w,
               label="novel → B", color=NB)
        ax.bar([i + w/2 for i in x], [dip(board, a, "return-A") for a in agents], w,
               label="return → A", color=RA)
        ax.set_xticks(list(x)); ax.set_xticklabels(["PPO", "FuN"])
        ax.set_title(board); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("dip depth (drop below baseline)")
    axes[0].legend()
    fig.suptitle("Policy reuse — dip on return to the trained opponent A vs a novel B\n"
                 "(a big return-A dip = the agent forgot A while adapting to B)", y=1.06)
    return fig_to_b64(fig)


def reuse_table(reuse):
    rows = sorted(reuse, key=lambda r: (r["board"], r["agent"], r["shift_type"]))
    head = ("<tr><th>Board</th><th>Agent</th><th>Shift</th><th>Disrupted</th>"
            "<th>Dip</th><th>Recovery (games)</th><th>Cost (k)</th></tr>")
    body = []
    for r in rows:
        agent = "FuN" if r["agent"] == "feudal" else "PPO"
        cls = ' class="fun"' if r["agent"] == "feudal" else ""
        body.append(
            f"<tr{cls}><td>{r['board']}</td><td>{agent}</td><td>{r['shift_type']}</td>"
            f"<td>{r['n_disrupted']}/{r['n_shifts']}</td>"
            f"<td>{r['mean_dip']:.3f}</td><td>{r['mean_recovery']:.0f}</td>"
            f"<td>{r['mean_cost_ksteps']:.1f}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def main():
    rows = load_summary()
    f1 = fig_disruption(rows)
    f2 = fig_recovery(rows, "9x9")
    f3 = fig_cost(rows)
    ov9  = fig_overlay("9x9",  "greedy-phase2-high-f1")
    ov13 = fig_overlay("13x13", "greedy-phase2-high-f1")
    reuse = load_reuse()
    reuse_html = ""
    if reuse:
        f_reuse = fig_reuse(reuse)
        reuse_html = f"""
<h2>6. Policy reuse — does the hierarchy retain its old skills?</h2>
<img src="data:image/png;base64,{f_reuse}">
<p class="cap">Every schedule ends by returning to A (the trained opponent). A large
dip on that return means the agent overwrote its A-policy while adapting to B.</p>
{reuse_table(reuse)}
<p>Both agents forget somewhat — even PPO dips when A returns, despite barely reacting
to a <i>novel</i> B. But <b>FuN forgets far more</b>: on 9×9, returning to A costs it a
0.34 dip and hundreds of games, versus PPO's ~0.15 and ~20 games. The hierarchy's
expected skill-<i>retention</i> advantage does not appear — a second, independent line
of evidence for the negative result.</p>
"""

    # aggregate disruption for the headline
    def drate(b, a):
        nd = sum(r["n_disrupted"] for r in rows if r["board"] == b and r["agent"] == a)
        ns = sum(r["n_shifts"]    for r in rows if r["board"] == b and r["agent"] == a)
        return f"{nd}/{ns}"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>RQ3 Phase-2 Results</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
   max-width:900px;margin:2rem auto;padding:0 1.2rem;color:#1a1a1a;line-height:1.5}}
 h1{{font-size:1.5rem;margin-bottom:.2rem}} h2{{font-size:1.15rem;margin-top:2rem;
   border-bottom:2px solid #eee;padding-bottom:.3rem}}
 .sub{{color:#666;margin-top:0}}
 .headline{{background:#f4f7ff;border-left:4px solid {PPO_C};padding:.8rem 1rem;border-radius:4px}}
 table{{border-collapse:collapse;width:100%;font-size:.85rem;margin:.6rem 0}}
 th,td{{border:1px solid #ddd;padding:4px 8px;text-align:center}}
 th{{background:#f5f5f5}} tr.fun td{{background:#fff5f7}}
 img{{max-width:100%;height:auto;display:block;margin:1rem auto}}
 ul{{margin:.4rem 0}} code{{background:#f0f0f0;padding:1px 4px;border-radius:3px}}
 .cap{{color:#666;font-size:.82rem;text-align:center;margin-top:-.6rem}}
</style></head><body>

<h1>RQ3 — Recovery After Abrupt Opponent Strategy Shifts in Go</h1>
<p class="sub">Feudal Networks (FuN) vs flat PPO · Phase-2 results · Brad Scott (SCTBRA008)</p>

<div class="headline">
<b>Headline.</b> Across all 36 conditions and every metric, the flat PPO agent is
disrupted <i>less often</i>, dips <i>less deeply</i>, and recovers <i>faster</i>
than the FeUdal Network. PPO's win rate fell below the 80%-of-baseline bar in only
{drate('9x9','ppo')} shifts on 9×9 and {drate('13x13','ppo')} on 13×13, versus
{drate('9x9','feudal')} and {drate('13x13','feudal')} for FuN. The hierarchy did
<b>not</b> improve recovery — a clean negative result against the proposal's hypothesis.
</div>

<h2>Setup</h2>
<ul>
 <li><b>Environment:</b> PettingZoo <code>go_v5</code>, boards 9×9 and 13×13, agent plays black vs fixed heuristic opponents.</li>
 <li><b>Phase 1:</b> train each agent to its per-opponent capability baseline.</li>
 <li><b>Phase 2:</b> resume a Phase-1 checkpoint, then apply abrupt mid-training opponent shifts —
   3 magnitudes (LOW corner→edge, MED corner→defensive, HIGH greedy→defensive) × 3 frequencies
   (f1 single, f2 periodic, f3 frequent). All shifts are between opponents both agents can master,
   so recovery is not confounded with raw capability. <b>Three seeds</b> per condition
   (108 Phase-2 runs); results below are mean over seeds.</li>
 <li><b>Metrics</b> (all vs each agent's <i>own</i> pre-shift baseline, rolling window = 100 games):
   <b>disruption rate</b> (did win rate fall below 0.8×baseline);
   <b>recovery time</b> (games spent below the bar until it returns);
   <b>dip depth</b> (how far below baseline); and
   <b>adaptation cost</b> (area under the dip, floor-free, comparable across boards).</li>
</ul>

<h2>1. Disruption rate — the headline signal</h2>
<img src="data:image/png;base64,{f1}">
<p class="cap">PPO is rarely knocked below the recovery bar; FuN routinely is, on both boards.</p>

<h2>2. Recovery duration (9×9)</h2>
<img src="data:image/png;base64,{f2}">
<p class="cap">When disrupted, FuN stays below baseline far longer, worsening with shift magnitude.</p>

<h2>3. Adaptation cost — both boards</h2>
<img src="data:image/png;base64,{f3}">
<p class="cap">Area under the win-rate dip (log scale). FuN pays 3–100× PPO's cost in every case.</p>

<h2>4. Representative win-rate trajectories (single HIGH shift)</h2>
<img src="data:image/png;base64,{ov9}">
<img src="data:image/png;base64,{ov13}">
<p class="cap">After the greedy→defensive shift, PPO barely moves; FuN collapses and recovers slowly.</p>

<h2>5. Full results — all 36 conditions</h2>
{summary_table(rows)}
<p class="cap">Recovery = mean games below the bar over disrupted shifts (— = never disrupted).</p>
{reuse_html}
<h2>Limitations (stated up front)</h2>
<ul>
 <li><b>Three seeds</b> per condition — the FuN–PPO gaps are large and unanimous across all
   36 conditions and hold across seeds; some single-frequency cells still carry wide spread
   (few shifts), so per-cell numbers should be read as indicative.</li>
 <li><b>13×13 recovery <i>duration</i> is short</b> (a 13×13 game ≈ 3× the training of a 9×9 game),
   so on the larger board the disruption <i>rate</i>, <i>dip depth</i> and <i>adaptation cost</i>
   carry the signal; all three still show FuN clearly worse.</li>
 <li>Metrics are measured against each agent's own baseline, which controls for raw capability;
   a residual caveat is that FuN is a weaker, noisier learner overall.</li>
</ul>

<p class="cap">Generated from <code>phase2_results/summary_by_condition.csv</code>. Reproduce:
<code>python3 phase2/make_report.py</code></p>
</body></html>"""

    out = os.path.join(RESULTS, "RESULTS_REPORT.html")
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out, f"({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
