# ══════════════════════════════════════════════════════════════
# PHASE 2 ANALYSIS (offline) — RQ3 recovery-time results & figures
#
# Reads every games.csv produced by the Phase-2 runs (both agents),
# RECOMPUTES recovery per shift straight from the per-game rows (so the
# results don't depend on the run-time detector), and writes:
#
#   phase2_results/recovery_table.csv   one row per (run, shift):
#       agent, board, magnitude, frequency, seed, condition, shift_idx,
#       to_opponent, baseline, threshold80 (=0.8*baseline), min_rolling,
#       perf_drop (dip), end_rolling, disrupted (1/0), recovered (1/0),
#       recovery_games (0 if never dipped below the band; duration if it
#       returned; censored lower bound if it dipped but never returned),
#       adapt_cost_games, adapt_cost_ksteps, segment_games.
#   phase2_results/summary_by_condition_by_seed.csv   EACH SEED's own metrics,
#       one row per (board, agent, magnitude, frequency, seed).
#   phase2_results/summary_by_condition.csv   MERGED across seeds, one row per
#       (board, agent, magnitude, frequency): n_seeds, n_shifts, n_disrupted,
#       recovery_mean/recovery_se, dip_mean/dip_se, worst_dip,
#       adapt_cost_mean/adapt_cost_se. Each metric is collapsed per
#       (condition, seed) first, then meaned over seeds with SE=stdev/sqrt(n)
#       (SE=0 with a single seed).
#   phase2_results/<run>__rolling.png   rolling win rate vs games,
#       shift points marked (needs matplotlib; skipped if missing)
#
# Definitions (must match phase2.py):
#   * ONE continuous rolling window of W=100 completed games (matches the CSV
#     'rolling' col; recomputed here from the raw 'win' column so it is exact).
#   * baseline(shift k) = rolling value on the last game BEFORE shift k
#   * recovery(shift k) = DURATION (games) of the disruption: from the first drop
#     below the 0.8*baseline band to the first return to it (0 if it never dips
#     that far; censored lower bound if it dips but never returns). Measuring the
#     excursion length (not distance from the shift) times a real post-shift dip
#     correctly while a trivial late variance blip counts only its own few games.
#     (An "any dip -> full baseline" variant was tried and dropped — it inverts
#     for near-ceiling agents; recomputable offline via _excursion(seg, baseline).)
#   * perf_drop(shift k) = baseline - min(rolling) within segment k  (the "dip")
#   * adapt_cost(shift k) = EXTRA metric: area under the dip = sum of
#                           max(0, baseline-rolling) over the WHOLE post-shift
#                           segment (deficit-only: games at/above baseline add
#                           0), in win-rate*games and win-rate*ksteps. Floor-free
#                           (never saturates) and folds dip depth + duration
#                           into one number; the step-integrated form does not
#                           depend on game length. Compare within a
#                           (board, magnitude, frequency) cell (segment lengths
#                           match there for both agents).
#
# Usage (local or HPC — only needs the CSVs):
#   python3 analyze_phase2.py [root ...]
# Default roots: ./models/ppo_go ./models/feudal
# ══════════════════════════════════════════════════════════════

import csv
import os
import statistics
import sys
from collections import defaultdict, deque

WINDOW        = 100   # rolling/recovery window, in completed games (both boards)
RECOVERY_FRAC = 0.8
OUT_DIR       = "phase2_results"

# ── Magnitude relabel ─────────────────────────────────────────
# The Phase-2 runs were RE-RUN clean with corrected tags, so the folder names
# now already encode the TRUE magnitude:
#   low  = corner->edge   med = corner->defensive   high = greedy->defensive
# Hence identity mapping. (The earlier swap {"med":"high","high":"med"} was only
# needed for the original mislabeled runs, which have since been deleted.)
RELABEL   = {"low": "low", "med": "med", "high": "high"}
MAG_ORDER = {"low": 0, "med": 1, "high": 2}   # for stable, readable sorting


def parse_condition(cond):
    """Folder name -> (magnitude, frequency, seed, label).
    Handles 'greedy-phase2-med-f1' (seed 0) and 'greedy-phase2-med-f1-s1'.
    Magnitude corrected via RELABEL; missing seed suffix => seed '0'."""
    mag, freq, seed = "", "", "0"
    if "phase2-" in cond:
        parts = cond.split("phase2-", 1)[1].split("-")   # e.g. ['med','f1','s1']
        mag  = parts[0] if len(parts) >= 1 else ""
        freq = parts[1] if len(parts) >= 2 else ""
        for p in parts[2:]:                              # find the -s<N> token
            if p.startswith("s") and p[1:].isdigit():
                seed = p[1:]
    mag = RELABEL.get(mag, mag)                           # <-- the correction
    label = f"{mag}-{freq}" if freq else (mag or cond)
    return mag, freq, seed, label


def find_runs(roots):
    """Yield (agent, board, condition, csv_path) for every Phase-2 games.csv.
    Walks the tree so board-nested layouts (models/<agent>/<board>/<run>/) work."""
    for root in roots:
        agent = "ppo" if "ppo" in os.path.basename(root.rstrip("/")) else "feudal"
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            d = os.path.basename(dirpath)
            if "phase2" in d and "games.csv" in filenames:
                board = os.path.basename(os.path.dirname(dirpath))   # e.g. 9x9 / 13x13
                yield agent, board, d, os.path.join(dirpath, "games.csv")


def load_rows(path):
    with open(path) as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({
                "game_idx":          int(r["game_idx"]),
                "global_step":       int(r["global_step"]),
                "opponent":          r["opponent"],
                "win":               int(r["win"]),
                "rolling":           float(r["rolling"]),
                "shift_idx":         int(r["shift_idx"]),
                "games_since_shift": int(r["games_since_shift"]),
            })
    return rows


def rolling_series(wins, W):
    """Trailing mean of the last W games (expanding until the window fills) —
    the same definition phase2.RecoveryTracker uses at run time."""
    dq, out = deque(maxlen=W), []
    for w in wins:
        dq.append(w)
        out.append(sum(dq) / len(dq))
    return out


def _excursion(seg_roll, level):
    """Below-`level` excursion within a segment. Returns (disrupted, recovered,
    games): disrupted = rolling fell below `level` at least once; recovered = it
    later returned to >= level; games = duration below `level` from the first
    drop to the first return (0 if never disrupted; if disrupted but never
    returns, a censored lower bound = first-drop .. segment end)."""
    if min(seg_roll) >= level:
        return False, True, 0
    fb  = next(i for i, v in enumerate(seg_roll) if v < level)
    ret = next((j for j in range(fb, len(seg_roll)) if seg_roll[j] >= level), None)
    if ret is None:
        return True, False, len(seg_roll) - fb
    return True, True, ret - fb


def analyze_run(rows, W):
    """Recompute per-shift baseline / dip / recovery / adaptation-cost from raw
    rows, using a rolling window of W games. Annotates each row with 'roll_w'
    (the W-window rolling value) so plots can use the same series."""
    for r, rv in zip(rows, rolling_series([r["win"] for r in rows], W)):
        r["roll_w"] = rv
    results = []
    shift_ids = sorted({r["shift_idx"] for r in rows if r["shift_idx"] >= 0})
    # A = the checkpoint-source opponent = the one played in the earliest
    # (baseline) segment. Schedules alternate B,A,B,A... and always end on A, so
    # a shift back to A is a policy-REUSE probe (did it retain its A-policy?).
    A_opp = None
    if shift_ids:
        base_seg = [r for r in rows if r["shift_idx"] == shift_ids[0]]
        if base_seg:
            A_opp = base_seg[0]["opponent"]
    for k in shift_ids:
        seg  = [r for r in rows if r["shift_idx"] == k]
        prev = [r for r in rows if r["shift_idx"] == k - 1]
        if not seg or not prev:
            continue
        baseline  = prev[-1]["roll_w"]           # rolling just before shift k
        thr80     = RECOVERY_FRAC * baseline     # the 80% band
        seg_roll  = [r["roll_w"] for r in seg]
        min_roll  = min(seg_roll)

        # adaptation cost: area under the dip over the WHOLE segment, deficit-only
        # (games at/above baseline add 0), so it captures every dip regardless of
        # when it occurs and never saturates.
        area_g = area_s = 0.0
        for i, r in enumerate(seg):
            short   = max(0.0, baseline - r["roll_w"])
            dstep   = r["global_step"] - seg[i - 1]["global_step"] if i else 0
            area_g += short
            area_s += short * dstep

        # Recovery time = DURATION (games) of the disruption: from the first drop
        # below the 0.8*baseline band to the first return to it. 0 if the shift
        # never breached the band; a censored lower bound (recovered=0) if it
        # breached but never returned. Measuring the excursion length (not the
        # distance from the shift) times a real post-shift dip correctly while a
        # trivial late variance blip counts only its own few games. (An "any dip
        # -> full baseline" variant was tried and dropped: it inverts for
        # near-ceiling agents; it can be recomputed offline from this CSV if ever
        # wanted, via _excursion(seg_roll, baseline).)
        disrupted, recovered, recovery = _excursion(seg_roll, thr80)

        results.append({
            "shift_idx":         k,
            "to_opponent":       seg[0]["opponent"],
            "returns_to_A":      1 if (A_opp and seg[0]["opponent"] == A_opp) else 0,
            "shift_at_step":     seg[0]["global_step"],
            "baseline":          round(baseline, 4),
            "threshold80":       round(thr80, 4),
            "min_rolling":       round(min_roll, 4),
            "perf_drop":         round(baseline - min_roll, 4),
            "end_rolling":       round(seg_roll[-1], 4),
            "disrupted":         1 if disrupted else 0,
            "recovered":         1 if recovered else 0,
            "recovery_games":    recovery,   # games below 0.8*baseline (0 if none)
            "adapt_cost_games":  round(area_g, 2),
            "adapt_cost_ksteps": round(area_s / 1000, 2),
            "segment_games":     len(seg),
        })
    return results


def plot_run(rows, title, out_png, W):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    games   = [r["game_idx"] for r in rows]
    rolling = [r.get("roll_w", r["rolling"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(games, rolling, lw=1)
    # mark each shift (first game of each segment)
    seen = set()
    for r in rows:
        if r["shift_idx"] >= 0 and r["shift_idx"] not in seen:
            seen.add(r["shift_idx"])
            ax.axvline(r["game_idx"], ls="--", lw=0.8, color="red")
            ax.annotate(r["opponent"], (r["game_idx"], 1.02),
                        fontsize=7, rotation=45, annotation_clip=False)
    ax.set_xlabel("completed games")
    ax.set_ylabel(f"rolling win rate (W={W})")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def main():
    roots = sys.argv[1:] or ["./models/ppo_go", "./models/feudal"]
    os.makedirs(OUT_DIR, exist_ok=True)

    table = []
    for agent, board, condition, path in find_runs(roots):
        rows = load_rows(path)
        if not rows:
            continue
        magnitude, frequency, seed, label = parse_condition(condition)
        res = analyze_run(rows, WINDOW)
        for r in res:
            table.append({"agent": agent, "board": board, "magnitude": magnitude,
                          "frequency": frequency, "seed": seed, "label": label,
                          "condition": condition, **r})
        plotted = plot_run(rows, f"{agent} {board} — {label} s{seed}  ({condition})",
                           os.path.join(OUT_DIR,
                                        f"{agent}__{board}__{label}__s{seed}__rolling.png"), WINDOW)
        print(f"  {agent:6s} {board:6s} {label:10s} s{seed} "
              f"shifts={len(res)} games={rows[-1]['game_idx']:>6,} "
              f"plot={'yes' if plotted else 'no (no matplotlib)'}")

    if not table:
        print("No Phase-2 games.csv found under:", roots)
        return

    out_csv = os.path.join(OUT_DIR, "recovery_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"\n✓ wrote {out_csv} ({len(table)} shift records)")

    def _mag(k):
        return MAG_ORDER.get(k, 9)

    # ---- Step 1: collapse each (condition, SEED) to one number per metric ---
    # Recovery is 0-filled over ALL shifts (0 = never breached the band) so every
    # agent gets a number in every cell. dip = mean drop below baseline.
    per_cs = defaultdict(lambda: {"rec": [], "dip": [], "cost": [], "dis": 0, "n": 0})
    for t in table:
        d = per_cs[(t["board"], t["agent"], t["magnitude"], t["frequency"], t["seed"])]
        d["rec"].append(float(t["recovery_games"]))
        d["dip"].append(t["perf_drop"])
        d["cost"].append(t["adapt_cost_ksteps"])
        d["dis"] += t["disrupted"]
        d["n"]   += 1

    def _avg(x):
        return sum(x) / len(x)

    # ---- per-SEED personal metrics: one row per (condition, seed) ----------
    seed_stats, by_seed_rows = {}, []
    for (board, agent, mag, freq, seed), d in per_cs.items():
        st = {"rec": _avg(d["rec"]), "dip": _avg(d["dip"]), "wdip": max(d["dip"]),
              "cost": _avg(d["cost"]), "dis": d["dis"], "n": d["n"]}
        seed_stats[(board, agent, mag, freq, seed)] = st
        by_seed_rows.append({
            "board": board, "agent": agent, "magnitude": mag, "frequency": freq,
            "seed": seed, "n_shifts": d["n"], "n_disrupted": d["dis"],
            "recovery_games": round(st["rec"], 1),
            "mean_dip": round(st["dip"], 3), "worst_dip": round(st["wdip"], 3),
            "adapt_cost_ksteps": round(st["cost"], 2),
        })
    by_seed_rows.sort(key=lambda r: (r["board"], r["agent"], _mag(r["magnitude"]),
                                     r["frequency"], r["seed"]))
    out_seed = os.path.join(OUT_DIR, "summary_by_condition_by_seed.csv")
    with open(out_seed, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_seed_rows[0].keys()))
        w.writeheader(); w.writerows(by_seed_rows)
    print(f"✓ wrote {out_seed} ({len(by_seed_rows)} condition×seed rows)")

    # ---- Step 2: MERGE across seeds -> mean +/- SE -------------------------
    def mean_se(vals):
        mu = sum(vals) / len(vals)
        se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
        return round(mu, 2), round(se, 2)

    conds = sorted({k[:4] for k in seed_stats},
                   key=lambda c: (c[0], c[1], _mag(c[2]), c[3]))
    summary_rows = []
    for (board, agent, mag, freq) in conds:
        S = [seed_stats[k] for k in seed_stats if k[:4] == (board, agent, mag, freq)]
        rec_mu, rec_se   = mean_se([s["rec"] for s in S])
        dip_mu, dip_se   = mean_se([s["dip"] for s in S])
        cost_mu, cost_se = mean_se([s["cost"] for s in S])
        summary_rows.append({
            "board": board, "agent": agent, "magnitude": mag, "frequency": freq,
            "n_seeds": len(S), "n_shifts": sum(s["n"] for s in S),
            "n_disrupted": sum(s["dis"] for s in S),
            "recovery_mean": rec_mu, "recovery_se": rec_se,
            "dip_mean": dip_mu, "dip_se": dip_se,
            "worst_dip": round(max(s["wdip"] for s in S), 3),
            "adapt_cost_mean": cost_mu, "adapt_cost_se": cost_se,
        })

    out_summary = os.path.join(OUT_DIR, "summary_by_condition.csv")
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    n_seeds = max(r["n_seeds"] for r in summary_rows)
    print(f"✓ wrote {out_summary} ({len(summary_rows)} conditions, merged over up to {n_seeds} seed(s))")

    # ---- console view: recovery / dip / cost as mean +/- SE across seeds ----
    print(f"\n=== Per-condition summary — merged over {n_seeds} seed(s), mean±SE "
          "(recovery = games below 0.8×baseline, 0-filled so both agents appear) ===")
    print(f"  {'board':6s} {'agent':6s} {'cond':8s} {'sd':>2s} {'disr':>6s} "
          f"{'recovery(g)':>14s} {'dip':>12s} {'cost_ks':>12s}")
    for r in summary_rows:
        cond = f"{r['magnitude']}-{r['frequency']}"
        disr = f"{r['n_disrupted']}/{r['n_shifts']}"
        rec  = f"{r['recovery_mean']:.0f}±{r['recovery_se']:.0f}"
        dip  = f"{r['dip_mean']:.2f}±{r['dip_se']:.2f}"
        cost = f"{r['adapt_cost_mean']:.1f}±{r['adapt_cost_se']:.1f}"
        print(f"  {r['board']:6s} {r['agent']:6s} {cond:8s} {r['n_seeds']:>2d} "
              f"{disr:>6s} {rec:>14s} {dip:>12s} {cost:>12s}")
    print("  recovery = mean games below 0.8×baseline (0 = never breached it); "
          "dip = mean drop below baseline; cost = area under the dip.")

    # ---- Policy reuse: novel-to-B vs return-to-A shifts --------------------
    # Every shift is either TO the novel opponent B or BACK to A (the trained
    # opponent). Small disruption on return-to-A = the agent retained/reused its
    # A-policy (low forgetting); a large dip means it overwrote A while adapting
    # to B. Comparing PPO vs FuN here directly tests the hierarchy's skill-reuse
    # claim. Aggregated over all seeds/magnitudes/frequencies per (board, agent).
    reuse = defaultdict(lambda: {"dip": [], "rec": [], "cost": [], "dis": 0, "n": 0})
    for t in table:
        tgt = "return-A" if t["returns_to_A"] else "novel-B"
        d = reuse[(t["board"], t["agent"], tgt)]
        d["dip"].append(t["perf_drop"])
        d["rec"].append(float(t["recovery_games"]))
        d["cost"].append(t["adapt_cost_ksteps"])
        d["dis"] += t["disrupted"]
        d["n"]   += 1
    reuse_rows = []
    for (board, agent, tgt), d in reuse.items():
        reuse_rows.append({
            "board": board, "agent": agent, "shift_type": tgt,
            "n_shifts": d["n"], "n_disrupted": d["dis"],
            "mean_dip": round(_avg(d["dip"]), 3),
            "mean_recovery": round(_avg(d["rec"]), 1),
            "mean_cost_ksteps": round(_avg(d["cost"]), 2),
        })
    reuse_rows.sort(key=lambda r: (r["board"], r["agent"], r["shift_type"]))
    out_reuse = os.path.join(OUT_DIR, "policy_reuse.csv")
    with open(out_reuse, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reuse_rows[0].keys()))
        w.writeheader(); w.writerows(reuse_rows)
    print(f"\n✓ wrote {out_reuse}")

    print("\n=== Policy reuse — novel-to-B vs return-to-A (low dip on return-A = "
          "retained/reused A-policy) ===")
    print(f"  {'board':6s} {'agent':6s} {'shift':9s} {'disr':>7s} {'dip':>7s} "
          f"{'recov(g)':>9s} {'cost_ks':>8s}")
    for r in reuse_rows:
        print(f"  {r['board']:6s} {r['agent']:6s} {r['shift_type']:9s} "
              f"{str(r['n_disrupted']) + '/' + str(r['n_shifts']):>7s} "
              f"{r['mean_dip']:>7.3f} {r['mean_recovery']:>9.1f} "
              f"{r['mean_cost_ksteps']:>8.2f}")


if __name__ == "__main__":
    main()
