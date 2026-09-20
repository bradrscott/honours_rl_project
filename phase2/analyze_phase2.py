# Phase-2 analysis reads every games.csv produced by the Phase-2
# runs, independently recomputes each shift's recovery time, dip depth and
# adaptation cost straight from the raw per-game rows - writes the
# three summary CSVs described in results/phase2/

# Standard libraries
import csv
import os
import statistics
import sys
from collections import defaultdict, deque


# settings and paths this analysis uses throughout
WINDOW        = 100
RECOVERY_FRAC = 0.8
OUT_DIR       = "results/phase2"
SUMMARY_DIR   = os.path.join(OUT_DIR, "summary")

# folder names already carry the correct magnitude, so this is an identity mapping 
RELABEL   = {"low": "low", "med": "med", "high": "high"}
MAG_ORDER = {"low": 0, "med": 1, "high": 2}   


# Pulls the condition out of a run's folder name.
def parse_condition(cond):
    mag, freq, seed = "", "", "0"
    if "phase2-" in cond:
        parts = cond.split("phase2-", 1)[1].split("-")  
        mag  = parts[0] if len(parts) >= 1 else ""
        freq = parts[1] if len(parts) >= 2 else ""

        # find the -s<N> seed token, if present
        for p in parts[2:]:
            if p.startswith("s") and p[1:].isdigit():
                seed = p[1:]
    mag = RELABEL.get(mag, mag)                           
    label = f"{mag}-{freq}" if freq else (mag or cond)
    return mag, freq, seed, label


# Searches the given root folders for every Phase-2 run and for each one
# found, figures out which agent and board it belongs to from its folder
# path then hands back that run's games.csv file to be analysed.
def find_runs(roots):
    for root in roots:
        agent = "ppo" if "ppo" in os.path.basename(root.rstrip("/")) else "feudal"
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            d = os.path.basename(dirpath)
            if "phase2" in d and "games.csv" in filenames:
                board = os.path.basename(os.path.dirname(dirpath))   # e.g. 9x9 / 13x13
                yield agent, board, d, os.path.join(dirpath, "games.csv")


# Read one games.csv into a list of dictionaries, with every column converted to its proper type.
def load_rows(path):
    with open(path) as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({
                "game_idx": int(r["game_idx"]),
                "global_step": int(r["global_step"]),
                "opponent": r["opponent"],
                "win": int(r["win"]),
                "rolling": float(r["rolling"]),
                "shift_idx": int(r["shift_idx"]),
                "games_since_shift": int(r["games_since_shift"]),
            })
    return rows


# Trailing mean of the last W games (expanding until the window fills) — the
# same definition phase2.RecoveryTracker uses at run time, recomputed here
# straight from the raw win column so it is exact.
def rolling_series(wins, W):
    dq, out = deque(maxlen=W), []
    for w in wins:
        dq.append(w)
        out.append(sum(dq) / len(dq))
    return out


# Measures one dip below the 0.8 level - whether it happened, whether it recovered,
# and how many games that took (a lower-bound estimate if it never recovered).
def _excursion(seg_roll, level):
    if min(seg_roll) >= level:
        return False, True, 0
    fb  = next(i for i, v in enumerate(seg_roll) if v < level)
    ret = next((j for j in range(fb, len(seg_roll)) if seg_roll[j] >= level), None)
    if ret is None:
        return True, False, len(seg_roll) - fb
    return True, True, ret - fb


# Recompute per-shift baseline, dip, recovery and adaptation-cost from one
# run's raw rows, using a rolling window of W games. Annotates each row with
# 'roll_w' (the W-window rolling value) used by the metric computations below.
def analyze_run(rows, W):
    for r, rv in zip(rows, rolling_series([r["win"] for r in rows], W)):
        r["roll_w"] = rv
    results = []
    shift_ids = sorted({r["shift_idx"] for r in rows if r["shift_idx"] >= 0})

    # A = the checkpoint-source opponent, the one played in the earliest
    # (baseline) segment. Recorded per shift as returns_to_A so it's in the
    # raw data if ever needed, but not used in any metric here.
    A_opp = None
    if shift_ids:
        base_seg = [r for r in rows if r["shift_idx"] == shift_ids[0]]
        if base_seg:
            A_opp = base_seg[0]["opponent"]

    # one iteration per shift in this run
    for k in shift_ids:
        seg  = [r for r in rows if r["shift_idx"] == k]
        prev = [r for r in rows if r["shift_idx"] == k - 1]
        if not seg or not prev:
            continue
        baseline  = prev[-1]["roll_w"]           
        thr80     = RECOVERY_FRAC * baseline     
        seg_roll  = [r["roll_w"] for r in seg]
        min_roll  = min(seg_roll)

        # Adaptation cost - area under the dip over the whole segment,
        # deficit-only (games at/above baseline add 0), in win-rate games and
        # win-rate ksteps. This captures every dip regardless of when it
        # occurs and never saturates, unlike recovery time.
        area_g = area_s = 0.0
        for i, r in enumerate(seg):
            short = max(0.0, baseline - r["roll_w"])
            dstep = r["global_step"] - seg[i - 1]["global_step"] if i else 0
            area_g += short
            area_s += short * dstep

        # Recovery time: duration (games) of the disruption, from the first
        # drop below the 0.8 baseline band to the first return to it.
        disrupted, recovered, recovery = _excursion(seg_roll, thr80)

        results.append({
            "shift_idx": k,
            "to_opponent": seg[0]["opponent"],
            "returns_to_A": 1 if (A_opp and seg[0]["opponent"] == A_opp) else 0,
            "shift_at_step": seg[0]["global_step"],
            "baseline": round(baseline, 4),
            "threshold80": round(thr80, 4),
            "min_rolling": round(min_roll, 4),
            "perf_drop": round(baseline - min_roll, 4),
            "end_rolling": round(seg_roll[-1], 4),
            "disrupted": 1 if disrupted else 0,
            "recovered": 1 if recovered else 0,
            "recovery_games": recovery,   # games below 0.8*baseline (0 if none)
            "adapt_cost_games": round(area_g, 2),
            "adapt_cost_ksteps": round(area_s / 1000, 2),
            "segment_games": len(seg),
        })
    return results


# Analyses every Phase-2 run under and writes the three summary CSVs
def main():
    roots = sys.argv[1:] or ["./results/phase2/ppo_go", "./results/phase2/feudal"]
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # 1. Walk through every run's games.csv and recompute its per-shift metrics
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
        print(f"  {agent:6s} {board:6s} {label:10s} s{seed} "
              f"shifts={len(res)} games={rows[-1]['game_idx']:>6,}")

    if not table:
        print("No Phase-2 games.csv found under:", roots)
        return

    # write recovery_table.csv - one row per (run, shift)
    out_csv = os.path.join(SUMMARY_DIR, "recovery_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"\nwrote {out_csv} ({len(table)} shift records)")

    # sort key - magnitude in the fixed low/med/high order not alphabetical
    def _mag(k):
        return MAG_ORDER.get(k, 9)

    # 2. Make each condition, seed  to one number per metric.
    # Recovery is 0-filled over ALL shifts (0 = never breached the band) so
    # every agent gets a number in every cell. dip = mean drop below baseline.
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

    # 3. Write summary_by_condition_by_seed.csv - each seed's own metrics, one row per (condition, seed)
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
    out_seed = os.path.join(SUMMARY_DIR, "summary_by_condition_by_seed.csv")
    with open(out_seed, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_seed_rows[0].keys()))
        w.writeheader(); w.writerows(by_seed_rows)
    print(f"wrote {out_seed} ({len(by_seed_rows)} condition×seed rows)")

    # mean +/- standard error across a list of per-seed values (SE=0 for a
    # single seed, since stdev needs at least 2 points)
    def mean_se(vals):
        mu = sum(vals) / len(vals)
        se = statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
        return round(mu, 2), round(se, 2)

    # per-condition raw shifts (pooled over seeds) — extra column info
    shifts_by_cond = defaultdict(list)
    for t in table:
        shifts_by_cond[(t["board"], t["agent"], t["magnitude"], t["frequency"])].append(t)

    # 4. Merge across seeds into summary_by_condition.csv - one row per
    # condition, with mean +/- SE over the seeds
    conds = sorted({k[:4] for k in seed_stats},
                   key=lambda c: (c[0], c[1], _mag(c[2]), c[3]))
    summary_rows = []
    for (board, agent, mag, freq) in conds:
        S = [seed_stats[k] for k in seed_stats if k[:4] == (board, agent, mag, freq)]
        rec_mu, rec_se = mean_se([s["rec"] for s in S])
        dip_mu, dip_se = mean_se([s["dip"] for s in S])
        cost_mu, cost_se = mean_se([s["cost"] for s in S])

        sh = shifts_by_cond[(board, agent, mag, freq)]
        disr = [t for t in sh if t["disrupted"]]
        n_rec = sum(1 for t in disr if t["recovered"])
        mean_rec_disr = round(sum(t["recovery_games"] for t in disr) / len(disr), 1) if disr else ""

        summary_rows.append({
            "board": board, "agent": agent, "magnitude": mag, "frequency": freq,
            "n_seeds": len(S), "n_shifts": sum(s["n"] for s in S),
            "n_disrupted": sum(s["dis"] for s in S),
            "recovery_mean": rec_mu, "recovery_se": rec_se,
            "dip_mean": dip_mu, "dip_se": dip_se,
            "worst_dip": round(max(s["wdip"] for s in S), 3),
            "adapt_cost_mean": cost_mu, "adapt_cost_se": cost_se,
            # extra per-condition columns beyond the mean+-SE metrics above
            "n_recovered": n_rec, "n_censored": len(disr) - n_rec,
            "mean_recovery_disrupted": mean_rec_disr,
            "mean_dip": dip_mu,
            "mean_adapt_cost_ksteps": cost_mu,
        })

    out_summary = os.path.join(SUMMARY_DIR, "summary_by_condition.csv")
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    n_seeds = max(r["n_seeds"] for r in summary_rows)
    print(f"wrote {out_summary} ({len(summary_rows)} conditions, merged over up to {n_seeds} seed(s))")


if __name__ == "__main__":
    main()
