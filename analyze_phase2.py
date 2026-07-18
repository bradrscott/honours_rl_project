# ══════════════════════════════════════════════════════════════
# PHASE 2 ANALYSIS (offline) — RQ3 recovery-time results & figures
#
# Reads every games.csv produced by the Phase-2 runs (both agents),
# RECOMPUTES recovery per shift straight from the per-game rows (so the
# results don't depend on the run-time detector), and writes:
#
#   phase2_results/recovery_table.csv   one row per (run, shift):
#       agent, condition, shift_idx, to_opponent, baseline,
#       perf_drop, recovery_games (blank = never recovered in budget)
#   phase2_results/<run>__rolling.png   rolling win rate vs games,
#       shift points marked (needs matplotlib; skipped if missing)
#
# Definitions (must match phase2.py):
#   * ONE continuous rolling window of W=100 games (the CSV 'rolling' col)
#   * baseline(shift k) = rolling value on the last game BEFORE shift k
#   * recovery(shift k) = first game with games_since_shift >= W and
#                         rolling >= 0.8 * baseline
#   * perf_drop(shift k) = baseline - min(rolling) within segment k
#
# Usage (local or HPC — only needs the CSVs):
#   python3 analyze_phase2.py [root ...]
# Default roots: ./models/ppo_go ./models/feudal
# ══════════════════════════════════════════════════════════════

import csv
import os
import sys

W_RECOVERY    = 100
RECOVERY_FRAC = 0.8
OUT_DIR       = "phase2_results"


def find_runs(roots):
    """Yield (agent, condition, csv_path) for every Phase-2 games.csv."""
    for root in roots:
        agent = "ppo" if "ppo" in os.path.basename(root.rstrip("/")) else "feudal"
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            path = os.path.join(root, d, "games.csv")
            if "phase2" in d and os.path.isfile(path):
                yield agent, d, path


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


def analyze_run(rows):
    """Recompute per-shift baseline / perf drop / recovery from raw rows."""
    results = []
    shift_ids = sorted({r["shift_idx"] for r in rows if r["shift_idx"] >= 0})
    for k in shift_ids:
        seg  = [r for r in rows if r["shift_idx"] == k]
        prev = [r for r in rows if r["shift_idx"] == k - 1]
        if not seg or not prev:
            continue
        baseline  = prev[-1]["rolling"]          # rolling just before shift k
        min_roll  = min(r["rolling"] for r in seg)
        recovery  = None
        for r in seg:
            if (r["games_since_shift"] >= W_RECOVERY
                    and r["rolling"] >= RECOVERY_FRAC * baseline):
                recovery = r["games_since_shift"]
                break
        results.append({
            "shift_idx":      k,
            "to_opponent":    seg[0]["opponent"],
            "shift_at_step":  seg[0]["global_step"],
            "baseline":       round(baseline, 4),
            "min_rolling":    round(min_roll, 4),
            "perf_drop":      round(baseline - min_roll, 4),
            "recovery_games": recovery if recovery is not None else "",
            "segment_games":  len(seg),
        })
    return results


def plot_run(rows, title, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    games   = [r["game_idx"] for r in rows]
    rolling = [r["rolling"] for r in rows]
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
    ax.set_ylabel(f"rolling win rate (W={W_RECOVERY})")
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
    for agent, condition, path in find_runs(roots):
        rows = load_rows(path)
        if not rows:
            continue
        for res in analyze_run(rows):
            table.append({"agent": agent, "condition": condition, **res})
        plotted = plot_run(rows, f"{agent} — {condition}",
                           os.path.join(OUT_DIR, f"{agent}__{condition}__rolling.png"))
        print(f"  {agent:6s} {condition:35s} shifts={len(analyze_run(rows))} "
              f"games={rows[-1]['game_idx']:>6,} plot={'yes' if plotted else 'no (no matplotlib)'}")

    if not table:
        print("No Phase-2 games.csv found under:", roots)
        return

    out_csv = os.path.join(OUT_DIR, "recovery_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"\n✓ wrote {out_csv} ({len(table)} shift records)")

    # quick console summary: mean recovery per (agent, condition)
    print("\n=== Mean recovery games per run (blank recoveries excluded) ===")
    byrun = {}
    for t in table:
        key = (t["agent"], t["condition"])
        byrun.setdefault(key, []).append(t["recovery_games"])
    for (agent, cond), recs in sorted(byrun.items()):
        vals = [r for r in recs if r != ""]
        mean = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {agent:6s} {cond:35s} shifts={len(recs)} recovered={len(vals)} "
              f"mean_recovery={mean:.0f}" if vals else
              f"  {agent:6s} {cond:35s} shifts={len(recs)} recovered=0")


if __name__ == "__main__":
    main()
