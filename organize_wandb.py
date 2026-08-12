# ══════════════════════════════════════════════════════════════
# Tidy up a messy wandb project: parse each run's NAME and stamp
# structured tags + config on it, so the wandb UI can group/filter.
#
# It sets, per run:  tags = {agent, board, phase, opp-<name>, magnitude, freq}
# and config.meta_* = the same fields, so you can "Group by" any of them.
#
# SAFE: merges into existing tags (never deletes), only ADDS meta_* config
# keys (won't touch training config). Dry-run by default.
#
#   # preview what it WOULD do (no changes):
#   python organize_wandb.py
#   # actually apply:
#   python organize_wandb.py --apply
#
# Entity/project default to your project; override with --entity/--project.
# Needs to be logged in (you already are, since runs log online). If it
# complains about auth, run:  wandb login
# ══════════════════════════════════════════════════════════════

import argparse
import re
import sys

import wandb

DEF_ENTITY = "bradrscott4-university-of-cape-town"
DEF_PROJECT = "honours-rl-go"


def parse_name(name):
    """Extract structured fields from a run name like
    'feudal-9x9-vs-corner-phase2-med-f1' or 'ppo-13x13-vs-greedy'."""
    n = name.lower()
    meta = {}

    if "gnugo" in n:
        meta["agent"] = "gnugo"
    elif "feudal" in n:
        meta["agent"] = "feudal"
    elif "ppo" in n:
        meta["agent"] = "ppo"
    else:
        meta["agent"] = "other"

    if "13x13" in n:
        meta["board"] = "13x13"
    elif "9x9" in n:
        meta["board"] = "9x9"
    else:
        meta["board"] = "unknown"

    if meta["agent"] == "gnugo":
        meta["phase"] = "baseline"
    elif "phase2" in n:
        meta["phase"] = "phase2"
    else:
        meta["phase"] = "phase1"

    mo = re.search(r"vs-([a-z]+)", n)          # opponent = token after 'vs-'
    meta["opponent"] = mo.group(1) if mo else "unknown"

    if meta["phase"] == "phase2":
        mm = re.search(r"phase2-(low|med|high)", n)
        if mm:
            meta["magnitude"] = mm.group(1)
        mf = re.search(r"-(f[123])(?:-|\b|$)", n)
        if mf:
            meta["frequency"] = mf.group(1)

    ms = re.search(r"-s(\d+)(?:\b|$)", n)      # seed suffix; absent => seed 0
    meta["seed"] = ms.group(1) if ms else "0"

    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=DEF_ENTITY)
    ap.add_argument("--project", default=DEF_PROJECT)
    ap.add_argument("--apply", action="store_true",
                    help="actually write tags/config (default: dry-run preview)")
    args = ap.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")
    print(f"Found {len(runs)} runs in {args.entity}/{args.project}"
          f"  ({'APPLYING' if args.apply else 'DRY-RUN'})\n")

    counts = {}
    for run in runs:
        meta = parse_name(run.name)
        new_tags = {meta["agent"], meta["board"], meta["phase"],
                    f"opp-{meta['opponent']}"}
        if meta.get("magnitude"):
            new_tags.add(f"mag-{meta['magnitude']}")
        if meta.get("frequency"):
            new_tags.add(meta["frequency"])
        if meta["phase"] == "phase2":
            new_tags.add(f"seed-{meta['seed']}")

        key = f"{meta['board']}/{meta['phase']}/{meta['agent']}"
        counts[key] = counts.get(key, 0) + 1

        print(f"  {run.name:45s} -> {sorted(new_tags)}")

        if args.apply:
            run.tags = sorted(set(run.tags) | new_tags)      # merge, never delete
            for k in ("agent", "board", "phase", "opponent",
                      "magnitude", "frequency", "seed"):
                if meta.get(k):
                    run.config[f"meta_{k}"] = meta[k]
            try:                                             # native grouping (best-effort)
                run.group = f"{meta['board']}-{meta['phase']}"
            except Exception:
                pass
            run.update()

    print("\n=== group summary (board / phase / agent) ===")
    for k in sorted(counts):
        print(f"  {k:28s} {counts[k]}")
    if not args.apply:
        print("\n(DRY-RUN — nothing written. Re-run with --apply to commit.)")


if __name__ == "__main__":
    main()
