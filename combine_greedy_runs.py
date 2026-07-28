# Stitch the two greedy 13x13 wandb runs into ONE continuous 5M run.
#   original: feudal-13x13-vs-greedy            (steps 0 -> ~4.2M, killed at wall)
#   resume:   feudal-13x13-vs-greedy-resume-to-5M (steps 0 -> 0.8M, from the 4.2M ckpt)
# We keep the original up to 4.2M, offset the resume by +4.2M, and re-log both
# into a new run so it reads as a single 0 -> 5.0M curve.
#
# Run on login1 (needs internet + WANDB_API_KEY). No pandas needed (scan_history).
#   export WANDB_API_KEY=<key>; python combine_greedy_runs.py
#
# NOTE: custom/win_rate is a rolling-200 window that RESET at the resume, so
# expect a small dip right at the 4.2M join — cosmetic; note it in the writeup.

import wandb

ENTITY  = "bradrscott4-university-of-cape-town"
PROJECT = "honours-rl-go"
ORIG_ID   = "m9pgqhso"   # feudal-13x13-vs-greedy
RESUME_ID = "yjknmhia"   # feudal-13x13-vs-greedy-resume-to-5M
OFFSET    = 4_200_000    # resume was from the feudal_go_4200000.pt checkpoint
# Skip the resume's first RESUME_SKIP steps: its rolling-200 window starts empty
# and reads artificially low (~-1) until it re-fills (~200 games ≈ 60k steps).
# Dropping this transient removes the dip at the join — the segments meet at the
# true ~0.8 level with a small flat bridge instead.
RESUME_SKIP = 80_000

# Only fetch the metrics we actually plot — far less data than every column,
# so scan_history returns much faster.
KEYS = ["custom/win_rate", "custom/total_episodes",
        "rollout/ep_rew_mean", "rollout/ep_len_mean",
        "loss/total", "loss/worker", "loss/manager",
        "loss/value_worker", "loss/value_manager",
        "worker/entropy", "worker/advantage", "worker/intrinsic_reward",
        "manager/cosines", "manager/advantage"]

api = wandb.Api()
orig = api.run(f"{ENTITY}/{PROJECT}/{ORIG_ID}")
res  = api.run(f"{ENTITY}/{PROJECT}/{RESUME_ID}")

# history(samples=N) is a SINGLE downsampled request (fast), unlike
# scan_history which pages through every row. ~4000 points is plenty for
# smooth curves over 5M steps.
print("fetching original run history (downsampled)...", flush=True)
orig_df = orig.history(samples=3000, keys=KEYS, pandas=True)
print(f"  original: {len(orig_df)} points", flush=True)

print("fetching resume run history (downsampled)...", flush=True)
res_df = res.history(samples=1000, keys=KEYS, pandas=True)
print(f"  resume: {len(res_df)} points", flush=True)

combined = []
for _, row in orig_df.iterrows():
    s = row.get("_step")
    if s is not None and s <= OFFSET:
        combined.append((int(s), row))
for _, row in res_df.iterrows():
    s = row.get("_step")
    if s is not None and s >= RESUME_SKIP:      # drop the window-refill transient
        combined.append((int(s) + OFFSET, row))

combined.sort(key=lambda x: x[0])
print(f"total {len(combined)} points; re-logging into a combined run...", flush=True)

run = wandb.init(entity=ENTITY, project=PROJECT,
                 name="feudal-13x13-vs-greedy-5M-combined")
last = -1
for step, r in combined:
    if step <= last:
        continue
    # skip private cols (_step etc) and NaN (v != v is True only for NaN)
    d = {k: float(v) for k, v in r.items()
         if not k.startswith("_") and v is not None and v == v}
    if d:
        run.log(d, step=int(step))
        last = step
run.finish()
print("done: created run 'feudal-13x13-vs-greedy-5M-combined' (0 -> 5M)")
