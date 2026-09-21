# Phase-2 results

Phase 2 measures recovery after an abrupt opponent strategy shift. Each run
resumes a Phase-1 checkpoint (a stable baseline against the specific opponent), then
switches the opponent one or more times mid-training and records how the agent
recovers. This folder holds the raw per-game logs and the analysed summaries.

## Folder layout

- `feudal/` and `ppo_go/` are the two agents. Under each is a board size
  (`9x9`, `13x13`), and under that one folder per run, each containing a single
  `games.csv` (the per-game log for that run).
- `summary/`** is the analysis output. 

### Run-folder naming

A run folder is named `<A>-phase2-<magnitude>-<frequency>[-s<seed>]`, e.g.
`corner-phase2-low-f1` or `greedy-phase2-high-f2-s2`.

- `<A>` — the pre-shift opponent the run resumes from (the Phase-1 checkpoint).
- `<magnitude>` — how far the shift moves: `low` = corner to edge, `med` = corner to defensive, `high` = greedy→defensive.
- `<frequency>` — how often the opponent switches: `f1` single (2 switches), `f2` periodic (4), `f3` frequent (10).
- `-s<seed>` — random seed. `-s1`, `-s2` are the other two seeds. Three seeds per condition.

Conditions per agent per board = 3 magnitudes × 3 frequencies = 9, × 3 seeds = 27 runs; × 2 agents × 2 boards = 108 runs total.

## `games.csv` — raw per-game log (one row per game)
The columns with their meanings below:

`game_idx` - Sequential game number in the run. 
`global_step` - Environment steps progressed at that game. 
`opponent` - The fixed opponent faced in that game (changes at each shift). 
`win` - Game outcome for the trained agent: 1 = win, 0 = loss. 
`rolling` - Rolling win rate over the last 100 games (0–1). The recovery signal. 
`shift_idx` - Which shift-schedule segment the game is in: 0 before the first shift, 1 after the first switch, 2 after the second 
`games_since_shift` - Games played since the last opponent switch (resets to 0 at each shift). 

All three summary files below are computed from the raw per-game data in
`games.csv` — the `win` outcomes and the `rolling` win rate they produce. Every recovery metric comes from that rolling signal measured against the pre-shift baseline (its value just before a shift): recovery time is how long the rolling win rate stays below the 0.8×baseline band, dip depth is how far it falls below baseline at its lowest point (`baseline − min_rolling`) and adaptation cost is the area under that dip — the shortfall below baseline summed over the whole post-shift segment (reported in games and in thousands of steps).

## `recovery_table.csv` — per-shift recovery (one row per run × shift)

Recovery is recomputed here straight from `games.csv`. One row per opponent switch within each run.

The columns with their meanings below:

`agent`, `board`, `magnitude`, `frequency`, `seed` - Identify the run. 
`label` - `<magnitude>-<frequency>` (e.g. `low-f1`). 
`condition` - The run-folder name. 
`shift_idx` - Which shift within the run (0-based). 
`to_opponent` - The opponent switched to at this shift. 
`returns_to_A` - 1 if this shift returns to the pre-shift opponent A, else 0. 
`shift_at_step` - Environment step at which the shift happened. 
`baseline` - Rolling win rate on the last game before the shift. 
`threshold80` - The recovery band, `0.8 × baseline`. 
`min_rolling` - Lowest rolling win rate reached after the shift. 
`perf_drop` - Dip depth = `baseline − min_rolling`. 
`end_rolling` - Rolling win rate at the end of the segment. 
`disrupted` - 1 if the rolling win rate fell below `threshold80` at any point. 
`recovered` - 1 if it returned above `threshold80` before the segment ended. 
`recovery_games` - Recovery time: games spent below the band (first drop → first return). 0 if it never dipped below; a censored lower bound (first drop → segment end) if it dipped but never returned. 
`adapt_cost_games`  Adaptation cost: area under the dip, in win-rate × games. 
`adapt_cost_ksteps`  Same area in win-rate × thousand env-steps (length-independent). 
`segment_games`  Number of games in this post-shift segment. 

## `summary_by_condition_by_seed.csv` — per seed (one row per condition × seed)

Each seed's own metrics, aggregated over that run's shifts, before averaging.

The columns with their meanings below:


`board`, `agent`, `magnitude`, `frequency`, `seed`  Identify the condition + seed. 
`n_shifts`  Number of shifts in the run. 
`n_disrupted`  How many of those shifts breached the band. 
`recovery_games`  Mean recovery time over the run's shifts (non-disrupted shifts count as 0). 
`mean_dip`  Mean dip depth over the shifts. 
`worst_dip`  Deepest dip in the run. 
`adapt_cost_ksteps`  Mean adaptation cost (win-rate × ksteps) over the shifts. 

## `summary_by_condition.csv` — merged over seeds (one row per condition)

Each metric is collapsed per seed first, then averaged
over the (up to 3) seeds with a standard error.

The columns with their meanings below:


`board`, `agent`, `magnitude`, `frequency` Identify the condition. 
`n_seeds` Seeds averaged (normally 3). 
`n_shifts` Total shifts pooled across seeds. 
`n_disrupted` Total disrupted shifts pooled across seeds. 
`recovery_mean`, `recovery_se` Mean ± SE of recovery time across seeds (0-filled: non-disrupted shifts count as 0).
`dip_mean`, `dip_se` Mean ± SE of dip depth across seeds. 
`worst_dip` Deepest dip seen in any seed. 
`adapt_cost_mean`, `adapt_cost_se` Mean ± SE of adaptation cost (ksteps) across seeds. 
`n_recovered` Disrupted shifts (pooled) that returned above the band. |
`n_censored` Disrupted shifts (pooled) that never returned before the segment ended (recovery time is a lower bound). 
`mean_recovery_disrupted` Mean recovery time over disrupted shifts only (unlike `recovery_mean`, which 0-fills the undisrupted ones). 
`mean_dip` Alias of `dip_mean`. 
`mean_adapt_cost_ksteps` Alias of `adapt_cost_mean`.

