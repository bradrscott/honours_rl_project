# paperMaterials

Figures, tables and pictures for the paper. Build artifacts live here; the
generator scripts read from the repo (`opponents/`, the trained checkpoints,
`phase2_results/`).

## Section 3.1 — Environment & Opponents

| File | Paper element | What it is |
|---|---|---|
| `fig_boards.png` | `fig:boards` | 9×9 vs 13×13 boards side by side (go_v5 render) |
| `fig_obs_encoding.png` | obs encoding (appendix) | schematic of the 17 feature planes |
| `fig_opponents_5panel.png` | `fig:opponents` | one short-game board per opponent style |
| `fig_opponent_heatmaps.png` | opponent move-density | where each bot plays, over 40 games each |
| `tab_opponents.tex` | `tab:opponents` | opponent name / focus / key behaviour |

Regenerate all four figures:

```bash
/opt/miniconda3/envs/rl_project/bin/python paperMaterials/make_env_opponent_figs.py
```

Opponent set shown = the paper's five: greedy, defensive, corner, edge, random
(aggressive is excluded from Phase 2 for the capability confound; it can be
added to the figures by editing `OPPS` in the script).

## Dropping them into the paper (LaTeX)

```latex
% --- fig:boards ---
\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figs/fig_boards.png}
  \caption{The two board sizes used, rendered in PettingZoo \texttt{go\_v5}.}
  \label{fig:boards}
\end{figure}

% --- fig:opponents ---
\begin{figure*}[t]\centering
  \includegraphics[width=\textwidth]{figs/fig_opponents_5panel.png}
  \caption{A representative early position for each fixed opponent.}
  \label{fig:opponents}
\end{figure*}

% --- opponent move-density heatmaps ---
\begin{figure*}[t]\centering
  \includegraphics[width=\textwidth]{figs/fig_opponent_heatmaps.png}
  \caption{Move-density heatmaps (9×9, 40 games each): Corner concentrates
    top-left and Edge on the perimeter, while Greedy, Defensive and Random are
    spatially diffuse --- their identity is strategic, not positional.}
  \label{fig:opponent-heatmaps}
\end{figure*}

% --- observation encoding (appendix) ---
\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figs/fig_obs_encoding.png}
  \caption{The go\_v5 observation: 8 time-steps of (opponent, own) stone planes
    plus a colour-to-play plane, 17 in total.}
  \label{fig:obs-encoding}
\end{figure}

% --- opponent table ---
\input{tab_opponents.tex}
```

(Upload the PNGs to a `figs/` folder in the Overleaf project, or adjust the
paths above to wherever you keep images.)

## Section 3.2 — Agents

| File | Paper element | What it is |
|---|---|---|
| `fig_agents.png` | agent architectures | flat PPO vs Feudal (FuN), side by side, shared backbone |
| `tab_hyperparams.tex` | `tab:hyperparams` | shared + PPO-specific + FuN-specific hyperparameters |
| `tab_paramcount.tex` | `tab:paramcount` | parameter counts (PPO vs FuN, both boards) |

Regenerate the figure + print the parameter counts:

```bash
/opt/miniconda3/envs/rl_project/bin/python paperMaterials/make_agent_figs.py
```

**Important:** the networks are **not** matched-capacity — FuN is 26× (9×9) to
47× (13×13) larger, dominated by the Worker `LSTMCell(256, k·|A|)`. Frame the
param table as "FuN has far more capacity yet still does not recover faster",
not "matched capacity".

```latex
% --- agent architectures ---
\begin{figure*}[t]\centering
  \includegraphics[width=\textwidth]{figs/fig_agents.png}
  \caption{The two agents. Both share the CNN backbone; PPO splits into a
    policy and a value head, while FuN routes it into a Manager (sets goals)
    and a Worker (acts each step toward the goal).}
  \label{fig:agents}
\end{figure*}

\input{tab_hyperparams.tex}
\input{tab_paramcount.tex}
```

## Section 3.3 — Experimental Design

| File | Paper element | What it is |
|---|---|---|
| `tab_experiments.tex` | `tab:experiments` | the full factorial design (108 runs) |
| `tab_magnitudes.tex` | `tab:magnitudes` | LOW/MED/HIGH shift definitions + disruption |
| `fig_shift_schedule.png` | shift-schedule schematic | timeline of f1/f2/f3 (9×9) |
| `tab_gnugo.tex` | `tab:gnugo` | GNU Go win rate vs each opponent (skill anchor) |
| `fig_phase1_phase2_flow.png` | Phase 1 → Phase 2 flow (appendix) | the pipeline |

Regenerate the figures:

```bash
/opt/miniconda3/envs/rl_project/bin/python paperMaterials/make_experiment_figs.py
```

```latex
\input{tab_experiments.tex}
\input{tab_magnitudes.tex}
\input{tab_gnugo.tex}

\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figs/fig_shift_schedule.png}
  \caption{Phase-2 shift schedules for the three frequencies (9×9): the agent
    resumes against A, then the opponent switches to B and back on the f1/f2/f3
    timelines shown.}
  \label{fig:shift-schedule}
\end{figure}

\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figs/fig_phase1_phase2_flow.png}
  \caption{The two-phase pipeline: train to a baseline, checkpoint, then resume
    and inject opponent shifts while measuring recovery.}
  \label{fig:pipeline}
\end{figure}
```

Note: the shift-schedule figure shows the **9×9** schedule (budget 2.3M). The
13×13 schedule is the same shape scaled to the 4M budget (first shift at 500k).

## Section 3.4 — Metrics

| File | Paper element | What it is |
|---|---|---|
| `fig_metrics_annotated.png` | annotated metric figure | one real curve defining all four measures |

Regenerate:

```bash
/opt/miniconda3/envs/rl_project/bin/python paperMaterials/make_metric_fig.py
```

```latex
\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figs/fig_metrics_annotated.png}
  \caption{The four recovery measures on one real rolling win-rate segment
    (feudal, 9×9, MED-f1): the pre-shift baseline and 0.8× band, the dip depth,
    the recovery time (games below the band), and the adaptation cost (shaded
    area between baseline and curve).}
  \label{fig:metrics}
\end{figure}
```

## Section 3.5 — Tools (appendix)

| File | Paper element | What it is |
|---|---|---|
| `tab_tools.tex` | `tab:tools` | compute + software versions + logging (reproducibility) |

Versions are as pinned in the `rl_project` conda environment
(Python 3.10, PyTorch 2.11, PettingZoo 1.25, Gymnasium 1.2, NumPy 2.2, GNU Go 3.8).

```latex
\input{tab_tools.tex}
```

## Section 4.1 — Phase-1 Capability  (in `results/`)

| File | Paper element | What it is |
|---|---|---|
| `results/tab_phase1.tex` | `tab:phase1` | win rate per agent × opponent × board |
| `results/fig_phase1_curves.png` | `fig:phase1-curves` | rolling win rate, PPO vs FuN, faceted by board |
| `results/fig_phase1_settle_bars.png` | `fig:phase1-settle` | steps (M) to reach & stay within 10% of final win rate |
| `results/fig_phase1_scatter.png` | `fig:phase1-scatter` | slope chart: 9×9 → 13×13 win rate, a line per (agent, opponent) |

Built from **W&B** (canonical = most-recent Phase-1 run per condition):

```bash
/opt/miniconda3/envs/rl_project/bin/python paperMaterials/make_phase1_results.py   # table, curves, csv
/opt/miniconda3/envs/rl_project/bin/python paperMaterials/make_phase1_settle.py    # settling-step bars
/opt/miniconda3/envs/rl_project/bin/python paperMaterials/make_phase1_scatter.py   # board-transfer scatter
```

`fig_phase1_scatter.png` is a slope chart: two vertical axes (9×9 left, 13×13
right), one line per (agent, opponent) connecting its two baseline win rates.
Downward = dropped on the larger board, upward = improved. PPO declines on every
opponent (all blue lines slope down); FuN is mixed — Edge (60→87) and Random
(64→83) climb, while Corner (88→54) collapses. Reads off the same
`phase1_win_rates.csv`. (File name kept as `fig_phase1_scatter.png` for stable
LaTeX references.)

`fig_phase1_settle_bars.png` **replaces** the old `fig_phase1_bars.png` (which
just duplicated the table). Settling step = the first training point after
which the rolling win rate stays within a `target × 0.90` band (target = that
cell's final win rate) for a sustained window of 15% of the training budget —
the standard control-theory settling time (tolerance band + hold window). This
is robust to isolated late noise dips, which the naive "last dip, then stays
forever" rule would let pin the settling step at the end of training. A run
that never holds the band for a full window is **censored** ("did not settle",
`n/s`) and drawn as a hatched bar at the budget, not a solid one. Per-board
y-axes (9×9 up to 3M, 13×13 up to 5M) keep both panels legible.

```latex
\input{results/tab_phase1.tex}

\begin{figure*}[t]\centering
  \includegraphics[width=\textwidth]{figs/fig_phase1_curves.png}
  \caption{Phase-1 training curves: rolling win rate for PPO and FuN against
    each opponent, on $9\times9$ (top) and $13\times13$ (bottom); dashed line is
    the GNU~Go reference.}
  \label{fig:phase1-curves}
\end{figure*}

\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figs/fig_phase1_settle_bars.png}
  \caption{Training steps (millions) after which each agent stays within 10\%
    of its final win rate (Table~\ref{tab:phase1}) for a sustained window;
    \texttt{n/s} marks runs that never settle within training.}
  \label{fig:phase1-settle}
\end{figure}

\begin{figure}[t]\centering
  \includegraphics[width=0.72\linewidth]{figs/fig_phase1_scatter.png}
  \caption{Per-opponent baseline win rate from $9\times9$ to $13\times13$.
    Each line joins an agent's two win rates; downward lines dropped on the
    larger board, upward lines improved.}
  \label{fig:phase1-scatter}
\end{figure}
```
