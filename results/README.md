# Results

All experiment data for the study, split into three folders. Each subfolder has
its own README with the details.

- `gnugo/` — the GNU Go external-reference baselines: one CSV per board × opponent giving GNU Go's win rate against each fixed opponent.
- `phase1/` — Phase-1 capability results: each agent's settled per-opponent win-rate baseline (the target Phase-2 recovery is measured against). See `phase1/README.md`.
- `phase2/` — Phase-2 recovery results: the per-game logs for every run plus the analysed recovery summaries. See `phase2/README.md`.

Model weights are not stored here (available on request). Raw training curves for all runs are on Weights & Biases:
<https://api.wandb.ai/links/bradrscott4-university-of-cape-town/fct92j4k>

## Opening the CSV files

These are standard comma-delimited CSVs. Any text editor reads them correctly.

If you open one in Excel on a Mac and every value lands in one column, that is
an Excel regional-separator halt, not a problem with the file. To view it as a proper table:
- Import it: open a blank workbook → `Data` tab → `Get Data (Power Query)` - `Text/CSV` (or `From Text/CSV`) - pick the file - set the delimiter to Comma - `Load`.


