# Run scripts

Run scripts that reproduce every experiment in the paper for the two clusters used:

- `*_uct.sh` — SLURM scripts for the University of Cape Town cluster (submit with `sbatch <script>`).
- `*_chpc.pbs` — PBS Pro scripts for the CHPC Lengau cluster (submit with `qsub <script>`).

Each script is a job array with one task per condition, so a single submission launches the whole set.

## Before you run — change the notification email

Every script sends job notifications to the author's address. Anyone else running these (supervisor, marker, or otherwise) should change it to their own or the notifications will come to the author:

- UCT (`*_uct.sh`) — edit the `#SBATCH --mail-user=sctbra008@myuct.ac.za` line (`#SBATCH --mail-type=ALL` if fewer emails are wanted).
- CHPC (`*_chpc.pbs`) — edit the `#PBS -M sctbra008@myuct.ac.za` line(`#PBS -m abe` for which events trigger mail).

To disable notifications entirely, remove those lines (or set `#SBATCH --mail-type=NONE` / `#PBS -m n`).

## Before you run — set the paths and environment

Every path in these scripts points at the author's cluster accounts and will not exist on yours. Change them to your own locations before running:

- UCT (`*_uct.sh`) — update the `cd /scratch/sctbra008/HonoursProjectV2` line to your own repo path, and make sure the `conda activate rl_project` line names a conda environment you have actually built.
- CHPC (`*_chpc.pbs`) — update the `cd /mnt/lustre/users/bscott/HonoursProject` line, the environment lines (`module load chpc/python/anaconda/...` and `source /mnt/lustre/users/bscott/HonoursProject/rl_env/bin/activate`), and the `#PBS -o` / `#PBS -e` output paths — replace `bscott` with your CHPC username throughout and point at an environment you have built.

## Contents

- `run_phase1_ppo_9x9_uct.sh` / `run_phase1_ppo_13x13_uct.sh` PPO, 9×9 and 13×13, UCT
- `run_phase1_feudal_9x9_uct.sh` / `run_phase1_feudal_13x13_uct.sh` Feudal, 9×9 and 13×13, UCT
- `run_phase1_ppo_9x9_chpc.pbs` / `run_phase1_ppo_13x13_chpc.pbs` PPO, 9×9 and 13×13, CHPC
- `run_phase1_feudal_9x9_chpc.pbs` / `run_phase1_feudal_13x13_chpc.pbs` Feudal, 9×9 and 13×13, CHPC
- `run_phase2_ppo_9x9_uct.sh` / `run_phase2_ppo_13x13_uct.sh` PPO, 9×9 and 13×13, UCT
- `run_phase2_feudal_9x9_uct.sh` / `run_phase2_feudal_13x13_uct.sh` Feudal, 9×9 and 13×13, UCT
- `run_gnugo_baseline_9x9_uct.sh` / `run_gnugo_baseline_13x13_uct.sh` GNU Go ref, 9×9 and 13×13, UCT

Phase 1 trains each agent against the five fixed opponents (one array task each). 
Phase 2 resumes a Phase-1 checkpoint and applies an abrupt opponent shift (9 conditions = 3 magnitudes × 3 frequencies with one array task each).

## Model checkpoints are NOT in this repository

The trained weights (`phase1_checkpoints/` and `phase2_checkpoints/`) are not in this submission — they are several GB and too large for version control. This has one practical consequence for anyone re-running the scripts:

- Phase-1 scripts train from scratch They need no checkpoints and run as is, writing their finals to `phase1_checkpoints/<agent>/<board>/<opponent>/`.
- Phase-2 scripts resume a Phase-1 checkpoint Each sets
  `RESUME_FROM=./phase1_checkpoints/<agent>/<board>/<opponent>/<agent>_go_final.pt`.
  So Phase 2 cannot run until those Phase-1 finals exist.

To run Phase 2, either:
1. run the matching Phase-1 script first to regenerate the checkpoints, or
2. request the trained weights from the author (contact via the email on
   the paper) and drop them into `phase1_checkpoints/`.

The complete training and evaluation curves for all runs (GNU Go baseline, Phase 1, Phase 2) are available on Weights & Biases: <https://api.wandb.ai/links/bradrscott4-university-of-cape-town/fct92j4k>

## Configuration

All behaviour is driven by environment variables that the config files
(`ppo/config_ppo_go.py`, `feudal/config_feudal.py`) read at import:

`BOARD_SIZE`, `KOMI` - Board (9x9 / 13x13) and komi (5.5 / 7.5) respectively 
`OPPONENT` - Fixed opponent for the run (set per array task). 
`TOTAL_TIMESTEPS` - Training budget (Phase 1: 3M on 9×9, 5M on 13×13; Phase 2: 2.3M / 4.0M). 
`RESUME_FROM` - Phase-1 checkpoint a Phase-2 run resumes from. 
`SHIFT_SCHEDULE` - Phase-2 opponent-shift schedule. 
`RUN_TAG` - Phase-2 tag (e.g. `phase2-high-f1`) that names the checkpoint dir and W&B run. 
`SEED` - Random seed. Seeds 1+ append `-s<N>` to the tag so runs stay distinct. 

