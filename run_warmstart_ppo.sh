#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=08:00:00
#SBATCH --job-name="PPO_warm_agg"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── PPO vs aggressive, WARM-STARTED from the 13x13 defensive final ─
# Tests whether entering aggressive as a competent Go player (not from
# scratch) lets it reach a decent win rate. Run for BOTH agents (this +
# run_warmstart_feudal.sh) so the comparison stays fair.
# NOTE: RESUME_FROM points at the existing (flat-path) Phase-1 defensive
# checkpoint. This run SAVES to models/ppo_go/aggressive-warmstart-from-
# defensive/ (RUN_TAG), so it can't overwrite anything. No SHIFT_SCHEDULE
# => normal training that just starts from a checkpoint (not a Phase-2 shift).

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProject

export OPPONENT=aggressive
export RESUME_FROM=./models/ppo_go/${BOARD_SIZE:-13}x${BOARD_SIZE:-13}/defensive/ppo_go_final.pt
export RUN_TAG=warmstart-from-defensive

echo "PPO warm-start vs aggressive, resume from ${RESUME_FROM}"
python ppo/ppo_go.py
