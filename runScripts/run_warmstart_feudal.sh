#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=14:00:00
#SBATCH --job-name="FEU_warm_agg"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── Feudal vs aggressive, WARM-STARTED from the 13x13 defensive final ─
# The key experiment: FuN's documented strength is transfer, so entering
# aggressive as a competent player (not from scratch) is its best shot at a
# non-trivial win rate. Mirror of run_warmstart_ppo.sh for a fair comparison.
# SAVES to models/feudal/aggressive-warmstart-from-defensive/. 14h walltime.

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProject

export OPPONENT=aggressive
export RESUME_FROM=./models/feudal/${BOARD_SIZE:-13}x${BOARD_SIZE:-13}/defensive/feudal_go_final.pt
export RUN_TAG=warmstart-from-defensive

echo "Feudal warm-start vs aggressive, resume from ${RESUME_FROM}"
python feudal/feudalAgent.py
