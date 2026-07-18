#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=14:00:00
#SBATCH --job-name="FEU_Go_arr"
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── Feudal vs the 5 strategic opponents, one array task each ──────
# Mirrors run_ppo_go_array.sh. 5 tasks = MaxSubmitPU (5), so submit this
# only once the PPO runs have freed your slots (nothing else queued).
# "random" is the 6th opponent — run it in a later batch.
# OPPONENT env var picks the opponent; config_feudal.py reads it and writes
# to a per-opponent models/feudal/<opp>/ + logs/ dir + wandb run.
#   Submit with:  sbatch run_feudal_array.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProject

OPPONENTS=(aggressive greedy defensive corner edge random)
export OPPONENT=${OPPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID} -> OPPONENT=${OPPONENT}"
python feudal/feudalAgent.py
