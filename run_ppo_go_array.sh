#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=08:00:00
#SBATCH --job-name="PPO_Go_arr"
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── PPO vs the 5 strategic opponents, one array task each ─────────
# 5 tasks = MaxSubmitPU (5), so this must be your ONLY submitted jobs
# (cancel any standalone PPO job first). "random" is the 6th opponent —
# run it in a later batch once a slot frees.
# The OPPONENT env var picks the opponent; config_ppo_go.py reads it and
# writes to a per-opponent models/ + logs/ dir + wandb run. No %throttle
# (MaxJobsPU unlimited) so they run as GPUs free.
#   Submit with:  sbatch run_ppo_go_array.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProject

OPPONENTS=(aggressive greedy defensive corner edge random)
export OPPONENT=${OPPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "Array task ${SLURM_ARRAY_TASK_ID} -> OPPONENT=${OPPONENT}"
python ppo_go.py
