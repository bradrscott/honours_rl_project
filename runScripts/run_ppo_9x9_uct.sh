#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=08:00:00
#SBATCH --job-name="PPO9_uct"
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── PPO 9x9 Phase-1 re-run on UCT (CHPC down till Fri) — strategic opponents ──
# Regenerates the 9x9 PPO finals (3M steps, KOMI 5.5) so Phase 2 can resume from
# them while CHPC is unavailable. QOS MaxSubmitPU=5 -> array 0-4 (5 tasks); run
# "random" (index 5) in a 2nd batch (sbatch --array=5 ...) once a slot frees.
# GPU cap = 2 -> ~3 waves; PPO 9x9 is fast (~1-2h/run) so all finish overnight.
# wandb runs ONLINE (UCT nodes have internet) -> live; they appear alongside the
# CHPC 9x9 PPO runs with the same names — tell them apart by date/run-id.
#   Submit from /scratch/sctbra008/HonoursProjectV2:  sbatch run_ppo_9x9_uct.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export BOARD_SIZE=9
export KOMI=5.5
export TOTAL_TIMESTEPS=3000000

OPPONENTS=(aggressive greedy defensive corner edge random)
export OPPONENT=${OPPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "UCT PPO 9x9 task ${SLURM_ARRAY_TASK_ID} -> OPPONENT=${OPPONENT}"
python ppo/ppo_go.py
