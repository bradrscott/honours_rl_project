#!/bin/bash
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name="PPO9c"
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── PPO 9x9 Phase-1 on UCT CPU (ada partition) — all 6 opponents ──────
# CPU avoids the congested l40s GPU queue; these runs are env-bound anyway so
# GPU barely helps. 3M steps, KOMI 5.5. ada has idle nodes -> starts fast.
# wandb ONLINE (UCT nodes have internet) -> live; shares names with the CHPC
# 9x9 PPO runs, tell apart by date.  Submit from /scratch/sctbra008/HonoursProjectV2:
#   sbatch run_ppo_9x9_uct_cpu.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export OMP_NUM_THREADS=8
export BOARD_SIZE=9
export KOMI=5.5
export TOTAL_TIMESTEPS=3000000

OPPONENTS=(aggressive greedy defensive corner edge random)
export OPPONENT=${OPPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "UCT-CPU PPO 9x9 task ${SLURM_ARRAY_TASK_ID} -> OPPONENT=${OPPONENT}"
python ppo/ppo_go.py
