#!/bin/bash
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name="P1PPO9"
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# Phase 1 (baseline training): PPO vs the 5 fixed opponents, 9x9
# One array task per opponent. BOARD_SIZE=9, KOMI=5.5, 3M steps on the UCT ada CPU partition 
# sbatch run_phase1_ppo_9x9_uct.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export OMP_NUM_THREADS=8
export BOARD_SIZE=9
export KOMI=5.5
export TOTAL_TIMESTEPS=3000000

OPPONENTS=(greedy defensive corner edge random)
export OPPONENT=${OPPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "Phase-1 PPO 9x9 task ${SLURM_ARRAY_TASK_ID} -> OPPONENT=${OPPONENT}"
python ppo/ppo_go.py
