#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=08:00:00
#SBATCH --job-name="P1PPO13"
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# Phase 1 (baseline training): PPO vs the 5 fixed opponents, 13x13 
# One array task per opponent. BOARD_SIZE=13, KOMI=7.5, 5M steps on the l40s GPU partition
# sbatch run_phase1_ppo_13x13_uct.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export BOARD_SIZE=13
export KOMI=7.5
export TOTAL_TIMESTEPS=5000000

OPPONENTS=(greedy defensive corner edge random)
export OPPONENT=${OPPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "Phase-1 PPO 13x13 task ${SLURM_ARRAY_TASK_ID} -> OPPONENT=${OPPONENT}"
python ppo/ppo_go.py
