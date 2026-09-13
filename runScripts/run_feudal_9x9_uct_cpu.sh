#!/bin/bash
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --job-name="FEU9c"
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── Feudal 9x9 Phase-1 on UCT CPU (ada partition) — all 6 opponents ───
# Same CPU rationale as the PPO script. Feudal is slower than PPO (~8-12h for
# 3M on CPU), hence the longer walltime. 3M steps, KOMI 5.5, fixed code.
# wandb ONLINE. Submit from /scratch/sctbra008/HonoursProjectV2:
#   sbatch run_feudal_9x9_uct_cpu.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export OMP_NUM_THREADS=8
export BOARD_SIZE=9
export KOMI=5.5
export TOTAL_TIMESTEPS=3000000

OPPONENTS=(greedy defensive corner edge random)
export OPPONENT=${OPPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "UCT-CPU feudal 9x9 task ${SLURM_ARRAY_TASK_ID} -> OPPONENT=${OPPONENT}"
python feudal/feudalAgent.py
