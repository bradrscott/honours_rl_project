#!/bin/bash
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --job-name="P2PPO9"
#SBATCH --array=0-8
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# Phase 2: PPO abrupt-shift & recovery — UCT CPU (ada), 9x9 
# 9 conditions = 3 magnitudes x 3 frequencies (array 0-8)
# Each resumes the 9x9 Phase-1 PPO checkpoint of opponent A and applies a shift schedule, measuring recovery (W=100)
# sbatch run_phase2_ppo_9x9_uct.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export OMP_NUM_THREADS=8
export BOARD_SIZE=9
export KOMI=5.5

IDX=$SLURM_ARRAY_TASK_ID
MAGS=(low med high)
AS=(corner corner greedy)         
BS=(edge defensive defensive)     
FREQS=(f1 f2 f3)

M=$((IDX / 3)); F=$((IDX % 3))
A=${AS[$M]}; B=${BS[$M]}

case ${FREQS[$F]} in
  f1) SCHED="${B}@300000,${A}@1800000" ;;
  f2) SCHED="${B}@300000,${A}@800000,${B}@1300000,${A}@1800000" ;;
  f3) SCHED="${B}@300000,${A}@500000,${B}@700000,${A}@900000,${B}@1100000,${A}@1300000,${B}@1500000,${A}@1700000,${B}@1900000,${A}@2100000" ;;
esac

export OPPONENT=$A
export RESUME_FROM=./phase1_checkpoints/ppo_go/${BOARD_SIZE}x${BOARD_SIZE}/${A}/ppo_go_final.pt
export SHIFT_SCHEDULE=$SCHED
export TOTAL_TIMESTEPS=2300000

export SEED="${SEED:-0}"
SEEDSUF=""; [ "$SEED" != "0" ] && SEEDSUF="-s${SEED}"
export RUN_TAG=phase2-${MAGS[$M]}-${FREQS[$F]}${SEEDSUF}

echo "Task ${IDX}: ${RUN_TAG}  A=${A} B=${B}  (9x9)"
echo "  RESUME_FROM=${RESUME_FROM}"
echo "  SHIFT_SCHEDULE=${SHIFT_SCHEDULE}"
python ppo/ppo_go.py
