#!/bin/bash
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name="P2PPO13"
#SBATCH --array=0-8
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# Phase 2: PPO abrupt-shift & recovery — UCT CPU (ada), 13x13
# 9 conditions = 3 magnitudes x 3 frequencies (array 0-8)
# Resumes the 13x13 Phase-1 PPO checkpoint of opponent A and applies a shift schedule
# sbatch run_phase2_ppo_13x13_uct.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export OMP_NUM_THREADS=8
export BOARD_SIZE=13
export KOMI=7.5

IDX=$SLURM_ARRAY_TASK_ID
MAGS=(low med high)
AS=(corner corner greedy)
BS=(edge defensive defensive)
FREQS=(f1 f2 f3)

M=$((IDX / 3)); F=$((IDX % 3))
A=${AS[$M]}; B=${BS[$M]}

case ${FREQS[$F]} in
  f1) SCHED="${B}@500000,${A}@3200000" ;;
  f2) SCHED="${B}@500000,${A}@1400000,${B}@2300000,${A}@3200000" ;;
  f3) SCHED="${B}@500000,${A}@850000,${B}@1200000,${A}@1550000,${B}@1900000,${A}@2250000,${B}@2600000,${A}@2950000,${B}@3300000,${A}@3650000" ;;
esac

export OPPONENT=$A
export RESUME_FROM=./phase1_checkpoints/ppo_go/${BOARD_SIZE}x${BOARD_SIZE}/${A}/ppo_go_final.pt
export SHIFT_SCHEDULE=$SCHED
export TOTAL_TIMESTEPS=4000000

export SEED="${SEED:-0}"
SEEDSUF=""; [ "$SEED" != "0" ] && SEEDSUF="-s${SEED}"
export RUN_TAG=phase2-${MAGS[$M]}-${FREQS[$F]}${SEEDSUF}

echo "Task ${IDX}: ${RUN_TAG}  A=${A} B=${B}  (13x13)"
echo "  RESUME_FROM=${RESUME_FROM}"
echo "  SHIFT_SCHEDULE=${SHIFT_SCHEDULE}"
python ppo/ppo_go.py
