#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=08:00:00
#SBATCH --job-name="P2_PPO"
#SBATCH --array=0-8
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── PHASE 2 (RQ3): PPO abrupt-shift & recovery runs ───────────────
# 9 conditions = 3 magnitudes x 3 frequencies. Each resumes the Phase-1
# PPO checkpoint of opponent A and applies the shift schedule.
#   Magnitudes:  LOW  corner->edge | MED  corner->defensive | HIGH greedy->defensive
#   Frequencies: f1 single shift  | f2 every 500k          | f3 every 200k
# Every schedule ends with A reintroduced (policy-reuse probe).
#
# QOS allows only 5 submitted jobs — submit in WAVES:
#   sbatch --array=0-4 run_phase2_ppo_array.sh
#   sbatch --array=5-8 run_phase2_ppo_array.sh   (once slots free)
#
# Index map: idx = magnitude*3 + frequency
#   0 low-f1   1 low-f2   2 low-f3
#   3 med-f1   4 med-f2   5 med-f3
#   6 high-f1  7 high-f2  8 high-f3

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProject

IDX=$SLURM_ARRAY_TASK_ID
MAGS=(low med high)
AS=(corner corner greedy)      # pre-shift opponent A (checkpoint source)
BS=(edge defensive defensive)    # post-shift opponent B
FREQS=(f1 f2 f3)

M=$((IDX / 3))
F=$((IDX % 3))
A=${AS[$M]}
B=${BS[$M]}

case ${FREQS[$F]} in
  f1) SCHED="${B}@300000,${A}@1800000" ;;
  f2) SCHED="${B}@300000,${A}@800000,${B}@1300000,${A}@1800000" ;;
  f3) SCHED="${B}@300000,${A}@500000,${B}@700000,${A}@900000,${B}@1100000,${A}@1300000,${B}@1500000,${A}@1700000,${B}@1900000,${A}@2100000" ;;
esac

export OPPONENT=$A
export RESUME_FROM=./models/ppo_go/${BOARD_SIZE:-13}x${BOARD_SIZE:-13}/${A}/ppo_go_final.pt
export SHIFT_SCHEDULE=$SCHED
export TOTAL_TIMESTEPS=2300000
export RUN_TAG=phase2-${MAGS[$M]}-${FREQS[$F]}

echo "Task ${IDX}: ${RUN_TAG}  A=${A} B=${B}"
echo "  RESUME_FROM=${RESUME_FROM}"
echo "  SHIFT_SCHEDULE=${SHIFT_SCHEDULE}"

python ppo/ppo_go.py
