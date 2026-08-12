#!/bin/bash
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --job-name="P2FEU13"
#SBATCH --array=0-8
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── PHASE 2 (RQ3): FEUDAL abrupt-shift & recovery — UCT CPU (ada), 13x13 ─
# Mirrors run_phase2_ppo_13x13_uct_cpu.sh EXACTLY (same 9 conditions, schedules,
# 2.3M budget, magnitudes) for a fair PPO-vs-feudal comparison on 13x13.
# BOARD_SIZE=13, KOMI=7.5; resumes from the 13x13 feudal Phase-1 finals.
# Feudal 13x13 on CPU is slow -> 30h walltime.  sbatch run_phase2_feudal_13x13_uct_cpu.sh
# Index map: idx = magnitude*3 + frequency.

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export OMP_NUM_THREADS=8
export BOARD_SIZE=13
export KOMI=7.5

IDX=$SLURM_ARRAY_TASK_ID
MAGS=(low med high)
# Empirically-calibrated magnitudes (all shifts BETWEEN masterable bots):
#   LOW  = corner->edge   MED = corner->defensive   HIGH = greedy->defensive
AS=(corner corner greedy)         # pre-shift opponent A (checkpoint source)
BS=(edge defensive defensive)     # post-shift opponent B
FREQS=(f1 f2 f3)

M=$((IDX / 3)); F=$((IDX % 3))
A=${AS[$M]}; B=${BS[$M]}

# Budget + schedule SCALED UP ~1.7x from the 9x9 Phase-2 (2.3M -> 4.0M), matching
# the Phase-1 13x13/9x9 ratio (5M/3M): 13x13 learns slower, so recovery segments
# are stretched proportionally to give recovery a fair chance (else censored).
# IDENTICAL to run_phase2_ppo_13x13_uct_cpu.sh so the comparison stays fair.
case ${FREQS[$F]} in
  f1) SCHED="${B}@500000,${A}@3200000" ;;
  f2) SCHED="${B}@500000,${A}@1400000,${B}@2300000,${A}@3200000" ;;
  f3) SCHED="${B}@500000,${A}@850000,${B}@1200000,${A}@1550000,${B}@1900000,${A}@2250000,${B}@2600000,${A}@2950000,${B}@3300000,${A}@3650000" ;;
esac

export OPPONENT=$A
export RESUME_FROM=./models/feudal/${BOARD_SIZE}x${BOARD_SIZE}/${A}/feudal_go_final.pt
export SHIFT_SCHEDULE=$SCHED
export TOTAL_TIMESTEPS=4000000
# Seed-aware tag: seed 0 keeps the original tag (backward compatible); seeds 1+
# append -s<N> so folder + wandb name are distinct per seed. Launch with e.g.
#   sbatch --export=ALL,SEED=1 run_phase2_feudal_13x13_uct_cpu.sh
export SEED="${SEED:-0}"
SEEDSUF=""; [ "$SEED" != "0" ] && SEEDSUF="-s${SEED}"
export RUN_TAG=phase2-${MAGS[$M]}-${FREQS[$F]}${SEEDSUF}

echo "Task ${IDX}: ${RUN_TAG}  A=${A} B=${B}  (13x13)"
echo "  RESUME_FROM=${RESUME_FROM}"
echo "  SHIFT_SCHEDULE=${SHIFT_SCHEDULE}"
python feudal/feudalAgent.py
