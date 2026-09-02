#!/bin/bash
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --job-name="P2FEUc"
#SBATCH --array=0-8
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── PHASE 2 (RQ3): FEUDAL abrupt-shift & recovery — UCT CPU (ada), 9x9 ──
# Resumes from the 9x9 Phase-1 finals (corner/greedy/defensive) and applies an
# opponent shift schedule, measuring recovery (W=100). normal QOS has NO caps,
# so submit all 9 at once. Mirrors the PPO Phase-2 script for a fair comparison.
#   sbatch run_phase2_feudal_uct_cpu.sh
# Index map: idx = magnitude*3 + frequency
#   0 low-f1 1 low-f2 2 low-f3 | 3 med-f1 4 med-f2 5 med-f3 | 6 high-f1 7 high-f2 8 high-f3

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export OMP_NUM_THREADS=8
export BOARD_SIZE=9
export KOMI=5.5

IDX=$SLURM_ARRAY_TASK_ID
MAGS=(low med high)
# Magnitude axis calibrated EMPIRICALLY by observed disruption (dip depth +
# recovery time), NOT by a-priori strategic distance. All three are shifts
# BETWEEN MASTERABLE bots, so no capability confound (why aggressive was dropped):
#   LOW  = corner->edge       tiny  (both positional, near-identical strategy)
#   MED  = corner->defensive  moderate (recovers near the W=100 floor)
#   HIGH = greedy->defensive  deep, slow recovery (~thousands of games)
# NOTE: earlier runs tagged greedy->defensive as "med" and corner->defensive as
# "high"; the labels were SWAPPED here to match measured severity. analyze_phase2
# maps the old wandb/dir tags accordingly.
AS=(corner corner greedy)         # pre-shift opponent A (checkpoint source)
BS=(edge defensive defensive)     # post-shift opponent B
FREQS=(f1 f2 f3)

M=$((IDX / 3)); F=$((IDX % 3))
A=${AS[$M]}; B=${BS[$M]}

case ${FREQS[$F]} in
  f1) SCHED="${B}@300000,${A}@1800000" ;;
  f2) SCHED="${B}@300000,${A}@800000,${B}@1300000,${A}@1800000" ;;
  f3) SCHED="${B}@300000,${A}@500000,${B}@700000,${A}@900000,${B}@1100000,${A}@1300000,${B}@1500000,${A}@1700000,${B}@1900000,${A}@2100000" ;;
esac

export OPPONENT=$A
export RESUME_FROM=./models/feudal/${BOARD_SIZE}x${BOARD_SIZE}/${A}/feudal_go_final.pt
export SHIFT_SCHEDULE=$SCHED
export TOTAL_TIMESTEPS=2300000
# Seed-aware tag: seed 0 keeps the original tag (backward compatible with the
# existing seed-0 runs); seeds 1+ append -s<N> so the checkpoint folder AND the
# wandb run name are distinct per seed (no collision). Launch a seed with e.g.
#   sbatch --export=ALL,SEED=1 run_phase2_feudal_uct_cpu.sh
export SEED="${SEED:-0}"
SEEDSUF=""; [ "$SEED" != "0" ] && SEEDSUF="-s${SEED}"
export RUN_TAG=phase2-${MAGS[$M]}-${FREQS[$F]}${SEEDSUF}

echo "Task ${IDX}: ${RUN_TAG}  A=${A} B=${B}"
echo "  RESUME_FROM=${RESUME_FROM}"
echo "  SHIFT_SCHEDULE=${SHIFT_SCHEDULE}"
python feudal/feudalAgent.py
