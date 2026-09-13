#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=12:00:00
#SBATCH --job-name="GNUGO_base"
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── GNU Go baseline: one wandb run per opponent (gnugo-13x13-vs-<opp>) ──
# GNU Go (black) vs each heuristic (white), go_v5 refereed. Overlays the
# agents' 5M-step charts as a flat reference line (games mapped across the
# 0..5M axis). GNU Go does NOT learn, so a few hundred games is a complete
# baseline — NOT 5M literal plies (that would take weeks at level 10).
#
# GNU Go is CPU-ONLY: the requested GPU is UNUSED (header reused so it runs).
# Prereq (one-time): conda install -c conda-forge gnugo
# Smoke test first: OPPONENT=greedy python gnugo/run_baseline.py --games 1 --verify
#
# 5-job QOS limit -> waves:
#   sbatch --array=0-4 run_gnugo_baseline.sh   (greedy defensive corner edge random)

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProject

OPPONENTS=(greedy defensive corner edge random)
export OPPONENT=${OPPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "GNU Go baseline vs ${OPPONENT}"
python gnugo/run_baseline.py --opponent "${OPPONENT}" --games 200 --level 10 \
       --board-size 13 --komi 7.5
