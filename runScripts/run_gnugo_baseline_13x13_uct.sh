#!/bin/bash
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name="GNUGO13"
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# GNU Go external-reference baseline, 13x13 
# GNU Go (black) vs each heuristic opponent (white), go_v5
# One array task per opponent -> one wandb run each
# GNU Go does NOT learn, so a few hundred games is a complete baseline
# GNU Go is CPU-only. Prereq (one-time): conda install -c conda-forge gnugo
# sbatch run_gnugo_baseline_13x13_uct.sh

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

OPPONENTS=(greedy defensive corner edge random)
export OPPONENT=${OPPONENTS[$SLURM_ARRAY_TASK_ID]}

echo "GNU Go baseline (13x13) vs ${OPPONENT}"
python gnugo/run_baseline.py --opponent "${OPPONENT}" --games 200 --level 10 \
       --board-size 13 --komi 7.5
