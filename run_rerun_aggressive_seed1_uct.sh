#!/bin/bash
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=14:00:00
#SBATCH --job-name="FEU_agg_s1"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── RE-RUN feudal 13x13 vs AGGRESSIVE with a NEW SEED (fixed code, UCT) ──
# The from-scratch SEED=0 aggressive run collapsed (always-pass, ep_len ~5).
# This re-samples with SEED=1 to test whether the collapse reproduces or was a
# one-off instability. RUN_TAG=seed1 + new seed => its OWN models/wandb dir, so
# the original collapsed run stays intact for comparison.
#   Submit from /scratch/sctbra008/HonoursProjectV2:  sbatch run_rerun_aggressive_seed1_uct.sh
#
# NOTE: 5M feudal may exceed the 14h walltime. It checkpoints every 200k, so if
# it's cut short, resume with RESUME_FROM pointed at the latest
# models/feudal/13x13/aggressive-seed1/feudal_go_*.pt (ask for a resume script).
# wandb runs ONLINE (UCT compute nodes have internet) -> live graphs, no sync.

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export SEED=1
export RUN_TAG=seed1
export BOARD_SIZE=13
export KOMI=7.5
export OPPONENT=aggressive
export TOTAL_TIMESTEPS=5000000

echo "UCT feudal 13x13 RE-RUN vs aggressive | SEED=${SEED} RUN_TAG=${RUN_TAG} board=${BOARD_SIZE} steps=${TOTAL_TIMESTEPS}"
python feudal/feudalAgent.py
