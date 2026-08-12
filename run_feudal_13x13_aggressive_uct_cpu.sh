#!/bin/bash
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --job-name="FEU13ag"
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

# ── Feudal 13x13 vs AGGRESSIVE — Phase-1, UCT CPU (ada) ───────────────
# Single job (not an array): one opponent, aggressive. 13x13, KOMI 7.5, 5M
# steps — same config CHPC used (48h walltime; feudal 13x13/5M ran ~35h there).
# wandb ONLINE (UCT nodes have internet). Fixed feudal code. Saves to
# models/feudal/13x13/aggressive/ (board in path; no collision with the
# existing 13x13 aggressive-seed1 dir). Submit from HonoursProjectV2:
#   sbatch run_feudal_13x13_aggressive_uct_cpu.sh
# If ada rejects 48h (check: sinfo -p ada -o "%l"), lower --time; the run
# checkpoints every 200k steps so it can be resumed with RESUME_FROM.

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProjectV2

export OMP_NUM_THREADS=8
export BOARD_SIZE=13
export KOMI=7.5
export TOTAL_TIMESTEPS=5000000
export OPPONENT=aggressive
# Seed 0 collapsed into early-passing vs aggressive (ep_len -> ~5, degenerate
# local optimum: agent passes -> empty board -> aggressive also passes -> game
# ends at -1). A restart MUST use a different seed or it reproduces the same
# collapse. Bump this each retry (1, 2, 3...). aggressive is the highest-variance
# matchup — some seeds escape the trap, some don't.
export SEED=1

echo "UCT-CPU feudal 13x13 -> OPPONENT=${OPPONENT}  KOMI=${KOMI}  STEPS=${TOTAL_TIMESTEPS}"
python feudal/feudalAgent.py
