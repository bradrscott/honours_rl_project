#!/bin/sh
#SBATCH --account=l40sugrd
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1 --ntasks=2
#SBATCH --time=08:00:00
#SBATCH --job-name="FEUDAL_Go"
#SBATCH --mail-user=sctbra008@myuct.ac.za
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate rl_project
cd /scratch/sctbra008/HonoursProject
python feudalAgent.py
