#!/bin/bash
#SBATCH --job-name=aggregate
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/work/smoekar1/logs/%x_%j.out
#SBATCH --error=/work/smoekar1/logs/%x_%j.err


source /home/smoekar1/miniforge3/etc/profile.d/conda.sh
conda activate thesis

cd /work/smoekar1/Bachelorarbeit/functions

experiment=$1

python run_experiments.py --experiment $experiment