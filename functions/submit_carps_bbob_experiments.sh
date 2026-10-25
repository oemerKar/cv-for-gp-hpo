#!/bin/bash
#SBATCH --job-name=carps_bo_bbob_experiments
#SBATCH --array=0-59                     # 3 Optimizer x 4 BBOB-Tasks x 5 Seeds - 1 = 59
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=long
#SBATCH --output=/work/smoekar1/Bachelorarbeit/functions/logs/slurm_%A_%a.out
#SBATCH --error=/work/smoekar1/Bachelorarbeit/functions/logs/slurm_%A_%a.err

set -e

cd /work/smoekar1/Bachelorarbeit/functions
mkdir -p logs

# activates the carps bbob environment (ioh-comaptible but without yahpo/hpobench)
source /home/smoekar1/miniforge3/etc/profile.d/conda.sh
conda activate carps-bbob-env

echo "Array-Task $SLURM_ARRAY_TASK_ID on $(hostname), Start: $(date)"

python run_carps_experiments.py --array-index "$SLURM_ARRAY_TASK_ID" --category bbob

echo "Done: $(date)"