#!/bin/bash
#SBATCH --job-name=carps_bo_experiments
#SBATCH --array=0-209                    # 3 Optimizer x 14 Tasks x 5 Seeds - 1 = 209
#SBATCH --time=12:00:00                 
#SBATCH --mem=32G                        
#SBATCH --cpus-per-task=4
#SBATCH --partition=long
#SBATCH --output=/work/smoekar1/Bachelorarbeit/functions/logs/slurm_%A_%a.out    # %A = Job-ID, %a = Array-Index
#SBATCH --error=/work/smoekar1/Bachelorarbeit/functions/logs/slurm_%A_%a.err
 
set -e
cd /work/smoekar1/Bachelorarbeit/functions
mkdir -p logs
 
# Aktivates the normal carps-Environment (see setup_carps_env.sh)
# Not for BBOB Carps exp as they are not compatible with the YAHPO and HBOBench ones
source /home/smoekar1/miniforge3/etc/profile.d/conda.sh
conda activate carps-env
 
echo "Array-Task $SLURM_ARRAY_TASK_ID on $(hostname), Start: $(date)"
 
python run_carps_experiments.py --array-index "$SLURM_ARRAY_TASK_ID"
 
echo "Done: $(date)"
 
