#!/bin/bash
#SBATCH --job-name=bo_exp
#SBATCH --partition=long
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --output=/work/smoekar1/logs/%x_%A_%a.out
#SBATCH --error=/work/smoekar1/logs/%x_%A_%a.err

source /home/smoekar1/miniforge3/etc/profile.d/conda.sh
conda activate thesis

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

cd /work/smoekar1/Bachelorarbeit/functions

experiment=$1

python run_experiments.py --experiment $experiment --seed $SLURM_ARRAY_TASK_ID