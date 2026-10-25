#!/bin/bash

cd "$(dirname "$0")"

source /home/smoekar1/miniforge3/etc/profile.d/conda.sh
conda activate thesis
#Goes through all experiments, starts them parallely and aggregates them after all seeds are done
for exp in experiment_1 experiment_2_v1 experiment_2_v2 experiment_2_v3 experiment_3 experiment_4 experiment_init_sensitivity experiment_low_dim experiment_more_iters experiment_mcmc_more_res experiment_mcmc_more_res_high_dim; do
    # 10 Seeds parallel
    if [ "$exp" = "experiment_3" ]; then
        job=$(sbatch --array=0-19 --parsable run_job.sh $exp)
    else
        job=$(sbatch --array=0-9 --parsable run_job.sh $exp)
    fi
    
    #Aggregation when all seeds done
    sbatch --dependency=afterok:$job aggregate_job.sh $exp
    echo "Submitted $exp aggregation after Job $job"
done