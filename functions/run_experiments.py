#Import needed libraries
import torch
import random
import numpy as np
import gpytorch
import matplotlib.pyplot as plt
import os

import argparse
import copy
import json
import gzip


torch.set_default_dtype(torch.double)

import warnings
from gpytorch.utils.warnings import NumericalWarning
from botorch.exceptions.warnings import OptimizationWarning, InputDataWarning

#Ignore warnings that are not problematic atp for now

warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)

#Import Benchmarking Functions
from botorch.test_functions import Branin
from botorch.test_functions import Ackley
from botorch.test_functions import Hartmann

#import kernels
from gpytorch.kernels import LinearKernel, PeriodicKernel, MaternKernel, RBFKernel

#Import the BO loops

from loops import bo_loop_mll
from loops import bo_loop_loocv
from loops import bo_loop_ea
from loops import bo_loop_mcmc

#import the plot functions

from plots import plot_bo_comparison, plot_regret_with_std, save_regret_with_std

#import the functions needed for experiments
from experiments import setSeed, generateInitialData, bo_experiment, bo_experiment_parallel, save_experiment

#Function to use bbob benchmarking
from experiments import get_bbob_wrapper, BBOBWrapper

#Controlled kernel misspecification
from experiments import GPFunction, make_gp_function

#Table Funciton
from experiments import build_summary_table

from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.models import SingleTaskGP

experiments = []

bbob_suite = [
    (1,  "Sphere"),
    (2,  "Ellipsoid_Separable"),
    (3,  "Rastrigin_Separable"),
    (4,  "Skew Rastrigin-Bueche"),
    (5,  "Linear Slope"),
    (6,  "Attractive_Sector"),
    (7,  "Step_Ellipsoid"),
    (8,  "Rosenbrock"),
    (9,  "Rosenbrock rotated"),
    (10, "Ellipsoid"),
    (11, "Discus"),
    (12, "Bent_Cigar"),
    (13, "Sharp_Ridge"),
    (14, "Sum_of_different_Powers"),
    (15, "Rastrigin"),
    (16, "Weierstrass"),
    (17, "Schaffer_F7_Cond10"),
    (18, "Schaffer_F7_Cond1000"),
    (19, "Griewank_Rosenbrock"),
    (20, "Schwefel"),
    (21, "Gallagher_101"),
    (23, "Katsuura"),
    (24, "Lunacek_biRastrigin"),
]

def experiment_one(seed):
    """
    Experiment 1: Baseline experiment
    """
    for fid, name in bbob_suite:
        f = get_bbob_wrapper(fid=fid, instance=1, dimension=5,seed = 1)
        results = run(
            f=f, num_loop=50,
            seed = seed,
            bo_loops=[
                ("MLL",   bo_loop_mll),
                ("LOOCV", bo_loop_loocv),
                ("MCMC",  bo_loop_mcmc),
            ],
            save_path= f"experiments/experiment_1/{name}/seed_{seed}.json.gz",
            experiment_name= "Experiment_1",
        )

def experiment_two_v1(seed):
    """
    Experiment 2: Misspecification on the BBOB Functions:
    Version 1: without ARD without Priors
    """
    misspec_configs = [
        ("Matérn 2.5", lambda: MaternKernel(nu=2.5)),
        ("Matérn 0.5", lambda: MaternKernel(nu=0.5)),
        ("Periodic",   lambda: PeriodicKernel()),
    ]

    for fid, name in bbob_suite:
        f = get_bbob_wrapper(fid=fid, instance=1, dimension=5, seed = 1)
        for misspec_name, misspec_kernel in misspec_configs:
            results = run(
                f=f, num_loop=50,
                seed = seed,
                kernel=misspec_kernel(),
                bo_loops=[
                    ("MLL",   bo_loop_mll),
                    ("LOOCV", bo_loop_loocv),
                    ("MCMC", bo_loop_mcmc),
                ],
                save_path =f"experiments/experiment_2_v1/{name}_{misspec_name}/seed_{seed}.json.gz",
                experiment_name= "Experiment_2_v1",
                variant = "v1",
            )

def experiment_two_v2(seed):
    """
    Experiment 2: Misspecification on the BBOB Functions:
    Version 2: with ARD without Priors 
    """
    misspec_configs = [
        ("Matérn 2.5", lambda: MaternKernel(nu=2.5, ard_num_dims = 5)),
        ("Matérn 0.5", lambda: MaternKernel(nu=0.5, ard_num_dims = 5)),
        ("Periodic",   lambda: PeriodicKernel(ard_num_dims = 5)),
    ]

    for fid, name in bbob_suite:
        f = get_bbob_wrapper(fid=fid, instance=1, dimension=5, seed = 1)
        for misspec_name, misspec_kernel in misspec_configs:
            results = run(
                f=f, num_loop=50,
                seed = seed,
                kernel=misspec_kernel(),
                bo_loops=[
                    ("MLL",   bo_loop_mll),
                    ("LOOCV", bo_loop_loocv),
                    ("MCMC", bo_loop_mcmc),
                ],
                save_path= f"experiments/experiment_2_v2/{name}_{misspec_name}/seed_{seed}.json.gz",
                experiment_name= "Experiment_2_v2",
                variant = "v2",
            )

def experiment_two_v3(seed):
    """
    Experiment 2: Misspecification on the BBOB Functions:
    Version 3: with ARD with Priors
    """
    from math import sqrt, log
    from gpytorch.priors import LogNormalPrior
    from gpytorch.constraints import GreaterThan

    #keeping the hyperprior
    def make_misspecified_kernel(d, nu=0.5):
        lengthscale_prior = LogNormalPrior(
            loc=sqrt(2) + log(d) * 0.5,
            scale=sqrt(3)
        )
        
        return MaternKernel(
            nu=nu,
            ard_num_dims=d,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                0.025,
                transform=None,
                initial_value=lengthscale_prior.mode,
            ),
        )

    misspec_configs = [
        ("Matérn 2.5", lambda: make_misspecified_kernel(nu=2.5, d = 5)),
        ("Matérn 0.5", lambda: make_misspecified_kernel(nu=0.5, d = 5)),
    ]

    for fid, name in bbob_suite:
        f = get_bbob_wrapper(fid=fid, instance=1, dimension=5, seed = 1)
        for misspec_name, misspec_kernel in misspec_configs:
            results = run(
                f=f, num_loop=50,
                seed = seed,
                kernel=misspec_kernel(),
                bo_loops=[
                    ("MLL",   bo_loop_mll),
                    ("LOOCV", bo_loop_loocv),
                    ("MCMC", bo_loop_mcmc),
                ],
                save_path= f"experiments/experiment_2_v3/{name}_{misspec_name}/seed_{seed}.json.gz",
                experiment_name= "Experiment_2_v3",
                variant = "v3",
            )
        

def experiment_three(seed):
    """
    Experiment 3: GP-Sampled Misspecification:
    """
    #keep the hyperprior (only Periodic kernel does not have one same as before)
    from math import sqrt, log
    from gpytorch.priors import LogNormalPrior
    from gpytorch.constraints import GreaterThan
    def make_misspecified_kernel(kernel_class, d = 5, nu= None):
        lengthscale_prior = LogNormalPrior(
            loc=sqrt(2) + log(d) * 0.5,
            scale=sqrt(3)
        )

        kwargs = dict(
            ard_num_dims = d,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                0.025, transform=None, initial_value=lengthscale_prior.mode,
            ),
        )

        if nu is not None:
            kwargs["nu"] = nu
        return kernel_class(**kwargs)
        

    #Boolean at the end says whether MCMC can be used for this config
    gp_configs = [
        ("RBF→Matérn0.5",       lambda: RBFKernel(ard_num_dims= 5).double(),            lambda: make_misspecified_kernel(MaternKernel, nu = 0.5), True),
        ("RBF→Matérn2.5",       lambda: RBFKernel(ard_num_dims=5).double(),             lambda: make_misspecified_kernel(MaternKernel, nu = 2.5), True),
        ("Matérn2.5→RBF",       lambda: MaternKernel(nu=2.5,ard_num_dims = 5).double(), lambda: make_misspecified_kernel(RBFKernel),              True),
        ("Matérn2.5→Matérn0.5", lambda: MaternKernel(nu=2.5, ard_num_dims=5).double(),  lambda: make_misspecified_kernel(MaternKernel, nu = 0.5), True),
        ("Matérn0.5→RBF",       lambda: MaternKernel(nu=0.5, ard_num_dims=5).double(),  lambda: make_misspecified_kernel(RBFKernel),              True),
        ("RBF→Periodic",        lambda: RBFKernel(ard_num_dims= 5).double(),            lambda: PeriodicKernel(ard_num_dims= 5),                  False),
        ("Matérn0.5→Matérn2.5", lambda: MaternKernel(nu=0.5, ard_num_dims=5).double(),  lambda: make_misspecified_kernel(MaternKernel, nu=2.5),   True),
        ("Matérn2.5→Matérn2.5 (control)", lambda: MaternKernel(nu=2.5, ard_num_dims=5).double(), lambda: make_misspecified_kernel(MaternKernel, nu=2.5), True),
        ("Matérn0.5→Matérn0.5 (control)", lambda: MaternKernel(nu=0.5, ard_num_dims=5).double(), lambda: make_misspecified_kernel(MaternKernel, nu=0.5), True),
    ]
    for config_name, true_kernel, misspec_kernel, include_mcmc in gp_configs:
        kernel = true_kernel()
        kernel.lengthscale = 0.3

        f = make_gp_function(d=5, true_kernel = kernel, seed=seed)

        bo_loops = [
            ("MLL",   bo_loop_mll),
            ("LOOCV", bo_loop_loocv),
        ]
        if include_mcmc:
            bo_loops = [
            ("MLL",   bo_loop_mll),
            ("LOOCV", bo_loop_loocv),
            ("MCMC", bo_loop_mcmc),
        ]

        results = run(
            f=f, num_loop=50,
            seed = seed,
            kernel=misspec_kernel(),
            bo_loops= bo_loops,
            save_path= f"experiments/experiment_3/{config_name}/seed_{seed}.json.gz",
            experiment_name= "Experiment_3",
        )


def experiment_four(seed):
    """
    Experiment 4: Effect of higher dimensional data
    """
    bbob_suite_high_dim = [
        (1,  "Sphere"),
        (8,  "Rosenbrock"),
        (10, "Ellipsoid"),
        (15, "Rastrigin"),
        (21, "Gallagher"),
    ]

    for d in [10,20]:
        for fid, name in bbob_suite_high_dim:
                f = get_bbob_wrapper(fid=fid, instance=1, dimension=d,seed = 1)
                results = run(
                    f=f, num_loop=50,
                    seed = seed,
                    bo_loops=[
                        ("MLL",   bo_loop_mll),
                        ("LOOCV", bo_loop_loocv),
                        ("MCMC",  bo_loop_mcmc),
                    ],
                    save_path= f"experiments/experiment_4/d{d}/{name}/seed_{seed}.json.gz",
                    experiment_name= "Experiment_4",
                )
def experiment_init_sensitivity(seed):
    """
    Test important results on wheter they are replicable with different initial points, by doing the same experiments 
    with different seeds to generate different initial data. This is evaluated and test_significance and test_sensitivity 
    are done on this data aswell to ensure maximum significance in the notebook
    """
    functions = [
        (1, "Sphere"),
        (14, "Sum_of_different_Powers"),
        (15,"Rastrigin"),
        (10,"Ellipsoid"),
    ]

    n_init_values = [5,10,20]
    bo_loops = [("MLL", bo_loop_mll), ("LOOCV", bo_loop_loocv),("MCMC", bo_loop_mcmc)]

    for fid, name in functions:
        f = get_bbob_wrapper(fid= fid, instance = 1, dimension = 5, seed = 1)
        for n_init in n_init_values:
            save_path = f"experiments/experiment_init_sensitivity/{name}/n_init_{n_init}/seed_{seed}.json.gz"
            run(
                f=f, seed=seed, num_loop=50, bo_loops=bo_loops,
                save_path=save_path,
                experiment_name=f"Experiment_Init_Sensitivity",
                n_init=n_init,
            )

def experiment_low_dim(seed):
    """
    Test the hypotheses of LOOCV being bad at low dim
    """
    #12 and 20 can be directly compared to the carps tasks
    bbob_suite_low_dim = [
        (1,  "Sphere"),
        (12, "Bent_Cigar"),       
        (20, "Schwefel"),   
        (15, "Rastrigin"),
        (10, "Ellipsoid"),
    ]

    for fid, name in bbob_suite_low_dim:
        f = get_bbob_wrapper(fid = fid, instance = 1, dimension = 2, seed = 1)
        results = run(
            f=f, num_loop= 50,
            seed=seed,
            bo_loops=[
                ("MLL",   bo_loop_mll),
                ("LOOCV", bo_loop_loocv),
                ("MCMC",  bo_loop_mcmc),
            ],
            save_path=f"experiments/experiment_low_dim/{name}/seed_{seed}.json.gz",
            experiment_name="Experiment_Low_Dim",
        )

def experiment_baseline_mcmc_more_resources(seed):
    """
    Test whether more MCMC resources drastically change the results
    """

    bbob_suite_more_res = [
        (1,  "Sphere"),
        (13, "Sharp Ridge"),       
        (7, "Step_Ellipsoid"),   
        (24, "Lunacek_biRastrigin"),
        (14, "Sum_of_diferent_Powers"),
    ]

     
    for fid, name in bbob_suite_more_res:
        f = get_bbob_wrapper(fid=fid, instance=1, dimension=5,seed = 1)
        results = run_mcmc_more_resources(
            f=f, num_loop=50,
            seed = seed,
            bo_loops=[
                ("MLL",   bo_loop_mll),
                ("LOOCV", bo_loop_loocv),
                ("MCMC",  bo_loop_mcmc),
            ],
            save_path= f"experiments/experiment_mcmc_more_res/{name}/seed_{seed}.json.gz",
            experiment_name= "Experiment_MCMC_More_Res",
        )

def experiment_mcmc_more_resources_high_dim(seed):
    """
    Test whether more MCMC resources drastically change the results for the high dimensional functions
    """
    bbob_suite_more_res = [
        (1,  "Sphere"),
        (13, "Sharp Ridge"),       
        (7, "Step_Ellipsoid"),   
        (24, "Lunacek_biRastrigin"),
        (14, "Sum_of_diferent_Powers"),
    ]

    bbob_suite_high_dim = [
        (1,  "Sphere"),
        (8,  "Rosenbrock"),
        (10, "Ellipsoid"),
        (15, "Rastrigin"),
        (21, "Gallagher"),
    ]

    for d in [10, 20]:
        for fid, name in bbob_suite_high_dim:
            f = get_bbob_wrapper(fid=fid, instance=1, dimension=d,seed = 1)
            results = run_mcmc_more_resources(
                f=f, num_loop=50,
                seed = seed,
                bo_loops=[
                    ("MLL",   bo_loop_mll),
                    ("LOOCV", bo_loop_loocv),
                    ("MCMC",  bo_loop_mcmc),
                ],
                save_path= f"experiments/experiment_mcmc_more_res_high_dim/{name}_dim{d}/seed_{seed}.json.gz",
                experiment_name= "Experiment_MCMC_More_Res_High_Dim",
            )



def experiment_more_iterations(seed):
    """
    Test the effect of higher iteration count on the results
    """
    bbob_suite_high_dim = [
        (1,  "Sphere"),
        (8,  "Rosenbrock"),
        (10, "Ellipsoid"),
        (15, "Rastrigin"),
        (21, "Gallagher"),
    ]

    for l in [100,200]:
        for fid, name in bbob_suite_high_dim:
                f = get_bbob_wrapper(fid=fid, instance=1, dimension=5,seed = 1)
                results = run(
                    f=f, num_loop=l,
                    seed = seed,
                    bo_loops=[
                        ("MLL",   bo_loop_mll),
                        ("LOOCV", bo_loop_loocv),
                        ("MCMC",  bo_loop_mcmc),
                    ],
                    save_path= f"experiments/experiment_more_iters/loop_count{l}/{name}/seed_{seed}.json.gz",
                    experiment_name= "Experiment_More_Iterations",
                )

def run_mcmc_more_resources(f, seed, num_loop = 100, kernel = None, bo_loops = None, save_path = None, experiment_name = "", variant = "v3", n_init : int = 10, noiseless = False):
    """
    The same run function but with higher MCMC resources to test its effect on the results

    noiseless parameter is only for MCMC as the other functions are not noiseless by default
    """

    #Set bounds of f
    bounds = f.bounds
    #set the dimension
    d = bounds.shape[-1]
    default_kernel = get_covar_module_with_dim_scaled_prior(ard_num_dims= d, use_rbf_kernel= False)

    if bo_loops is None:
        bo_loops = [("MLL", bo_loop_mll), ("LOOCV", bo_loop_loocv)]

    setSeed(seed)
    X_init = generateInitialData(f.bounds, n = n_init)

    seed_result = {"experiment": experiment_name,
                   "seed": seed,
                   "methods": {}
                }
    for name, loop in bo_loops:
        setSeed(seed)
        k = copy.deepcopy(kernel) if kernel is not None else copy.deepcopy(default_kernel)
        if name == "MCMC":
            result = loop(f =f, X_init =X_init, maximize=False, num_loop= num_loop, kernel = k, variant = variant, warmup_steps = 512, num_samples = 256, thinning = 16, noiseless= noiseless)
        else:
            result = loop(f =f, X_init =X_init, maximize=False, num_loop= num_loop, kernel = k, noiseless= noiseless)
        seed_result["methods"][name] = {
            "regrets":      result["regrets"],
            "fit_times":    result["fit_times"],
            "best_vals":    result["best_vals"],
            "iter_times":   result["iter_times"],
            "lengthscales": result["lengthscales"],
            "noises":       result["noises"],
        }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok= True)
        with gzip.open(save_path, "wt") as file:
            json.dump(seed_result, file, default = json_serialize)
    
    return seed_result



def run(f, seed, num_loop = 100, kernel = None, bo_loops = None, save_path = None, experiment_name = "", variant = "v3", n_init : int = 10, noiseless = False):
    """
    Runs the whole Experiment for one given seed and given variant of Experiment

    Generates the inital data, sets the kernel and the seeds then starts the run for each method and saves its results
    so that aggregate can collect these data to aggregate them later
    """

    #Set bounds of f
    bounds = f.bounds
    #set the dimension
    d = bounds.shape[-1]
    default_kernel = get_covar_module_with_dim_scaled_prior(ard_num_dims= d, use_rbf_kernel= False)

    if bo_loops is None:
        bo_loops = [("MLL", bo_loop_mll), ("LOOCV", bo_loop_loocv)]

    setSeed(seed)
    X_init = generateInitialData(f.bounds, n = n_init)

    seed_result = {"experiment": experiment_name,
                   "seed": seed,
                   "methods": {}
                }
    for name, loop in bo_loops:
        setSeed(seed)
        k = copy.deepcopy(kernel) if kernel is not None else copy.deepcopy(default_kernel)
        if name == "MCMC":
            result = loop(f =f, X_init =X_init, maximize=False, num_loop= num_loop, kernel = k, variant = variant, noiseless= noiseless)
        else:
            result = loop(f =f, X_init =X_init, maximize=False, num_loop= num_loop, kernel = k, noiseless= noiseless)
        seed_result["methods"][name] = {
            "regrets":      result["regrets"],
            "fit_times":    result["fit_times"],
            "best_vals":    result["best_vals"],
            "iter_times":   result["iter_times"],
            "lengthscales": result["lengthscales"],
            "noises":       result["noises"],
        }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok= True)
        with gzip.open(save_path, "wt") as file:
            json.dump(seed_result, file, default = json_serialize)
    
    return seed_result

def json_serialize(obj):
    """
    Serializes non-serializable attr in json
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not serializable")

def aggregate(exp_path, n_runs=10):
    """
    aggregates all seed results of the experiment at the given path
    """
    all_regrets = {}
    all_fit_times = {}
    all_lengthscales = {}
    all_noises = {}

    for seed in range(n_runs):
        path = f"{exp_path}/seed_{seed}.json.gz"
        with gzip.open(path, "rb") as file:
            seed_result = json.load(file)
        for loop_name, metrics in seed_result["methods"].items():
            if loop_name not in all_regrets:
                all_regrets[loop_name] = []
                all_fit_times[loop_name] = []
                all_lengthscales[loop_name] = []
                all_noises[loop_name] = []

            all_regrets[loop_name].append(metrics["regrets"])
            all_fit_times[loop_name].append(metrics["fit_times"])
            all_lengthscales[loop_name].append(metrics["lengthscales"])
            all_noises[loop_name].append(metrics["noises"])

    results = {
        "regrets":   {n: np.array(r) for n, r in all_regrets.items()},
        "fit_times": {n: np.array(t) for n, t in all_fit_times.items()},
        "lengthscales": {n: np.array(t) for n, t in all_lengthscales.items()},
        "noises": {n: np.array(t) for n, t in all_noises.items()},
    }
    
    save_experiment(results, name=exp_path)
    return results


def aggregate_experiment(exp_path, n_runs=10):
    """
    Aggregates all Subexperiment in a Experiment, by iterating through the subfolders and
    saving the plots, aswell as building the summary_table.csv
    """
    experiment_results = []
    
    for subdir in sorted(os.listdir(exp_path)):
        full_path = os.path.join(exp_path, subdir)
        if not os.path.isdir(full_path):
            continue
        
        # Check whether all seeds are present
        missing = [s for s in range(n_runs) 
                   if not os.path.exists(os.path.join(full_path, f"seed_{s}.json.gz"))]
        if missing:
            print(f"Warning: {subdir} missing seeds {missing}, skipping")
            continue
        
        # Aggregate + save plot
        results = aggregate(full_path, n_runs=n_runs)
        save_experiment(results, name=full_path)
        
        # Collect for the summary table
        experiment_results.append((subdir, results))

    if experiment_results:
        table = build_summary_table(experiment_results)
        table.to_csv(os.path.join(exp_path, "summary_table.csv"), index = False)
        print(f"Global summary saved to {exp_path}/summary_table.csv")
    
    return experiment_results

def aggregate_experiment_4(base_path, n_runs=10):
    """
    Aggregates all SubFolders of Subfolders in experiment 4 (because of the d10/ and d20/ folders), 
    by iterating through the subfolders and saving the plots, aswell as building the summary_table.csv
    
    """
    experiment_results = []
    for d_dir in sorted(os.listdir(base_path)):  # d10, d20
        d_path = os.path.join(base_path, d_dir)
        if not os.path.isdir(d_path):
            continue
        
        for subdir in sorted(os.listdir(d_path)): #Experiments
            full_path = os.path.join(d_path, subdir)
            if not os.path.isdir(full_path):
                continue
            
            # Check whether all seeds are present
            missing = [s for s in range(n_runs) 
                    if not os.path.exists(os.path.join(full_path, f"seed_{s}.json.gz"))]
            if missing:
                print(f"Warning: {d_dir}/{subdir} missing seeds {missing}, skipping")
                continue
            
            # Aggregate + save plot
            results = aggregate(full_path, n_runs=n_runs)
            save_experiment(results, name=full_path)
            
            # Collect for the summary table
            experiment_results.append((f"{d_dir}/{subdir}", results))

    if experiment_results:
        table = build_summary_table(experiment_results)
        table.to_csv(os.path.join(base_path, "summary_table.csv"), index = False)
        print(f"Global summary saved to {base_path}/summary_table.csv")
    
    return experiment_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    runners = {
        "experiment_1":   experiment_one,
        "experiment_2_v1": experiment_two_v1,
        "experiment_2_v2": experiment_two_v2,
        "experiment_2_v3": experiment_two_v3,
        "experiment_3":   experiment_three,
        "experiment_4": experiment_four,
        "experiment_init_sensitivity": experiment_init_sensitivity,
        "experiment_low_dim": experiment_low_dim,
        "experiment_mcmc_more_res": experiment_baseline_mcmc_more_resources,
        "experiment_more_iters": experiment_more_iterations,
        "experiment_mcmc_more_res_high_dim": experiment_mcmc_more_resources_high_dim,
    }

    if args.seed is not None:
        # Singular seed for SLURM Array Jobs
        runners[args.experiment](args.seed)
    else:
        # Aggregation — after all Seeds are done
        if args.experiment in ("experiment_4","experiment_more_iters", "experiment_init_sensitivity"):
            aggregate_experiment_4(f"experiments/{args.experiment}")
        elif args.experiment == "experiment_3":
            aggregate_experiment(f"experiments/{args.experiment}", n_runs= 20)
        else:
            aggregate_experiment(f"experiments/{args.experiment}")