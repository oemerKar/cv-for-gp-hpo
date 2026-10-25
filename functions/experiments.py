from loops import bo_loop_loocv, bo_loop_mll, bo_loop_mcmc, bo_loop_ea
import torch
import random
import numpy as np
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
import gpytorch
from torch.distributions import MultivariateNormal
from scipy.stats import wilcoxon
from scipy.optimize import minimize
from itertools import combinations
import pandas as pd
import copy
from joblib import Parallel, delayed
import warnings
from botorch.exceptions.warnings import OptimizationWarning
import pickle, os
from plots import save_regret_with_std
import matplotlib.pyplot as plt


def setSeed(seed:int):
  """
  set all random seeds to given seed for reproducibility
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def generateInitialData(bounds, n=10):
  """
  generate n inital data points in the given bounds
  """
  d = bounds.shape[-1]
  X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, d)
  return X

# (deprecated as run_experiments.py is how the experiments were done finally)
def bo_experiment(f, n_runs = 10, num_loop = 100, kernel = None, bo_loops = None):
    """
        Starts the seperate BO_loops of each given method on the given function with the given kernel
        f: function to optimize
        n_runs: amount of seeds per optimization run per method
        num_loop: amount of loops in one seed
        kernel: kernel used, None corresponds to the standard BOTorch Matérn 2.5 kernel
        bo_loop: pairs of name of the hpo method and the method using it in a loop (in loops.py)
            if None is given only MLL and LOOCV are done
    """
    #Set bounds of f
    bounds = f.bounds
    #set the dimension
    d = bounds.shape[-1]
    default_kernel = get_covar_module_with_dim_scaled_prior(ard_num_dims= d, use_rbf_kernel= False)

    if bo_loops is None:
        bo_loops = [("MLL", bo_loop_mll), ("LOOCV", bo_loop_loocv)]

    all_regrets = {name: [] for name, _ in bo_loops}
    all_fit_times = {name: [] for name, _ in bo_loops}

    for seed in range(n_runs):
        setSeed(seed)
        X_init = generateInitialData(f.bounds, n =10)

        print(f"Run No. {seed}")

        for name, loop in bo_loops:
            setSeed(seed)
            k = copy.deepcopy(kernel) if kernel is not None else copy.deepcopy(default_kernel)
            results = loop(f =f, X_init =X_init, maximize=False, num_loop= num_loop, kernel = k)
            all_regrets[name].append(results["regrets"])
            all_fit_times[name].append(results["fit_times"])
    
    regrets = {name: np.array(regrets) for name, regrets in all_regrets.items()}
    fit_times = {name: np.array(time) for name, time in all_fit_times.items()}

    wilcoxon_comparison_all(regrets)
    return {
        "regrets": regrets,
        "fit_times": fit_times,
    }

def save_experiment(results, name, base_dir = "."):
    """
    Saves the given results, with the given name in the given base directory
    """
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok= True)

    with open(os.path.join(path, "results.pkl"),"wb") as f:
        pickle.dump(results, f)

    save_regret_with_std(
        results["regrets"],
        title = name,
        save_path= os.path.join(path, "plot_regret.pdf")
    )

class BBOBWrapper:
    """
    Wraps a BBOB wrapper, so that it can be used with the loops in loops.py
    per adding bounds, optimal_value and making it callable with a batch of points instead of only one point

    __getstate__ and __setstate__ are neeeded as the ioh problem itself (self.f) cannot be pickled (for parallel BO loops)
    so it is removed before pickling and reconstructed via fid, instance and dmimension
    """
    def __init__(self, problem, fid, instance, dimension):
        self.fid = fid
        self.instance = instance
        self.dimension = dimension
        self.f = problem
        lb = torch.tensor(self.f.bounds.lb)
        ub = torch.tensor(self.f.bounds.ub)
        self.bounds = torch.stack([lb, ub])
        self.optimal_value = self.f.optimum.y
    
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        results = []
        for x in X:
            val = self.f(x.numpy().tolist())
            results.append(val)
        return torch.tensor(results, dtype=torch.double)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["f"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from ioh import get_problem, ProblemClass
        self.f = get_problem(
            fid=self.fid,
            instance=self.instance,
            dimension=self.dimension,
            problem_class=ProblemClass.BBOB,
        )
    
def get_bbob_wrapper(fid, instance, dimension, seed):
    """
    gets BBOB Wrapper for chosen BBOB function (per fid), instance, and dimensionality
    """

    #bbob funktioniert seit auslagern nicht mehr stattdessen mit ioh was gleich funktioniert
    #f = bbob.get_bbob_problem(fid = fid, instance = instance, dimension = dimension, seed = seed)
    #obj = f[1]
    from ioh import get_problem, ProblemClass
    problem = get_problem(
        fid = fid,
        instance = instance,
        dimension =dimension,
        problem_class = ProblemClass.BBOB,
    )
    return BBOBWrapper(problem, fid, instance, dimension)


class GPFunction:
    """
    Function with controlled misspecification, sampled from a GP Prior with given kernel, per RFF (Random Fourier Features).
    Samples random Fourier Frequencies for given kernel that is sampled from (RBF and Matérn Kernels are supported)
    Optimum of the function is defined by searching grid-based and optimizing locally with L-BFGS
    """
    def __init__(
        self,
        d: int,
        bounds: torch.Tensor,
        kernel: gpytorch.kernels.Kernel,
        n_features: int = 500,
        maximize: bool = False,
        seed: int | None = None,
    ):

        if seed is not None: 
            setSeed(seed)
        
        self.d = d
        self.bounds = bounds
        self.kernel = kernel    
        self.maximize = maximize

        lb, ub = bounds[0], bounds[1]

        lengthscales = kernel.lengthscale.detach().squeeze()
        if lengthscales.dim() == 0: #wenn 1 Lengthscale
            lengthscales = lengthscales.expand(d)

        #RFF (Random Fourier Features)

        #different cases for the different kernels, as each one has different spectral density
        #RBF Kernel
        if isinstance(kernel, gpytorch.kernels.RBFKernel):
             self.w = torch.randn(n_features,d,dtype= torch.double) /lengthscales.double()
        #MaternKernel
        elif isinstance(kernel, gpytorch.kernels.MaternKernel):
            nu = kernel.nu
            v = 2*nu
            z = torch.randn(n_features, d, dtype=torch.double) / lengthscales.double()
            u = torch.distributions.Chi2(torch.tensor(v, dtype=torch.double)).sample((n_features,))
            self.w = z * torch.sqrt(v/u).unsqueeze(-1)
        else:
            raise NotImplementedError(f"RFF Sampling not implemented for {type(kernel)}")

        self.b = torch.rand(n_features, 1, dtype= torch.double) * 2 * torch.pi
        self.theta = torch.randn(n_features, 1, dtype= torch.double)
        self.scale = torch.sqrt(torch.tensor(2.0 / n_features, dtype=torch.double))

        #find optimum per grid
        X_dense = torch.rand(10000, d, dtype=torch.double) * (ub - lb) + lb
        with torch.no_grad():
            f_dense = self(X_dense)

        # Besten Punkt als Startpunkt für lokale Optimierung
        best_idx = f_dense.argmin() if not maximize else f_dense.argmax()
        x0 = X_dense[best_idx].numpy()

        def obj(x):
            with torch.no_grad():
                return self(torch.tensor(x, dtype=torch.double).unsqueeze(0)).item()

        result = minimize(
            obj if not maximize else lambda x: -obj(x),
            x0,
            bounds=[(lb[i].item(), ub[i].item()) for i in range(d)],
            method="L-BFGS-B",
        )

        self.optimal_value = result.fun if not maximize else -result.fun

    def __call__(self, X):
        with torch.no_grad():
            features = torch.cos(X @ self.w.T +self.b.T)
            return (features @ self.theta).squeeze(-1) * self.scale


def make_gp_function(
    d: int = 5,
    bounds: torch.Tensor | None = None,
    lengthscale: float = 0.3,
    true_kernel: gpytorch.kernels.Kernel | None = None,
    n_features: int = 500,
    maximize: bool = False,
    seed: int | None = None,
) -> GPFunction:
    """
    Makes Gp function for given dimensionality, true_kernel, bounds and n_features
    """
    if bounds is None:
        bounds = torch.tensor([[-1.0] * d, [1.0] * d], dtype=torch.double)

    if true_kernel is None:
        true_kernel = gpytorch.kernels.RBFKernel().double()
    if hasattr(true_kernel, 'lengthscale'):
        true_kernel.lengthscale = lengthscale

    return GPFunction(
        d=d,
        bounds=bounds,
        kernel=true_kernel,
        n_features= n_features,
        maximize=maximize,
        seed=seed,
    )
    

def wilcoxon_comparison_all(results: dict[str, dict[str,list]]):
    """
    print Wilcoxon comparison done on all method pairs in given results
    """
    methods = list(results.keys())
    pairs = list(combinations(methods,2))
    n_comparisons = len(pairs)

    for method_a, method_b in pairs:
        final_a = results[method_a][:,-1]
        final_b = results[method_b][:,-1]

        stat, p_value = wilcoxon(final_a, final_b)
        p_corrected = min(p_value * n_comparisons, 1)

        print(f"{method_a} vs {method_b}:")
        print(f"  Median {method_a}: {np.median(final_a):.4f}")
        print(f"  Median {method_b}: {np.median(final_b):.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  p-corrected: {p_corrected:.4f}")
        print(f"  Signifikant: {'Ja' if p_corrected < 0.05 else 'Nein'}")

#baut zusammenfassung der experimentdaten als tabelle mzum vergleich
def build_summary_table(experiments)-> pd.DataFrame:
    """
    Builds a summary table with all comparisons between two methods: 
    For each comparison (Experiment, Method A, Method B) a Bonferroni corrected
    Wilcoxon Test is done on the final regret (log10 clipped to epsilon)
    and Fit time ratio aswell as Median of Lengthscale and noise is printed
    
    """
    rows = []

    for experiment_name, results in experiments:
        regrets = results["regrets"]
        fit_times = results["fit_times"]
        lengthscales = results.get("lengthscales", {})
        noises = results.get("noises", {})

        methods = list(regrets.keys())
        pairs = list(combinations(regrets,2))
        n_comparisons = len(pairs)

        #fit time durchschnitt von mll berechnen
        mll_time = np.median(fit_times["MLL"]) if "MLL" in fit_times else 1.0
        fit_time_ratios = {
            name: round(np.median(fit_times[name]) / mll_time, 2)
            for name in methods
            if name in fit_times
        }

        final_lengthscale = {
            name: float(np.median(lengthscales[name][:,-1]))
            for name in methods if name in lengthscales
        }

        final_noise = {
            name: float(np.median(noises[name][:,-1]))
            for name in methods if name in noises
        }

        for method_a, method_b in pairs:
            final_a = regrets[method_a][:,-1]
            final_b = regrets[method_b][:,-1]

            stat, p_value = wilcoxon(final_a, final_b)
            p_corrected = min(p_value * n_comparisons, 1)

            #Clip regret to epsilon, as regret for GP sampled functions could be negative
            epsilon = 1e-10
            log_a = np.log10(np.clip(final_a, a_min=epsilon, a_max=None))
            log_b = np.log10(np.clip(final_b, a_min=epsilon, a_max=None))

            median_a = round(np.median(log_a), 4)
            median_b = round(np.median(log_b), 4)
            std_a = round(np.std(log_a), 4)
            std_b = round(np.std(log_b), 4)

            # Fit time ratio is based on MLL fit time
            #Wilcoxon is used without Log as it doesnt change anything
            rows.append({
                "Experiment": experiment_name,
                "Method A": method_a,
                "Log 10 Median A": median_a,
                "Log 10 Std A": std_a,
                "Fit Time Ratio A": fit_time_ratios.get(method_a, "-"),
                "Final Lengthscale A": final_lengthscale.get(method_a, "-"),
                "Final Noise A": final_noise.get(method_a, "-"),
                "Method B": method_b,
                "Log 10 Median B": median_b,
                "Log 10 Std B": std_b,
                "Fit Time Ratio B": fit_time_ratios.get(method_b, "-"),
                "Final Lengthscale B": final_lengthscale.get(method_b, "-"),
                "Final Noise B": final_noise.get(method_b, "-"),
                "p-value": round(p_corrected, 4),
                "Signifikant?": "Ja" if p_corrected < 0.05 else "Nein",
            })
    
    return pd.DataFrame(rows)

def run_single_seed(f, seed, num_loop, kernel, default_kernel, bo_loops, noiseless):
    """
    Runs all methods on a given function and with a given kernel for one seed
    Is used for parallel bo_experiment_parallel (deprecated)
    """
    torch.set_num_threads(1)
    setSeed(seed)
    warnings.filterwarnings("ignore", category= OptimizationWarning)
    X_init = generateInitialData(f.bounds, n =20)

    print(f"Run No. {seed}")
    seed_results = {}
    for name, loop in bo_loops:
        setSeed(seed)
        kwargs = {"f": f, "X_init": X_init, "maximize": False, "num_loop": num_loop, "noiseless": noiseless}
        kwargs["kernel"] = copy.deepcopy(kernel) if kernel is not None else copy.deepcopy(default_kernel)
        results = loop(**kwargs)
        seed_results[name] = {
            "regrets": results["regrets"],
            "fit_times": results["fit_times"],
        }
    
    return seed_results

#Same as bo_experiment but can be used in parallel (deprecated as run_experiments.py is how the experiments were done finally)
def bo_experiment_parallel(f, n_runs=10, num_loop=100, kernel=None, bo_loops=None, n_jobs=8, noiseless = False):
    """
        Starts the seperate BO_loops of each given method on the given function with the given kernel
        f: function to optimize
        n_runs: amount of seeds per optimization run per method
        num_loop: amount of loops in one seed
        kernel: kernel used, None corresponds to the standard BOTorch Matérn 2.5 kernel
        bo_loop: pairs of name of the hpo method and the method using it in a loop (in loops.py)
            if None is given only MLL and LOOCV are done
        n_jobs: amount of parallel seeds (not parallel for MCMC)
        noiseless: whether to include noise or not
    """
    #Set bounds of f
    bounds = f.bounds
    #set the dimension
    d = bounds.shape[-1]
    default_kernel = get_covar_module_with_dim_scaled_prior(ard_num_dims= d, use_rbf_kernel= False)

    if bo_loops is None:
        bo_loops = [("MLL", bo_loop_mll), ("LOOCV", bo_loop_loocv)]

    all_regrets = {name: [] for name, _ in bo_loops}
    all_fit_times = {name: [] for name, _ in bo_loops}

    # MCMC raussortieren — läuft nie parallel wegen Pyro global state
    mcmc_loops = [(name, loop) for name, loop in bo_loops if "mcmc" in name.lower()]
    other_loops = [(name, loop) for name, loop in bo_loops if "mcmc" not in name.lower()]

    #Parallel für alle non mcmc loops
    if other_loops:
        all_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_single_seed)(f, seed, num_loop, kernel, default_kernel, other_loops, noiseless)
            for seed in range(n_runs)
        )
        for seed_result in all_results:
            for name, metrics in seed_result.items():
                all_regrets[name].append(metrics["regrets"])
                all_fit_times[name].append(metrics["fit_times"])

    #Sequenziell für mcmc loop
    if mcmc_loops:
        for seed in range(n_runs):
            seed_result = run_single_seed(f, seed, num_loop, kernel, default_kernel, mcmc_loops, noiseless)
            for name, metrics in seed_result.items():
                all_regrets[name].append(metrics["regrets"])
                all_fit_times[name].append(metrics["fit_times"])
            print(f"MCMC seed {seed+1}/{n_runs} done")


    regrets = {name: np.array(regrets) for name, regrets in all_regrets.items()}
    fit_times = {name: np.array(time) for name, time in all_fit_times.items()}

    wilcoxon_comparison_all(regrets)
    return {
        "regrets": regrets,
        "fit_times": fit_times,
    }