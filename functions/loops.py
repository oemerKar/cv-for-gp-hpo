from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts

from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior

from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from sklearn.model_selection import KFold

from gpytorch.mlls import LeaveOneOutPseudoLikelihood
from torch.optim import LBFGS
import random

import time
import torch
import numpy as np
import gpytorch

from pyro.poutine.runtime import _PYRO_STACK
from botorch.exceptions.errors import ModelFittingError

import jax.numpy as jnp
from botorch.models.fully_bayesian import MaternPyroModel, compute_dists
from gpytorch.kernels import MaternKernel, Kernel, ScaleKernel

from pyromodels import get_pyro_model_for_kernel, Matern05PyroModel, RBFPyroModel
from pyromodels import get_pyro_model_for_kernel_and_variants, Matern25ConfigurablePriorModel, Matern05ConfigurablePriorModel, RBFConfigurablePriorModel
import jax
import os


def bo_loop_mll(f, X_init, maximize: bool, noiseless=False, num_loop: int = 10, kernel=None):
    """
    the BO loop for the MLL method: build GP, optimize hyperparamaters via method, 
    use acquisition function to get next point and conatenate data

    uses fit_gpytorch_mll method with the ExactMarginalLikelihood to opitmize hyperparameters

    f: the function to optimize
    X_init: initial data
    maximize: whether it is a problem where the function is to be maximized or minimized
    noiseless: wether to include noise or not (default is including noise)
    num_loop: amount of times the loop is repeated
    kernel: the kernel used, if None is given BoTorch's standard Matérn 2.5 kernel is used


    returns dict of arrays of best_vals, regrets, fit_times, iter_times, lengthscales and noises of each loop
    """
    # set f_opt to optimal value of given function
    f_opt = f.optimal_value
    f_opt = f_opt if maximize else -f_opt
    # set the bounds of the function
    bounds = f.bounds
    # set the dimension
    d = bounds.shape[-1]

    # generate initial data
    X = X_init.clone()
    y = f(X).unsqueeze(-1)
    y = y if maximize else -y

    best_vals = []
    regrets = []
    fit_times = []
    iter_times = []
    lengthscales = []
    noises = []

    for t in range(num_loop):
        iter_start = time.time()

        Yvar = torch.full_like(y, 1e-6)
        # Build GP
        if noiseless:
            model = SingleTaskGP(
                train_X=X,
                train_Y=y,
                train_Yvar=Yvar,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
                covar_module=kernel if kernel is not None else get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel= False),
            )
        else:
            model = SingleTaskGP(
                X,
                y,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
                covar_module=kernel if kernel is not None else get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel= False)
            )

        # Fit the Gp with marignal likelihood and count the time

        fit_start = time.time()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError as e:
            # Fallback: Hyperparameter vom letzten Schritt behalten (warm start)
            print(f"[FIT FAILED] iter, error: {e}")
            pass
        fit_times.append(time.time() - fit_start)

        # Hyperparameter loggen
        lengthscales.append(
            model.covar_module.lengthscale.detach().mean().item()
            if model.covar_module.lengthscale is not None
            else float("nan")
        )
        noises.append(
            model.likelihood.noise.detach().mean().item()
        )

        # set acquisition function to EI
        EI = qLogExpectedImprovement(model=model, best_f=y.max())

        # generate new sample by using the acquisition function
        try:
            x_next, _ = optimize_acqf(
                EI,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=200
            )
        except Exception as e:
            x_next = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, d, dtype=torch.double)
            print(f"optimize_acqf failed: {e}, using random point instead")
        # find out real value of the new sample
        y_next = f(x_next).unsqueeze(-1)
        y_next = y_next if maximize else -y_next

        # concatenate x_next and y_next to our data for new iteration
        X = torch.cat([X, x_next])
        y = torch.cat([y, y_next])

        # log new best value
        best = y.max().item()
        best_vals.append(best if maximize else -best)

        # log the regret
        regret = (best - f_opt) if maximize else (f_opt - best)
        regrets.append(regret)

        # log the time one iteration took
        iter_times.append(time.time() - iter_start)

    print("Done")

    return {
        "best_vals": best_vals,
        "regrets": regrets,
        "fit_times": fit_times,
        "iter_times": iter_times,
        "lengthscales": lengthscales,
        "noises": noises
    }


# Doesnt work/ Deprecated, I use fit_mll function instead, which can be used or LOO-PseudoLikelihood

# function to fit the gp with Leave one out crossvalidation with gradient based approach
def fit_gp_loocv(model: SingleTaskGP, loocv: LeaveOneOutPseudoLikelihood, X, y, num_iter=50):
    model.train()
    # better because we dont have many parameters
    optimizer = LBFGS(model.parameters(), lr=1)

    # flatten y
    y = y.view(-1)

    def closure():
        optimizer.zero_grad()
        # compute output
        output = model(X)
        # negative loss because we want to maximize but loss is minimized here
        loss = -loocv(output, y)
        # train with loss
        loss.backward()
        return loss

    # so that the optimizer can have middle steps with values out of the bounds
    with gpytorch.settings.debug(False), torch.distributions.Distribution.set_default_validate_args(False):
        for i in range(num_iter):
            optimizer.step(closure)
    torch.distributions.Distribution.set_default_validate_args

    model.eval()


def bo_loop_loocv(f, X_init, maximize: bool, noiseless=False, num_loop: int = 10, kernel=None):
    """
        the BO loop for the LOOCV method: build GP, optimize hyperparamaters via method, 
        use acquisition function to get next point and conatenate data

        uses the same fit_gpytorch_mll method but with the LeaveOneOutPseudoLikelihood
    
        f: the function to optimize
        X_init: initial data
        maximize: whether it is a problem where the function is to be maximized or minimized
        noiseless: wether to include noise or not (default is including noise)
        num_loop: amount of times the loop is repeated
        kernel: the kernel used, if None is given BoTorch's standard Matérn 2.5 kernel is used
    
    
        returns dict of arrays of best_vals, regrets, fit_times, iter_times, lengthscales and noises of each loop
    """
    fit_failures = 0
    # Set optimal value of f
    f_opt = f.optimal_value
    f_opt = f_opt if maximize else -f_opt

    # Set bounds of f
    bounds = f.bounds
    # set the dimension
    d = bounds.shape[-1]

    # generate initial data
    X = X_init.clone()
    y = f(X).unsqueeze(-1)
    y = y if maximize else -y

    best_vals = []
    regrets = []
    fit_times = []
    iter_times = []
    lengthscales = []
    noises = []

    for t in range(num_loop):
        iter_start = time.time()

        Yvar = torch.full_like(y, 1e-6)

        # build the gp
        if noiseless:
            model = SingleTaskGP(
                train_X=X,
                train_Y=y,
                train_Yvar=Yvar,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
                covar_module=kernel if kernel is not None else get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel= False)
            )
        else:
            model = SingleTaskGP(
                X,
                y,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
                covar_module=kernel if kernel is not None else get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel= False)
            )

        # fit the gp with LOOCV and count the time it takes

        fit_start = time.time()
        loocv = LeaveOneOutPseudoLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(loocv)
        except ModelFittingError:
            # Fallback: Hyperparameter vom letzten Schritt behalten (warm start)
            fit_failures +=1
            pass

        fit_times.append(time.time() - fit_start)

        # Hyperparameter loggen
        lengthscales.append(
            model.covar_module.lengthscale.detach().mean().item()
            if model.covar_module.lengthscale is not None
            else float("nan")
        )
        noises.append(
            model.likelihood.noise.detach().mean().item()
        )

        # set acquisition function
        EI = qLogExpectedImprovement(model, best_f=y.max())

        # get new sample through acquisition function
        try:
            x_next, _ = optimize_acqf(EI,
                                  bounds,
                                  q=1,
                                  num_restarts=20,
                                  raw_samples=200
                                  )
        except Exception as e:
            x_next = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, d, dtype=torch.double)
            print(f"optimize_acqf failed: {e}, using random point instead")

        # get true value of sample
        y_next = f(x_next).unsqueeze(-1)
        y_next = y_next if maximize else -y_next

        X = torch.cat([X, x_next])
        y = torch.cat([y, y_next])

        # log new best value
        best = y.max().item()
        best_vals.append(best if maximize else -best)

        # log the regret
        regret = (best - f_opt) if maximize else (f_opt - best)
        regrets.append(regret)

        # log the time one iteration took
        iter_times.append(time.time() - iter_start)

    print("Done")
    print(f"fit failures: {fit_failures}")

    return {
        "best_vals": best_vals,
        "regrets": regrets,
        "fit_times": fit_times,
        "iter_times": iter_times,
        "lengthscales": lengthscales,
        "noises": noises
    }


def build_model(X,y,theta):
    #different hyperparameters, depending on the kernel
    #for now only RBF kernel and hyperparameters

    #theta array: 0: lengthscale, 1: noise
    model = SingleTaskGP(X,y)
    lengthscale = theta[:-1]
    noise = theta[-1]
    with torch.no_grad():
        model.covar_module.lengthscale.copy_(lengthscale.unsqueeze(0))
        model.likelihood.noise_covar.noise.copy_(noise.unsqueeze(0))
    return model

#Function for k fold Crossvalidation (used for EA Crossvalidation approach)
def k_fold_crossvalidation(X,y,theta, k):
    k_fold = KFold(n_splits= k)
    scores = []

    for train_idx, test_idx in k_fold.split(X):
        #Split data in test and train sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #build model with train set
        model = build_model(X_train,y_train,theta)

        #evaluate the data
        model.eval()

        with torch.no_grad():
            pred = model(X_test)
            #use predictive performace as score
            score = pred.log_prob(y_test.squeeze(-1)).sum()
            scores.append(score)

    return torch.stack(scores).mean()


import cma


# Basic Evolutionairy Algorithm (deprecated) was replaced by CMA-ES


# initialize the population by using theta base and adding small noise to it
def init_population(theta_base, population_size, alpha=0.3):
    population = [theta_base.clone()]
    for _ in range(population_size):
        noise = alpha * torch.randn_like(theta_base)

        individual = theta_base + noise

        population.append(individual)
    return population


# mutate given parent and return child
def mutate(parent, alpha=0.2):
    # using log to guarantee positivity
    log_theta = torch.log(parent)
    mutation = alpha * torch.randn_like(parent)

    log_individual = parent + mutation

    return torch.exp(log_individual)


# Basic EA
def fit_basic_ea(X, y, model, k=5, population_size=20, num_iter=100):
    # generate initial population
    lengthscale = model.covar_module.lengthscale.detach().squeeze()
    noise = model.likelihood.noise_covar.noise.detach().squeeze()
    theta = torch.cat([lengthscale.flatten(), noise.view(1)])

    population = init_population(theta, population_size)

    for _ in range(num_iter):
        # compute fitness of the individuals in the population via k-fold-crossvalidation
        fitness = []
        for individual in population:
            fitness.append((individual, k_fold_crossvalidation(X, y, individual, k)))

        # select the n best indiviudals in the population

        # how high is the fraction of survivors in new population
        survivor_fraction = 0.5
        n = int(population_size * survivor_fraction)
        chosen = sorted(fitness, key=lambda x: x[1], reverse=True)[:n]

        # Mutation
        # Elitism: best n parents survive
        new_population = []
        for parent in chosen:
            new_population.append(parent[0].clone())

        # add mutated children till population_size is reached
        while (len(new_population) < population_size):
            parent = random.choice(chosen)

            child = mutate(parent[0])
            new_population.append(child)

        # set current population to new_population and loop
        population = new_population
    return max(population, key=lambda x: x[1])[0]


# Fit funktion für CMAES
def fit_cmaes(model, X, y, k=5):
    model.train()

    d = X.shape[-1]

    x0 = np.concatenate([
        model.covar_module.lengthscale.detach().log().numpy().flatten(),
        model.likelihood.noise.detach().log().numpy().flatten(),
    ])

    def objective(log_params):
        # setting parameters (in log ti guarantee positivity)
        lengthscales = torch.tensor(np.exp(log_params[:-1]))
        noise = torch.tensor([np.exp(log_params[-1])])

        theta = torch.cat([lengthscales.flatten(), noise])

        score = k_fold_crossvalidation(X, y, theta, k)
        return -score.item()

    # CMA-ES optimieren
    es = cma.CMAEvolutionStrategy(x0, sigma0=0.5, inopts={
        "verbose": -9,
        "maxfevals": 250,  # max 100 evaluations
        "tolfun": 1e-6,  #
        "bounds": [-5, 5],  # constraints so that it is more stable
    }
                                  )
    es.optimize(objective)

    best = es.result.xbest

    model.eval()

    return best


from sklearn.model_selection import KFold, cross_val_score


def bo_loop_ea(f, X_init, maximize: bool, noiseless=False, num_loop: int = 10, kernel=None):
    """
        the BO loop for the CMA-ES method: build GP, optimize hyperparamaters via method, 
        use acquisition function to get next point and conatenate data
    
        uses own fit_cmaes method to fit the hyperparameters
    
        f: the function to optimize
        X_init: initial data
        maximize: whether it is a problem where the function is to be maximized or minimized
        noiseless: wether to include noise or not (default is including noise)
        num_loop: amount of times the loop is repeated
        kernel: the kernel used, if None is given BoTorch's standard Matérn 2.5 kernel is used
    
    
        returns dict of arrays of best_vals, regrets, fit_times, iter_times, lengthscales and noises of each loop
        """
    # Set optimal value of f
    f_opt = f.optimal_value
    f_opt = f_opt if maximize else -f_opt
    # Set bounds of f
    bounds = f.bounds
    # set the dimension
    d = bounds.shape[-1]

    # generate initial data
    X = X_init.clone()
    y = f(X).unsqueeze(-1)
    y = y if maximize else -y

    best_vals = []
    regrets = []
    fit_times = []
    iter_times = []
    lengthscales = []
    noises = []

    for t in range(num_loop):
        iter_start = time.time()
        Yvar = torch.full_like(y, 1e-6)

        # build the gp
        if noiseless:
            model = SingleTaskGP(
                train_X=X,
                train_Y=y,
                train_Yvar=Yvar,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
                covar_module=kernel if kernel is not None else get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel= False)
            )
        else:
            model = SingleTaskGP(
                X,
                y,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1)
            )

        fit_start = time.time()

        # Fit data with cmaes using K-fold-CV as objective
        best_theta = fit_cmaes(X=X, y=y, model=model)
        with torch.no_grad():
            model.covar_module.lengthscale = torch.tensor(np.exp(best_theta[:-1])).unsqueeze(0)
            model.likelihood.noise = float(np.exp(best_theta[-1]))
        fit_times.append(time.time() - fit_start)

        # Hyperparameter loggen
        lengthscales.append(
            model.covar_module.lengthscale.detach().mean().item()
            if model.covar_module.lengthscale is not None
            else float("nan")
        )
        noises.append(
            model.likelihood.noise.detach().mean().item()
        )

        # set acquisition function
        EI = qLogExpectedImprovement(model, best_f=y.max())

        # get new sample through acquisition function
        try:
            x_next, _ = optimize_acqf(EI,
                                    bounds,
                                    q=1,
                                    num_restarts=20,
                                    raw_samples=200
                                    )
        except Exception as e:
            x_next = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, d, dtype=torch.double)
            print(f"optimize_acqf failed: {e}, using random point instead")
        # get true value of sample
        y_next = f(x_next).unsqueeze(-1)
        y_next = y_next if maximize else -y_next

        X = torch.cat([X, x_next])
        y = torch.cat([y, y_next])

        # log new best value
        best = y.max().item()
        best_vals.append(best if maximize else -best)

        # log the regret
        regret = (best - f_opt) if maximize else (f_opt - best)
        regrets.append(regret)

        iter_times.append(time.time() - iter_start)

    print("Done")

    return {
        "best_vals": best_vals,
        "regrets": regrets,
        "fit_times": fit_times,
        "iter_times": iter_times,
        "lengthscales": lengthscales,
        "noises": noises
    }


def bo_loop_mcmc(f, X_init, maximize: bool, kernel = None, noiseless=False, num_loop: int = 10,
                 warmup_steps: int = 128, num_samples: int = 64, thinning: int = 8, variant: str = "v3"):
    """
        the BO loop for the MCMC method: build GP, optimize hyperparamaters via method, 
        use acquisition function to get next point and conatenate data

        uses fit_fully_bayesian_model_nuts to optimize the hyperparameters
    
        f: the function to optimize
        X_init: initial data
        maximize: whether it is a problem where the function is to be maximized or minimized
        noiseless: wether to include noise or not (default is including noise)
        num_loop: amount of times the loop is repeated
        kernel: the kernel used, if None is given BoTorch's standard Matérn 2.5 kernel is used
        warmup_steps: the amount of warmup steps, default is 128
        num_samples: the number of samples, default is 64
        thinning: the amounf of thinning, default is 8
        variant: which version of the loop is used: options are (v1,v2,v3); needed for Experiment 2
            for the usage of the uninformative prior and (no) ARD
    
        returns dict of arrays of best_vals, regrets, fit_times, iter_times, lengthscales and noises of each loop
    """

    #Print statement to make sure that the correct amount of warmup steps, etc. is given to this method
    print(f"[MCMC CONFIG] warmup_steps={warmup_steps}, num_samples={num_samples}, "
          f"thinning={thinning}, num_loop={num_loop}")
    f_opt = f.optimal_value
    f_opt = f_opt if maximize else -f_opt
    bounds = f.bounds
    d = bounds.shape[-1]

    X = X_init.clone()
    y = f(X).unsqueeze(-1)
    y = y if maximize else -y

    best_vals = []
    regrets = []
    fit_times = []
    iter_times = []
    noises = []
    lengthscales = []

    pyro_model = get_pyro_model_for_kernel_and_variants(kernel,variant)

    class GPClass(FullyBayesianSingleTaskGP):
        _pyro_model_class = pyro_model

    for t in range(num_loop):
        iter_start = time.time()

        if noiseless:
            Yvar = torch.full_like(y, 1e-6)
            noises.append(1e-6)

            model = GPClass(
                train_X=X,
                train_Y=y,
                train_Yvar=Yvar,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
            )
        else:
            model = GPClass(
                train_X=X,
                train_Y=y,
                train_Yvar=None,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
            )

        fit_start = time.time()

        # Stack clearen
        _PYRO_STACK.clear()
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            thinning=thinning,
            disable_progbar=True,
        )
        fit_times.append(time.time() - fit_start)

        jax.clear_caches()

         # Hyperparameter loggen
        lengthscales.append(
            model.median_lengthscale.detach().mean().item()
        )
        if not noiseless:
            noises.append(
                model.likelihood.noise.detach().mean().item()
            )

        EI = qLogExpectedImprovement(model=model, best_f=y.max())

        x_next, _ = optimize_acqf(
            EI,
            bounds=bounds,
            q=1,
            num_restarts=40,
            raw_samples=512,
        )

        y_next = f(x_next).unsqueeze(-1)
        y_next = y_next if maximize else -y_next

        X = torch.cat([X, x_next])
        y = torch.cat([y, y_next])

        best = y.max().item()
        best_vals.append(best if maximize else -best)

        regret = (best - f_opt) if maximize else (f_opt - best)
        regrets.append(regret)
        iter_times.append(time.time() - iter_start)

    print("Done")

    return {
        "best_vals": best_vals,
        "regrets": regrets,
        "fit_times": fit_times,
        "iter_times": iter_times,
        "lengthscales": lengthscales,   # MCMC hat keine einzelne Lengthscale — Posterior über viele
        "noises": noises, #festes noise deswegen
    }