#Bo Loops Imports
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

import time
import copy
import torch
import numpy as np

from experiments import generateInitialData, setSeed


#Carps Imports
from ConfigSpace import Configuration, ConfigurationSpace
from carps.utils.trials import TrialInfo, TrialValue
from carps.optimizers.optimizer import Optimizer
from carps.loggers.abstract_logger import AbstractLogger
from carps.utils.types import Incumbent
from carps.utils.task import Task
from ConfigSpace.hyperparameters import (
    NumericalHyperparameter,   # Float + Integer
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Constant,
    IntegerHyperparameter
)

"""Wrapper Functions for CARPS: Adapting the existing methods (MLL,LOOCV,MCMC) to the ask/tell structure
   of CARPS, while keeping its functionality the same as for the synthetic experiments"""



# Base Optimizer, as most of the classes are the same for all Optimizers
class BaseOptimizer(Optimizer):
    """Base for all Optimizers for each method: Conversion, setup, tell and optimal value logic, 
       that stays the same across all of the Optimizers
       Most of the Optimizers only need to overwrite the ask_bo method of this Optimizer"""
    def __init__(
        self,
        task,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,
        expects_fidelities: bool = False,
        n_initial: int = 10,
        num_loop: int = 50,
    ) -> None:
        super().__init__(task, loggers, expects_multiple_objectives, expects_fidelities)
        self.n_initial = n_initial
        self.num_loop = num_loop

        #Are set in convert configspace
        self.configspace: ConfigurationSpace | None = None
        self.bounds: torch.Tensor | None = None
 
        self.X: torch.Tensor | None = None
        self.y: torch.Tensor | None = None
        self.iteration = 0

    
    def convert_configspace(self, configspace: ConfigurationSpace):
        """
        extracts bounds out configspace and sets them in tensor form, as BoTorch continuous acq. function optimizer
        only works with real bounds. Allows loop to be used on the config spaces with categorical/ordinal/integer hyperparameters
        
        each hyperparameter becomes one or more entries in self.encoding_spec (kind("numerical, ordinal,...), name, extra, placeholder):
         - integer/continuous: one entry bounded by [lower, upper]
         - ordinal: one entry bounded by [min(sequence), max(seqence)], rounded to nearest legal value
         - categorical: one-hot-encoding entry per choice (each bounded to [0,1])
         - Constant: not encoded, but stored in self.constants and re-added on every decode
        """
        self.encoding_spec = []
        self.constants = {}
        # Saves all conditions of hyperparameters in self.conditions
        # No need to change something for an inactive hyperparameter, as allow_inactive_with_values just ignores the value that is set for 
        # the hyperparameter if it is inactive (in _tensor_to_config)
        self.conditions : dict[str, list] = {}
        for cond in configspace.get_conditions():
            self.conditions.setdefault(cond.child.name, []).append(cond)

        lbs, ubs = [], []

        for name, hp in configspace.items():
            if(isinstance(hp, Constant)):
                self.constants[name] = hp.value
            elif(isinstance(hp,CategoricalHyperparameter)):
                self.encoding_spec.append(("categorical", name, tuple(hp.choices), None))
                lbs.extend([0.0] * len(hp.choices))
                ubs.extend([1.0]* len(hp.choices))
            elif(isinstance(hp, OrdinalHyperparameter)):
                seq = tuple(hp.sequence)
                placeholder = seq[len(seq) //2]
                self.encoding_spec.append(("ordinal", name, seq, placeholder))
                lbs.append(min(hp.sequence))
                ubs.append(max(hp.sequence))
            elif(isinstance(hp, IntegerHyperparameter)):
                placeholder = (hp.lower + hp.upper) / 2
                self.encoding_spec.append(("integer", name, None, placeholder))
                lbs.append(hp.lower)
                ubs.append(hp.upper)
            elif(isinstance(hp, NumericalHyperparameter)):
                placeholder = (hp.lower + hp.upper) / 2    
                self.encoding_spec.append(("continuous", name, None, placeholder))
                lbs.append(hp.lower)
                ubs.append(hp.upper)
            else:
                raise TypeError(f"Unknown hyperparamete type for {name!r}: {type(hp)}")


        self.bounds = torch.tensor([lbs,ubs], dtype= torch.double)
        self.configspace = configspace
        return configspace

    def convert_to_trial(self, x: torch.Tensor) -> TrialInfo:
        return TrialInfo(config=self._tensor_to_config(x))


    def get_current_incumbent(self) -> Incumbent:
        #Returns best calculated point
        if self.y is None:
            return None
        best_idx = self.y.argmax().item()
        trial_info = self.convert_to_trial(self.X[best_idx])
        trial_value = TrialValue(cost=-self.y[best_idx].item())
        return (trial_info, trial_value)
    


    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        # Add result to training set
        x = self._config_to_tensor(trial_info.config)
        # because we internally maximize y (same as loops.py), the trial value is made negative
        y = torch.tensor([[-trial_value.cost]], dtype=torch.double)

        if self.X is None:
            self.X = x.unsqueeze(0)
            self.y = y
        else:
            self.X = torch.cat([self.X, x.unsqueeze(0)])
            self.y = torch.cat([self.y, y])


    #From Configuration to Tensor
    def _config_to_tensor(self, config: Configuration) -> torch.Tensor:
        """
        Encodes a Configuration into the tensor from convert_configspace (self.encoding_spec)
        Inactive hyperparameters fallback to the placeholder value
        """
        cfg = dict(config)
        values : list[float] = []
        
        for kind, name, extra, placeholder in self.encoding_spec:
            if kind in ("continuous", "integer", "ordinal"):
                values.append(float(cfg[name]) if name in cfg else float(placeholder))
            elif kind == "categorical":
                if name in cfg:
                    chosen = config[name]
                    values.extend([1.0 if choice == chosen else 0.0 for choice in extra])
                else:
                    values.extend([0.0]* len(extra))

        return torch.tensor(values, dtype=torch.double)

    #From Tensor to Configuration
    def _tensor_to_config(self, x: torch.Tensor) -> Configuration:
        """
        Inverse of _config_to_tenso. Integer and Ordinal Values are rounded to the nearest allowed value,
        categorical values are decoded via argmax. allow_inactive_values= True just ignores values of hyperparameters 
        that are inactive, leading to them being set, but not used
        """
        values: dict = dict(self.constants)
        i = 0
        for kind, name, extra, placeholder in self.encoding_spec:
            if kind == "continuous":
                values[name] = float(x[i])
                i += 1
            elif kind == "integer":
                values[name] = int(round(float(x[i])))
                i += 1
            elif kind == "ordinal":
                raw = float(x[i])
                values[name] = min(extra, key=lambda v: abs(v - raw))
                i += 1
            elif kind == "categorical":
                block = x[i : i + len(extra)]
                values[name] = extra[int(block.argmax())]
                i += len(extra)

        return Configuration(self.configspace, values=values, allow_inactive_with_values=True)
    
    def _setup_common(self) -> None:
        #Is called everytime setup_optimizer is called
        configspace = self.task.objective_function.configspace
        self.convert_configspace(configspace)

        d = self.bounds.shape[-1]
        self.kernel = get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel= False)
        self.iteration = 0

        #sampling initial data
        X_init = generateInitialData(self.bounds, n=self.n_initial)
        for x in X_init:
            trial_info = self.convert_to_trial(x)
            trial_value = self.task.objective_function.evaluate(trial_info=trial_info)
            self.tell(trial_info, trial_value)

    def ask(self) -> TrialInfo:
        # Fallback if initial data hasnt been sampled yet
        if self.X is None or len(self.X) < self.n_initial:
            config = self.configspace.sample_configuration()
            return TrialInfo(config=config)
        return self._ask_bo()
 
    def _ask_bo(self) -> TrialInfo:
        #implemented in MLL, LOOCV, MCMC
        raise NotImplementedError

"""
For all Optimizers, _ask_bo does the same thing it does in the bo loops as specified in loops.py
"""

class MLLOptimizer(BaseOptimizer):
    """Optimizer for MLL"""

    def _setup_optimizer(self):
        self._setup_common()

    def _ask_bo(self):
        model = SingleTaskGP(
            self.X,
            self.y,
            input_transform=Normalize(d=self.X.shape[-1]),
            outcome_transform=Standardize(m=1),
            covar_module=copy.deepcopy(self.kernel)
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        #Get next samplepoint
        EI = qLogExpectedImprovement(model=model, best_f=self.y.max())
        x_next, _ = optimize_acqf(
            EI,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=200
        )

        self.iteration += 1
        return self.convert_to_trial(x_next.squeeze(0))

    


class LOOCVOptimizer(BaseOptimizer):
    """Optimizer for LOOCV"""
    def _setup_optimizer(self):
        self._setup_common()

    def _ask_bo(self):
        model = SingleTaskGP(
            self.X,
            self.y,
            input_transform=Normalize(d=self.X.shape[-1]),
            outcome_transform=Standardize(m=1),
            covar_module=copy.deepcopy(self.kernel)
        )
        loocv = LeaveOneOutPseudoLikelihood(model.likelihood, model)
        fit_gpytorch_mll(loocv)

        #Was used for debugging purposes, is deprecated now

        #kernel_module = model.covar_module.base_kernel if hasattr(model.covar_module, "base_kernel") else model.covar_module
        #print(f"[CARPS] iter={self.iteration} X.shape={self.X.shape} y.min={self.y.min().item():.4f} "
        #    f"kernel_type={type(model.covar_module).__name__} "
        #    f"lengthscale={kernel_module.lengthscale.detach().tolist()} "
        #   f"noise={model.likelihood.noise.detach().item():.6f}")

        #Get next samplepoint
        EI = qLogExpectedImprovement(model=model, best_f=self.y.max())
        x_next, _ = optimize_acqf(
            EI,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=200
        )

        self.iteration += 1
        return self.convert_to_trial(x_next.squeeze(0))
    

class MCMCOptimizer(BaseOptimizer):
    """Optimizer for MCMC"""
    #Has its own init, because we want warmup_steps, num_samples and thinning parameters to be changeable for CARPS as well
    def __init__(
        self,
        task: Task,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,
        expects_fidelities: bool = False,
        n_initial: int = 10,
        num_loop: int = 50,
        warmup_steps: int = 128,
        num_samples: int = 64,
        thinning: int = 8,
    ) -> None:
        super().__init__(
            task, loggers, expects_multiple_objectives, expects_fidelities, n_initial, num_loop
        )
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning

    def _setup_optimizer(self):
        self._setup_common()

    def _ask_bo(self):
        model = FullyBayesianSingleTaskGP(
            self.X,
            self.y,
            input_transform=Normalize(d=self.X.shape[-1]),
            outcome_transform=Standardize(m=1),
        )
        
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=True,
        )

        #Get next samplepoint
        EI = qLogExpectedImprovement(model=model, best_f=self.y.max())
        x_next, _ = optimize_acqf(
            EI,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=200
        )

        self.iteration += 1
        return self.convert_to_trial(x_next.squeeze(0))