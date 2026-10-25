
import jax.numpy as jnp
from botorch.models.fully_bayesian import MaternPyroModel, compute_dists
from gpytorch.kernels import MaternKernel, Kernel, ScaleKernel, RBFKernel
import torch
import pyro
import numpyro
import numpyro.distributions as numpyro_dist
from functools import partial


def get_pyro_model_for_kernel(kernel):
    """
    returns a pyromodel for the given kernel
    """
    if kernel is None: 
        return None
    
    if isinstance(kernel, MaternKernel):
        nu = kernel.nu
        # da man mit float nicht == machen sollte
        if abs(nu - 0.5) < 1e-6:
            return Matern05PyroModel
        else:
            return None
    elif isinstance(kernel, RBFKernel):
        return RBFPyroModel
    
    else:
        print("not supported for MCMC using default Matern Kernel 5/2")
        return None

def matern05_kernel(X, lengthscale):
    dist = compute_dists(X=X, lengthscale=lengthscale)
    return jnp.exp(-dist)

def rbf_kernel(X, lengthscale):
    dist = compute_dists(X=X, lengthscale=lengthscale)
    return jnp.exp(-0.5 * dist**2)

class Matern05PyroModel(MaternPyroModel):
    """
    The Matern 0.5 pyro model, as the NUTS PyroModel only has a Matérn 2.5 pyromodel.
    Here the nu values are changed to 0.5 in get_covar_module() and sample() uses matern05_kernel() .
    """
    def sample(self) -> None:
        """
        This samples the mean, noise variance, (optional) outputscale, and
        lengthscales according to a dimension-scaled prior.

        """
        outputscale = self.sample_outputscale(
            concentration=self._outputscale_prior_concentration,
            rate=self._outputscale_prior_rate,
        )
        mean = self.sample_mean()
        noise = self.sample_noise()
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims)
        X_tf = self._maybe_input_warp(X_jax=self.train_X_jax)
        X_tf = self._prepare_features(X=X_tf)
        K_noiseless = outputscale * matern05_kernel(X=X_tf, lengthscale=lengthscale)
        K_noiseless, mean = self._maybe_multitask_transform(K_noiseless, mean)
        self.sample_observations(mean=mean, K_noiseless=K_noiseless, noise=noise)

    def _get_covar_module(
        self,
        use_scale_kernel: bool,
        batch_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Kernel:
        covar_module = MaternKernel(
            nu = 0.5,
            ard_num_dims=self.ard_num_dims,
            batch_shape=batch_shape,
        )
        if use_scale_kernel:
            covar_module = ScaleKernel(covar_module, batch_shape=batch_shape)
        return covar_module.to(dtype=dtype, device=device)
     
class RBFPyroModel(MaternPyroModel):
    """
        The RBF pyro model, as the NUTS PyroModel only has a Matérn 2.5 pyromodel.
        Here the covar module is changed to a RBF Kernel and rbf_kernel() is used.
    """
    def sample(self) -> None:
        """
        Sample from the RBF pyro model.

        This samples the mean, noise variance, (optional) outputscale, and
        lengthscales according to a dimension-scaled prior.
        """
        outputscale = self.sample_outputscale(
            concentration=self._outputscale_prior_concentration,
            rate=self._outputscale_prior_rate,
        )
        mean = self.sample_mean()
        noise = self.sample_noise()
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims)
        X_tf = self._maybe_input_warp(X_jax=self.train_X_jax)
        X_tf = self._prepare_features(X=X_tf)
        K_noiseless = outputscale * rbf_kernel(X=X_tf, lengthscale=lengthscale)
        K_noiseless, mean = self._maybe_multitask_transform(K_noiseless, mean)
        self.sample_observations(mean=mean, K_noiseless=K_noiseless, noise=noise)

    def _get_covar_module(
        self,
        use_scale_kernel: bool,
        batch_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Kernel:
        covar_module = RBFKernel(
            ard_num_dims=self.ard_num_dims,
            batch_shape=batch_shape,
        )
        if use_scale_kernel:
            covar_module = ScaleKernel(covar_module, batch_shape=batch_shape)
        return covar_module.to(dtype=dtype, device=device)
     

class ConfigurablePriorModel:
    """
    Mixin, which is inherited from together with the PyroModels and combined, to 
    turn ARD and prior on/off for each kernel choice. For prior = "hvarfner" the standard
    prior for this thesis is used for prior = "weak" an uninformative HalfCauchy prior is used
    """

    def __init__(self, *args, use_ard: bool, prior : str = "hvarfner", **kwargs):
        #Only set ard and prior and rest is done normally
        self.use_ard = use_ard
        self.prior = prior
        super().__init__(*args, **kwargs)

    def sample_lengthscale(self, dim: int, **tkwargs):
        # If a prior is chosen in all experiments ARD is used aswell
        if self.prior == "hvarfner":
            return super().sample_lengthscale(dim, **tkwargs)

        #sample a lengthscale per dim
        sample_dim = dim if self.use_ard else 1
        if self.prior == "weak":
            raw_ls = numpyro.sample(
                "raw_lengthscale",
                numpyro_dist.HalfCauchy(
                    jnp.full((sample_dim,), 2.5)
                ),
            )

        else:
            raise ValueError(f"Unknown prior: {self.prior}")

        #if ard is not used one lengthscale is used for all dim
        ls = raw_ls * jnp.ones(dim) if not self.use_ard else raw_ls
        return numpyro.deterministic("lengthscale", ls)

#Classes that inherit from Configurable Prior and the respective kernelmodel

class Matern25ConfigurablePriorModel(ConfigurablePriorModel, MaternPyroModel):
    pass

class Matern05ConfigurablePriorModel(ConfigurablePriorModel, Matern05PyroModel):
    pass

class RBFConfigurablePriorModel(ConfigurablePriorModel, RBFPyroModel):
    pass


def get_pyro_model_for_kernel_and_variants(kernel, variant:str):
    """
    Method to get model for given kernel and for different variants (no ard + no prior, ard + no prior, ard + prior)  
    combines pyromodel (got via get_pyro_model_for_kernel(kernel)) with Variant (ARD and Prior choice). 
    Returns partial() with variant map, which can than be used by the MCMC loop itself (for the other two methods
    it is not needed)
    """
    variant_map = {
            "v1": {"use_ard": False, "prior": "weak"},
            "v2": {"use_ard": True, "prior": "weak"},
            "v3": {"use_ard": True, "prior": "hvarfner"},
        }
    kwargs = variant_map[variant]

    pyro_model = get_pyro_model_for_kernel(kernel)

    #partial is used to already set use_ard and prior parameters and leave the rest to be decided
    if pyro_model is Matern05PyroModel:
        return partial(Matern05ConfigurablePriorModel, **kwargs)
    elif pyro_model is RBFPyroModel:
        return partial(RBFConfigurablePriorModel, **kwargs)
    elif pyro_model is None:
        return partial(Matern25ConfigurablePriorModel, **kwargs)
    else:
        raise ValueError(f"No MCMC Pyromodel available for {kernel}")