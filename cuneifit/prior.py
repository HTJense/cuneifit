from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.scipy as jsc  # noqa: F401
from collections.abc import Callable

import logging

logger = logging.getLogger(__name__)


class Prior:
    """A Prior class is a set of priors that contain contributions to the
    log-posterior.

    In general, priors are created from a list of lambda functions. For
    example:

    .. code-block:: python

        prior = Prior(
            # A 1% calibration uncertainty prior.
            lambda calibration: jsc.stats.norm.logpdf(calibration, 1.0, 0.01),
            # A prior on optical depth tau = 0.06 +/- 0.01
            lambda tau: jsc.stats.norm.logpdf(tau, 0.06, 0.01),
            # A joint prior on the SZ amplitudes (cf. Planck 2018)
            lambda a_kSZ, a_tSZ: jsc.stats.norm.logpdf(
                a_kSZ + 1.6 * a_tSZ, 9.5, 3.0
            )

        pipeline = Pipeline(prior, likelihood, ...)

    For the most part, priors function the same as Likelihoods. However, they
    are designed to be more lightweight, and have very few joint parameters.

    You are able to combine two priors through addition, for example:

    .. code-block:: python

        cal_prior = Prior(
            lambda calibration: jsc.stats.norm.logpdf(calibration, 1.0, 0.01),
        )

        tau_prior = Prior(
            lambda tau: jsc.stats.norm.logpdf(tau, 0.06, 0.01),
        )

        sz_prior = Prior(
            lambda a_kSZ, a_tSZ: jsc.stats.norm.logpdf(
                a_kSZ + 1.6 * a_tSZ, 9.5, 3.0
            )
        )

        prior = cal_prior + tau_prior + sz_prior

        # These two are equivalent.
        pipeline = Pipeline(prior, ...)
        pipeline = Pipeline(cal_prior, tau_prior, sz_prior, ...)
    """
    def __init__(self, *priors):
        self.priors = []
        self.H_priors = []
        # All parameters, in order of appearance.
        self.parameters = []
        # Individual parameter names per prior.
        self.individual_parameters = []
        self.indices = []

        for logpi in priors:
            self.add_prior(logpi)

    @staticmethod
    def extract_parameter_names(function) -> list[str]:
        # Reads the names of the parameters from the function signature.
        import inspect

        sig = inspect.signature(function)
        return [
            par.name for par in sig.parameters.values()
            if par.kind in (par.POSITIONAL_ONLY, par.POSITIONAL_OR_KEYWORD)
            and par.default is par.empty
        ]
    
    def add_prior(self, logpi: str | Callable[[jnp.ndarray], float], /,
                  func_pars: list | None = None):
        if isinstance(logpi, str):
            logpi = eval(logpi)

        if func_pars is None:
            func_pars = self.extract_parameter_names(logpi)

        func_pars = list(func_pars)
        self.priors.append(logpi)
        self.individual_parameters.append(func_pars)

        func_idx = []
        for i, p in enumerate(func_pars):
            if p not in self.parameters:
                self.parameters.append(p)
            func_idx.append(self.parameters.index(p))

        self.H_priors.append(jax.hessian(logpi,
                                         argnums=list(range(len(func_idx)))))
        self.indices.append(jnp.array(func_idx, dtype=int))

    def logprior(self, theta: jnp.ndarray, /) -> jnp.ndarray:
        return jnp.sum(jnp.array([
            pi(*theta[idx]) for pi, idx in zip(self.priors, self.indices)
        ]))

    def fisher(self, theta: jnp.ndarray, /) -> jnp.ndarray:
        res = jnp.zeros((len(self.parameters), len(self.parameters)))

        for pi2, idx in zip(self.H_priors, self.indices):
            H = -jnp.array(pi2(*theta[idx]))
            res = res.at[jnp.ix_(idx, idx)].add(H)

        return res

    def extend(self, other: Prior | Callable | list | tuple | None):
        # Add another prior into this prior.
        if other is None:
            return
        elif isinstance(other, Prior):
            for prior, params in zip(other.priors, other.individual_parameters):
                self.add_prior(prior, func_pars=params)
        elif callable(other) or isinstance(other, str):
            self.add_prior(other)
        elif isinstance(other, (list, tuple)):
            for o in other:
                self.__add__(o)
        else:
            raise TypeError(f"Cannot add type {type(other)} to Prior.")

    def __add__(self, other: Prior | Callable | list[Callable]):
        result = Prior()
        result.extend(self)
        result.extend(other)
        return result
