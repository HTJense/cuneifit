from __future__ import annotations
import jax
import jax.numpy as jnp
from collections.abc import Callable


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
        self.parameters = []
        self.indices = []

        for logpi in priors:
            self.add_prior(logpi)

    def add_prior(self, logpi: Callable[[jnp.ndarray], float]):
        self.priors.append(logpi)

        co = logpi.__code__
        func_pars = co.co_varnames[:co.co_argcount]
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

    def __add__(self, other: Prior | Callable | list[Callable]):
        if other is None:
            return self
        elif type(other) is Prior:
            for prior in other.priors:
                self.add_prior(prior)
        elif type(other) is Callable:
            self.add_prior(other)
        elif type(other) is list:
            for o in other:
                self.__add__(o)
        else:
            raise TypeError(f"Cannot add type {type(other)} to Prior.")

        return self
