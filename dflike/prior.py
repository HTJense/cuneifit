import jax
import jax.numpy as jnp
from collections.abc import Callable


class Prior:
    def __init__(self, priors: list[Callable] = []):
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

    def logprior(self, theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(jnp.array([
            pi(*theta[idx]) for pi, idx in zip(self.priors, self.indices)
        ]))

    def fisher(self, theta: jnp.ndarray) -> jnp.ndarray:
        res = jnp.zeros((len(self.parameters), len(self.parameters)))

        for pi2, idx in zip(self.H_priors, self.indices):
            H = -jnp.array(pi2(*theta[idx]))
            res = res.at[jnp.ix_(idx, idx)].add(H)

        return res
