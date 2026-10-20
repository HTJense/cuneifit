from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import jax.scipy as jsc


class Likelihood(ABC):
    """Any Bayesian Likelihood object L(d|theta).

    Parameters
    ----------
    parameters : list[str]
        A list of parameters expected for any function that takes in `theta`.
        
    """
    parameters: list[str]

    @abstractmethod
    def loglike(self, theta: jnp.ndarray, **kwargs) -> float:
        """The log-likelihood function of this object.

        Parameters
        ----------
        theta : jnp.ndarray
            A list of scalar parameters used by this function. The meaning of
            each entry should match the name given by this class' `parameters`
            member.
        """
        ...

    @abstractmethod
    def fisher(self, theta: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """A shorthand for the Fisher matrix F = -H[ln L].

        Parameters
        ----------
        theta : jnp.ndarray
            A list of scalar parameters used by this function. The meaning of
            each entry should match the name given by this class' `parameters`
            member.
        """
        ...


class GaussianLikelihood(Likelihood):
    """A Gaussian likelihood class.

    This likelihood is a specialization of the form
    L(d|theta) ~ - 1/2 (d - mu(theta))^T C^-1 (d - mu(theta))
    with a series of associated functions to improve computational speed.
    """
    data_vec: jnp.ndarray
    covariance: jnp.ndarray
    logp_const: float

    @abstractmethod
    def get_model(self, theta, **kwargs):
        """The model function mu(theta)
        """
        ...

    def get_covariance(self, theta, **kwargs):
        return self.covariance

    def get_covariance_cholesky(self, theta, **kwargs):
        covariance = self.get_covariance(theta, **kwargs)
        return jnp.linalg.cholesky(covariance)

    def get_whitened_residual(self, theta, **kwargs):
        mu = self.get_model(theta, **kwargs)
        r = self.data_vec - mu
        L = self.get_covariance_cholesky(theta, **kwargs)
        return jsc.linalg.solve_triangular(L, r, lower=True)

    def loglike(self, theta, **kwargs):
        return -0.5 * self.chisquare(theta, **kwargs) + self.logp_const

    def chisquare(self, theta, **kwargs):
        """Compute the chi square
        `chi2 = (d - mu(theta))^T C^-1 (d - mu(theta))`.
        """
        w = self.get_whitened_residual(theta, **kwargs)
        return jnp.sum(w ** 2)

    def fisher(self, theta, **kwargs):
        J = jax.jacfwd(lambda x: self.get_model(x, **kwargs))(theta)
        L = self.get_covariance_cholesky(theta, **kwargs)
        W = jsc.linalg.solve_triangular(L, J, lower=True)

        return W.T @ W
