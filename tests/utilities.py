import jax
import jax.numpy as jnp

from cuneifit.likelihood import GaussianLikelihood


class TestLikelihood(GaussianLikelihood):
    parameters = ["a"]
    data_vec = jnp.array([0.0])
    covariance = jnp.array([[1.0]])
    logp_const = 0.0

    def get_model(self, theta, /, **kwargs):
        return theta[0]
