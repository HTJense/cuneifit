import pytest  # noqa: F401


def test_import():
    from cuneifit.likelihood import Likelihood, GaussianLikelihood  # noqa: F401


def test_inheritance():
    from cuneifit.likelihood import GaussianLikelihood
    from jax import numpy as jnp

    class TestLikelihood(GaussianLikelihood):
        parameters = ["a"]
        data_vec = jnp.array([0.0])
        covariance = jnp.array([[1.0]])

        def get_model(self, theta, /, **kwargs):
            return theta[0]

    TestLikelihood()


def test_chisquare():
    from cuneifit.likelihood import GaussianLikelihood
    from jax import numpy as jnp

    class TestLikelihood(GaussianLikelihood):
        parameters = ["a"]
        data_vec = jnp.array([0.0])
        covariance = jnp.array([[1.0]])

        def get_model(self, theta, /, **kwargs):
            return theta[0]

    like = TestLikelihood()
    assert like.chisquare(like.data_vec) == 0.0


def test_gradient():
    from cuneifit.likelihood import GaussianLikelihood
    import jax
    from jax import numpy as jnp

    class TestLikelihood(GaussianLikelihood):
        parameters = ["a"]
        data_vec = jnp.array([0.0])
        covariance = jnp.array([[1.0]])
        logp_const = 0.0

        def get_model(self, theta, /, **kwargs):
            return theta[0]

    like = TestLikelihood()
    grad = jax.grad(like.loglike)
    assert grad(like.data_vec) == 0.0

