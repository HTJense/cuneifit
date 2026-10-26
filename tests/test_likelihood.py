import pytest  # noqa: F401


def test_import():
    from cuneifit.likelihood import Likelihood  # noqa: F401
    from cuneifit.likelihood import GaussianLikelihood  # noqa: F401


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
    from utilities import TestLikelihood
    from jax import numpy as jnp

    like = TestLikelihood()
    assert like.chisquare(like.data_vec) == 0.0


def test_gradient():
    import jax
    from utilities import TestLikelihood

    like = TestLikelihood()
    grad = jax.grad(like.loglike)
    assert grad(like.data_vec) == 0.0
