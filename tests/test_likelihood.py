import pytest  # noqa: F401


def test_import():
    """Test whether the likelihood classes import.
    """
    from cuneifit.likelihood import Likelihood  # noqa: F401
    from cuneifit.likelihood import GaussianLikelihood  # noqa: F401


def test_inheritance():
    """Test whether we can make a simple subclass of a Likelihood.
    """
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
    """Test whether the chi square function has sane behaviour.
    """
    import utilities
    import jax.numpy as jnp

    like = utilities.TestLikelihood()
    assert jnp.isclose(like.chisquare(utilities.test_sample_data),
                       utilities.test_sample_chi2)


def test_gradient():
    """Test whether the gradient of the log-likelihood has sane behaviour.
    """
    import jax
    import jax.numpy as jnp
    import utilities

    like = utilities.TestLikelihood()
    grad = jax.grad(like.loglike)
    assert jnp.isclose(grad(utilities.test_sample_data),
                       utilities.test_sample_loglike_grad)


def test_hessian():
    """Test whether the hessian of the log-likelihood has sane behaviour.
    """
    import jax
    import jax.numpy as jnp
    import utilities

    like = utilities.TestLikelihood()
    hess = jax.hessian(like.loglike)
    assert jnp.isclose(hess(utilities.test_sample_data),
                       utilities.test_sample_loglike_hess)
