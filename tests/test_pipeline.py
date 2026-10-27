import pytest  # noqa: F401


def test_import():
    """Test whether the Pipeline wants to import.
    """
    from cuneifit import Pipeline, Prior  # noqa: F401


def test_pipeline_creation():
    """Test whether we can create a pipeline.
    """
    from cuneifit import Pipeline
    from utilities import TestLikelihood

    Pipeline([TestLikelihood()])


def test_pipeline_loglike():
    """Test whether a pipeline agrees with a Likelihood
    """
    from cuneifit import Pipeline
    from utilities import TestLikelihood

    like = TestLikelihood()
    pipeline = Pipeline([like])
    assert pipeline.loglike(like.data_vec) == 0.0


def test_prior_creation():
    """Test whether we can create a prior
    """
    import jax.scipy as jsc
    from cuneifit import Prior

    Prior([
        lambda a: jsc.stats.norm.logpdf(a, loc=0.0, scale=1.0)
    ])


def test_pipeline_prior():
    """Test whether we can create a pipeline with a prior
    """
    from cuneifit import Pipeline, Prior
    import jax.scipy as jsc
    import utilities

    prior = Prior([lambda a: jsc.stats.norm.logpdf(a, loc=0.0, scale=1.0)])
    like = utilities.TestLikelihood()
    Pipeline([like], prior=prior)


def test_pipeline_logpost():
    """Test whether the pipeline's functions are sane.
    """
    from cuneifit import Pipeline
    import utilities
    import jax.numpy as jnp

    prior = utilities.TestPrior()
    like = utilities.TestLikelihood()
    pipeline = Pipeline([like], prior=prior)

    assert jnp.isclose(
        pipeline.logprior(utilities.test_sample_data),
        utilities.test_sample_logprior
    ), "Log-prior did not match"
    assert jnp.isclose(
        pipeline.loglike(utilities.test_sample_data),
        utilities.test_sample_loglike
    ), "Log-likelihood did not match"
    assert jnp.isclose(
        pipeline.logposterior(utilities.test_sample_data),
        utilities.test_sample_logpost
    ), "Log-posterior did not match"


def test_pipeline_logpost_grad():
    """Test whether the pipeline's gradients are sane.
    """
    from cuneifit import Pipeline
    import jax
    import jax.numpy as jnp
    import utilities

    prior = utilities.TestPrior()
    like = utilities.TestLikelihood()
    pipeline = Pipeline([like], prior=prior)

    grad_prior = jax.grad(pipeline.logprior)
    grad_like = jax.grad(pipeline.loglike)
    grad_post = jax.grad(pipeline.logposterior)

    assert jnp.isclose(grad_prior(utilities.test_sample_data),
                       utilities.test_sample_logprior_grad)
    assert jnp.isclose(grad_like(utilities.test_sample_data),
                       utilities.test_sample_loglike_grad)
    assert jnp.isclose(grad_post(utilities.test_sample_data),
                       utilities.test_sample_logpost_grad)


def test_pipeline_logpost_hess():
    """Test whether the pipeline's hessians are sane.
    """
    from cuneifit import Pipeline
    import jax
    import jax.numpy as jnp
    import utilities

    prior = utilities.TestPrior()
    like = utilities.TestLikelihood()
    pipeline = Pipeline([like], prior=prior)

    hess_prior = jax.hessian(pipeline.logprior)
    hess_like = jax.hessian(pipeline.loglike)
    hess_post = jax.hessian(pipeline.logposterior)

    assert jnp.isclose(hess_prior(utilities.test_sample_data),
                       utilities.test_sample_logprior_hess)
    assert jnp.isclose(hess_like(utilities.test_sample_data),
                       utilities.test_sample_loglike_hess)
    assert jnp.isclose(hess_post(utilities.test_sample_data),
                       utilities.test_sample_logpost_hess)
