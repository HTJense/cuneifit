import pytest  # noqa: F401


def test_import():
    from cuneifit import Pipeline, Prior  # noqa: F401


def test_pipeline_creation():
    from cuneifit import Pipeline
    from utilities import TestLikelihood

    Pipeline([TestLikelihood()])


def test_pipeline_loglike():
    from cuneifit import Pipeline
    from utilities import TestLikelihood

    like = TestLikelihood()
    pipeline = Pipeline([like])
    assert pipeline.loglike(like.data_vec) == 0.0


def test_prior_creation():
    from cuneifit import Prior
    import jax.scipy as jsc

    Prior([
        lambda a: jsc.stats.norm.logpdf(a, loc=0.0, scale=1.0)
    ])


def test_pipeline_prior():
    from cuneifit import Pipeline, Prior
    import jax.scipy as jsc
    from utilities import TestLikelihood

    prior = Prior([lambda a: jsc.stats.norm.logpdf(a, loc=0.0, scale=1.0)])
    like = TestLikelihood()
    Pipeline([like], prior=prior)


def test_pipeline_logpost():
    from cuneifit import Pipeline, Prior
    import jax.scipy as jsc
    from utilities import TestLikelihood

    prior = Prior([lambda a: jsc.stats.norm.logpdf(a, loc=0.0, scale=1.0)])
    like = TestLikelihood()
    pipeline = Pipeline([like], prior=prior)
    target_logpost = jsc.stats.norm.logpdf(0.0, loc=0.0, scale=1.0)

    assert pipeline.logposterior(like.data_vec) == target_logpost


def test_pipeline_logpost_grad():
    from cuneifit import Pipeline, Prior
    import jax
    import jax.scipy as jsc
    from utilities import TestLikelihood

    prior = Prior([lambda a: jsc.stats.norm.logpdf(a, loc=0.0, scale=1.0)])
    like = TestLikelihood()
    pipeline = Pipeline([like], prior=prior)
    grad = jax.grad(pipeline.logposterior)

    assert grad(like.data_vec) == 0.0
