import pytest  # noqa: F401
import jax.numpy as jnp
import jax.scipy as jsc

from cuneifit.likelihood import Likelihood


# A mock likelihood used to test only prior contributions.
class FlatLikelihood(Likelihood):
    parameters = ["a", "b", "c"]

    def get_model(self, theta, /, **kwargs):
        return 0.0

    def loglike(self, theta, /, **kwargs):
        return 0.0

    def fisher(self, theta, /, **kwargs):
        return jnp.zeros((len(self.parameters), len(self.parameters)))


def test_lambdas():
    from cuneifit import Prior

    p_a = Prior(lambda a: jsc.stats.norm.logpdf(a, loc=0.0, scale=1.0))
    p_b = Prior(lambda b: jsc.stats.norm.logpdf(b, loc=0.0, scale=2.0))

    assert p_a.parameters == ["a"]
    assert p_b.parameters == ["b"]

    F_a = p_a.fisher(jnp.array([0.0]))
    assert F_a[0, 0] == 1.0
    F_b = p_b.fisher(jnp.array([0.0]))
    assert F_b[0, 0] == 0.25


def test_addition():
    from cuneifit import Prior

    p_a = Prior(lambda a: jsc.stats.norm.logpdf(a, loc=0.0, scale=1.0))
    p_b = Prior(lambda b: jsc.stats.norm.logpdf(b, loc=0.0, scale=2.0))

    p_ab = p_a + p_b

    assert set(p_ab.parameters) == set(p_a.parameters + p_b.parameters)

    F_a = p_a.fisher(jnp.array([0.0]))
    F_b = p_b.fisher(jnp.array([0.0]))
    F_ab = p_ab.fisher(jnp.array([0.0, 0.0]))

    i_a = p_ab.parameters.index("a")
    i_b = p_ab.parameters.index("b")

    assert F_ab[i_a,i_a] == F_a[0, 0]
    assert F_ab[i_b,i_b] == F_b[0, 0]

