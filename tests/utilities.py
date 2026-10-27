import jax  # noqa: F401
import jax.numpy as jnp
import jax.scipy as jsc

from cuneifit import Prior
from cuneifit.likelihood import GaussianLikelihood


test_likelihood_data = jnp.array([1.0])
test_likelihood_cov = jnp.array([[3.0]])
test_likelihood_inv_cov = jnp.linalg.inv(test_likelihood_cov)
test_likelihood_logp = 0.0

test_prior_mean = 0.0
test_prior_scale = 1.0

test_sample_data = jnp.array([1.0])
test_sample_chi2 = 0.0
test_sample_loglike = 0.0
test_sample_loglike_grad = jnp.array([0.0])
test_sample_loglike_hess = -test_likelihood_inv_cov

test_sample_logprior = jsc.stats.norm.logpdf(test_sample_data,
                                             loc=test_prior_mean,
                                             scale=test_prior_scale)
test_sample_logprior_hess = -1. / test_prior_scale ** 2
test_sample_logprior_grad = (test_sample_data - test_prior_mean) \
                            * test_sample_logprior_hess

test_sample_logpost = test_sample_loglike + test_sample_logprior
test_sample_logpost_grad = test_sample_loglike_grad + test_sample_logprior_grad
test_sample_logpost_hess = test_sample_loglike_hess + test_sample_logprior_hess

test_cosmopower_parameters = {
    "ombh2": 0.022,
    "omch2": 0.117,
    "h": 0.67,
    "ns": 0.96,
    "logA": 3.044,
    "tau": 0.06,
}


class TestLikelihood(GaussianLikelihood):
    parameters = ["a"]
    data_vec = test_likelihood_data
    covariance = test_likelihood_cov
    logp_const = test_likelihood_logp

    def get_model(self, theta, /, **kwargs):
        return theta[0]


def TestPrior():
    return Prior([
        lambda a: jsc.stats.norm.logpdf(a, loc=test_prior_mean,
                                        scale=test_prior_scale)
    ])
