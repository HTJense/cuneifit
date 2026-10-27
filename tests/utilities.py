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


test_cosmopower_parameters = {
    "ombh2": 0.022,
    "omch2": 0.117,
    "h": 0.67,
    "logA": 3.05,
    "ns": 0.97,
    "tau": 0.06,
}

test_dr6_parameters = test_cosmopower_parameters | {
    "a_tSZ": 3.4,
    "alpha_tSZ": -0.5,
    "a_kSZ": 1.5,
    "a_c": 3.7,
    "a_p": 7.7,
    "beta_cib": 1.9,
    "xi": 0.1,
    "a_s": 2.8,
    "beta_s": -2.8,
    "a_pste": -0.025,
    "a_psee": 0.02,

    "a_gtt": 7.95,
    "a_gte": 0.423,
    "a_gee": 0.1681,

    "alpha_dT": -0.6,
    "alpha_dE": -0.4,
    "alpha_s": 1.0,
    "T_effd": 19.6,
    "T_d": 9.7,
    "beta_d": 1.5,
    "alpha_p": 1.0,
    "alpha_c": 0.8,

    "calG_all": 1.0,

    "cal_dr6_pa4_f220": 0.98,
    "cal_dr6_pa5_f090": 1.0,
    "cal_dr6_pa5_f150": 1.0,
    "cal_dr6_pa6_f090": 1.0,
    "cal_dr6_pa6_f150": 1.0,

    "calE_dr6_pa4_f220": 1.0,
    "calE_dr6_pa5_f090": 1.0,
    "calE_dr6_pa5_f150": 1.0,
    "calE_dr6_pa6_f090": 1.0,
    "calE_dr6_pa6_f150": 1.0,

    "bandint_shift_dr6_pa4_f220": 5.0,
    "bandint_shift_dr6_pa5_f090": 1.0,
    "bandint_shift_dr6_pa5_f150": 0.0,
    "bandint_shift_dr6_pa6_f090": -1.0,
    "bandint_shift_dr6_pa6_f150": 0.0,
}

test_dr6_lite_parameters = test_cosmopower_parameters | {
    "calG": 1.0,
    "calT": 1.0,
    "calP": 0.95,
}

test_dr6_logpost = -3214.1987
test_dr6_chi2 = 2138.0544

test_dr6_lite_logpost = -671.5527
test_dr6_lite_chi2 = 1112.5648
