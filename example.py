import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsc
import dflike
import camb

""" One could use Cosmopower-JAX here for cosmology, but I'll keep cosmology
    fixed for this example. """

cosmo_params = {
    "ombh2": 0.0225,
    "omch2": 0.1193,
    "cosmomc_theta": 104.073e-4,
    "ns": 0.9709,
    "logA": 3.056,
    "tau": 0.0603
}

cosmo_params["As"] = 1e-10 * np.exp(cosmo_params.pop("logA"))

pars = camb.set_params(lens_potential_accuracy=8, lmax=8500, **cosmo_params)
res = camb.get_results(pars)
spec = res.get_cmb_power_spectra(CMB_unit="muK")["total"]

cls = {
    "tt": spec[:, 0],
    "te": spec[:, 3],
    "ee": spec[:, 1]
}

""" We can initialize the likelihood and foreground model with the exact same
    MFLike yaml files as the official MFLike likelihood. """
like = dflike.MultiFrequency("act_dr6")
fg_model = dflike.BandpowerForegrounds("act_dr6_foregrounds", like)

""" These parameters are (approximately) the ACT best-fitting foreground params
    from Louis et al (2025). """

fg_params = {
    "a_tSZ": 3.4,
    "alpha_tSZ": -0.5,
    "a_kSZ": 1.5,
    "a_c": 3.7,
    "a_p": 7.7,
    "beta_cib": 1.9,
    "xi": 0.09,
    "a_s": 2.9,
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
}

sys_params = {
    "calG_all": 1.0,

    "cal_dr6_pa4_f220": 1.0,
    "cal_dr6_pa5_f090": 1.0,
    "cal_dr6_pa5_f150": 1.0,
    "cal_dr6_pa6_f090": 1.0,
    "cal_dr6_pa6_f150": 1.0,

    "calE_dr6_pa4_f220": 1.0,
    "calE_dr6_pa5_f090": 1.0,
    "calE_dr6_pa5_f150": 1.0,
    "calE_dr6_pa6_f090": 1.0,
    "calE_dr6_pa6_f150": 1.0,

    "bandint_shift_dr6_pa4_f220": 0.0,
    "bandint_shift_dr6_pa5_f090": 0.0,
    "bandint_shift_dr6_pa5_f150": 0.0,
    "bandint_shift_dr6_pa6_f090": 0.0,
    "bandint_shift_dr6_pa6_f150": 0.0,
}

params_values = fg_params | sys_params

model_params = like.parameters + fg_model.parameters

priors = dflike.Prior([
    lambda calG_all: jsc.stats.norm.logpdf(calG_all, loc=1.0, scale=0.003),
    lambda cal_dr6_pa4_f220: jsc.stats.norm.logpdf(cal_dr6_pa4_f220, loc=1.0,
                                                   scale=0.013),
    lambda cal_dr6_pa5_f090: jsc.stats.norm.logpdf(cal_dr6_pa5_f090, loc=1.0,
                                                   scale=0.0016),
    lambda cal_dr6_pa5_f150: jsc.stats.norm.logpdf(cal_dr6_pa5_f150, loc=1.0,
                                                   scale=0.0020),
    lambda cal_dr6_pa6_f090: jsc.stats.norm.logpdf(cal_dr6_pa6_f090, loc=1.0,
                                                   scale=0.0018),
    lambda cal_dr6_pa6_f150: jsc.stats.norm.logpdf(cal_dr6_pa6_f150, loc=1.0,
                                                   scale=0.0024),

    lambda bandint_shift_dr6_pa4_f220:
        jsc.stats.norm.logpdf(bandint_shift_dr6_pa4_f220, loc=0.0, scale=3.6),
    lambda bandint_shift_dr6_pa5_f090:
        jsc.stats.norm.logpdf(bandint_shift_dr6_pa5_f090, loc=0.0, scale=1.0),
    lambda bandint_shift_dr6_pa5_f150:
        jsc.stats.norm.logpdf(bandint_shift_dr6_pa5_f150, loc=0.0, scale=1.3),
    lambda bandint_shift_dr6_pa6_f090:
        jsc.stats.norm.logpdf(bandint_shift_dr6_pa6_f090, loc=0.0, scale=1.2),
    lambda bandint_shift_dr6_pa6_f150:
        jsc.stats.norm.logpdf(bandint_shift_dr6_pa6_f150, loc=0.0, scale=1.1),

    lambda a_gtt: jsc.stats.norm.logpdf(a_gtt, loc=7.95, scale=0.32),
    lambda a_gte: jsc.stats.norm.logpdf(a_gte, loc=0.432, scale=0.03),
    lambda a_gee: jsc.stats.norm.logpdf(a_gee, loc=0.1681, scale=0.017),
])

""" Define our log-likelihood and log-prior functions. """
prior_idx = jnp.array([model_params.index(p) for p in priors.parameters])


@jax.jit
def logprior(theta):
    """ Note that this function is not normalized. """
    return priors.logprior(theta[prior_idx])


@jax.jit
def loglike(theta):
    theta_like = theta[:len(like.parameters)]
    theta_fg = theta[len(like.parameters):]
    foregrounds = fg_model.get_foreground_model(theta_fg)
    logl = -0.5 * like.chisquare(theta_like, cls=cls, foregrounds=foregrounds)
    return logl


@jax.jit
def logpost(theta):
    return logprior(theta) + loglike(theta)


""" These are the parameters with respect to which we want to
    differentiate. """
parameters_baseline = [
    "a_tSZ", "a_kSZ", "alpha_tSZ", "xi",
    "a_c", "beta_cib", "a_p", "a_s", "beta_s",
    "a_gtt", "a_gte", "a_gee", "a_pste", "a_psee",
    "calG_all",
    "cal_dr6_pa4_f220", "cal_dr6_pa5_f090", "cal_dr6_pa5_f150",
    "cal_dr6_pa6_f090", "cal_dr6_pa6_f150",
    "calE_dr6_pa5_f090", "calE_dr6_pa5_f150",
    "calE_dr6_pa6_f090", "calE_dr6_pa6_f150",
    "bandint_shift_dr6_pa4_f220", "bandint_shift_dr6_pa5_f090",
    "bandint_shift_dr6_pa5_f150", "bandint_shift_dr6_pa6_f090",
    "bandint_shift_dr6_pa6_f150",
]

""" We need a log-posterior function that only takes in baseline parameters and
    keeps the remainder fixed. We create a default list of parameters, and then
    keep track of the indices of the free parameters. """
fit_idx = jnp.array([model_params.index(p) for p in parameters_baseline])
theta_default = jnp.array([params_values[p] for p in model_params])


@jax.jit
def logpost_baseline(theta_x):
    """ Note that in numpy we could write `theta[fit_idx] = theta_x`
        But in jax, arrays are immutable, so we need to use this
        `theta = theta.at[fit_idx].set(theta_x)`. """
    theta = theta_default.at[fit_idx].set(theta_x)
    return logpost(theta)


theta_baseline = jnp.array([params_values[p] for p in parameters_baseline])

""" We can simply evaluate these functions directly: """
print(f"log(prior)        = {logprior(theta_default):>9.4f}")
print(f"log(like)         = {loglike(theta_default):>9.4f}")
print(f"log(post)         = {logpost(theta_default):>9.4f}")

""" Or we can do some jax things with them: """
grad = jax.jit(jax.grad(logpost_baseline))
hess = jax.jit(jax.hessian(logpost_baseline))

""" Example: find the best-fitting point with OPTAX. """
import optax  # noqa: E402
import tqdm  # noqa: E402

grad_adam = jax.jit(jax.grad(lambda x: -logpost_baseline(x)))
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(theta_baseline)
theta_opt = theta_baseline.copy()

for _ in tqdm.tqdm(range(100)):
    updates, opt_state = optimizer.update(grad_adam(theta_opt), opt_state,
                                          theta_opt)
    theta_opt = optax.apply_updates(theta_opt, updates)

print(f"log(post) minimum = {logpost_baseline(theta_opt):>9.4f}")

""" These two are functions, we can evaluate them at the same values and find
    derivatives: """
H = hess(theta_opt)

""" Fisher estimate of the error. """
cov = jnp.linalg.inv(-H)
err = np.sqrt(np.diag(cov))

for i, par in enumerate(parameters_baseline):
    print(f"{par:>30s} = {theta_opt[i]:8.5f} +/- " +
          (f"{err[i]:4.2e}" if err[i] < 0.01 else f"{err[i]:4.2f}"))
