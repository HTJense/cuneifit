import argparse
import jax.numpy as jnp
import jax.scipy as jsc
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import cuneifit
import cuneifit.plots
from . import config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args: x  # noqa: E731


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="Cuneifit",
        description=("Cuneifit --- Differentiable Inference "
                     "for Simons Observatory")
    )
    parser.add_argument("run_file", action="store",
                        help="A run configuration to analyze.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug logging.")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Make plots as you go.")

    args = parser.parse_args(argv)

    # Setup the logger
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    # Load the configuration file.
    run_config = config.load_config(args.run_file)
    plot_args = run_config.get("plot_args", {})

    # Prepare output directory.
    output = run_config["output"]
    if not os.path.exists(dirname := os.path.dirname(output)):
        logger.info(f"Creating output directory {output}.")
        os.makedirs(dirname, exist_ok=True)

    # Build the pipeline.
    components = []
    component_config = run_config["components"]
    for i, entry in enumerate(component_config):
        for name, conf in entry.items():
            try:
                comp = getattr(cuneifit, name)(conf)
                components.append(comp)
                run_config["components"][i][name] = comp.config
            except AttributeError:
                raise RuntimeError("Failed to find cuneifit component "
                                   f"{name}.")

    config.save_config(output + ".config.yaml", run_config)

    # Add a prior if requested.
    priors = []
    for prior in run_config.get("priors", []):
        params = prior["par"]
        if type(params) is str:
            params = [params]
        dist = prior.get("dist", "norm")

        if hasattr(jsc.stats, dist):
            settings = {
                k: v for k, v in prior.items() if k not in ["dist", "par"]
            }

            logpi = lambda x, dist=dist, settings=settings: (  # noqa: E731
                getattr(jsc.stats, dist).logpdf(x, **settings)
            )
            priors.append((params, logpi))
        else:
            raise RuntimeError(f"Unknown probability function {dist}.")

    if len(priors) > 0:
        prior = cuneifit.Prior()
        for params, logpi in priors:
            prior.add_prior(logpi, func_pars=params)
        components.append(prior)

    pipeline = cuneifit.Pipeline(*components)
    pipeline.jeffreys = run_config.get("jeffreys", False)
    pipeline.add_bounds(run_config.get("bounds", {}))

    # Set up the parameter restrictions.
    restrict = pipeline.get_default_restrictions()
    for par in run_config["start"]:
        if par in restrict:
            restrict.pop(par)

    restrict.update((user_restrict := run_config.get("restrict", {})))
    for par in run_config["start"]:
        if par in user_restrict:
            raise RuntimeError(f"Parameter {par} is present in both the "
                               "`restrict` block and the `start` block: must "
                               "be one or the other!")

    pipeline.restrict(restrict, include_defaults=False)

    logger.info(pipeline.parameters)

    # See if all the parameters are given at the start.
    params_missing = [
        par for par in pipeline.parameters if par not in run_config["start"]
    ]
    if len(params_missing) > 0:
        raise RuntimeError(f"Missing values for parameters {params_missing}.")

    theta = pipeline.params_to_vec(run_config["start"])

    if (min_args := run_config.get("minimizer")):
        logger.info("Optimizing parameters.")

        num_runs = min_args.get("runners", 1)
        method = min_args.get("method", "adam")
        patience = min_args.get("patience", 10 + len(theta) * 5)
        min_steps = min_args.get("min_steps", 1)
        max_steps = min_args.get("max_steps", 100 + len(theta) * 50)
        start_spread = min_args.get("start_spread", 1.0)
        rtol = min_args.get("rtol", 0.0)
        atol = min_args.get("atol", 1e-2)
        gtol = min_args.get("gtol", 0.0)
        opt_args = min_args.get("args", {})

        logp, thetas = pipeline.minimize(
            theta, num_runs=num_runs, method=method, patience=patience,
            min_steps=min_steps, max_steps=max_steps,
            start_spread=start_spread, rtol=rtol, atol=atol, gtol=gtol,
            tqdm=tqdm, opt_args=opt_args,
        )

        logp = np.asarray(logp)
        thetas = np.asarray(thetas)
        np.savetxt(output + ".minima", np.c_[logp, thetas],
                   header="log-posterior " + " ".join(pipeline.parameters),
                   fmt=["%.5f"] + ["%.5e"] * len(pipeline.parameters))

        theta = thetas[0]

        logprior = pipeline.logprior(theta)
        loglike = pipeline.loglike(theta)
        logpost = pipeline.logposterior(theta)

        F = pipeline.fisher(theta)
        cov = jnp.linalg.inv(F)
        err = jnp.sqrt(jnp.diag(cov))

        for i, p in enumerate(pipeline.parameters):
            logger.info(f"{p} = {theta[i]:.3e} +/- {err[i]:.3e}")

        with open(output + ".minimum", "w") as fp:
            fp.write(f"    -log(prior)      {-logprior:.5f}\n")
            fp.write(f"    -log(likelihood) {-loglike:.5f}\n")
            fp.write(f"    -log(posterior)  {-logpost:.5f}\n\n")
            for i, p in enumerate(pipeline.parameters):
                fp.write(f"{p:>20s} {theta[i]:.5e}\n")
    else:
        F = pipeline.fisher(theta)
        cov = jnp.linalg.inv(F)

    np.savetxt(output + ".fisher_cov", cov,
               header=" ".join(pipeline.parameters), fmt="%.8e")

    if args.plot:
        cuneifit.plots.plot_fisher_triangle(pipeline, theta, F, **plot_args)
        plt.savefig(output + "_fisher.pdf", bbox_inches="tight")

    if (chain_args := run_config.get("chain")):
        logger.info("Running chains.")

        n_runners = chain_args.get("runners", 2)
        n_samples = chain_args.get("samples")
        method = chain_args.get("method", "nuts")

        start_spread = chain_args.get("start_spread", 0.0)
        warmup_steps = chain_args.get("warmup", 0)
        kernel_args = chain_args.get("args", {})

        converge_target = chain_args.get("converge_target")
        check_every = chain_args.get("check_every")
        prior_samples = chain_args.get("prior_samples", 0)

        pipeline.sample(
            theta, num_steps=n_samples, num_samplers=n_runners, method=method,
            warmup_steps=warmup_steps, kernel_args=kernel_args,
            converge_target=converge_target, check_every=check_every,
            start_spread=start_spread, tqdm=tqdm, output=output,
            plot=args.plot, prior_samples=prior_samples,
        )

    logger.info("Done.")
