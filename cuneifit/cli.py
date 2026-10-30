import argparse
import logging
import os

import jax  # noqa: F401
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np

import cuneifit  # noqa: F401
from . import config

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="Cuneifit",
        description="Cuneifit --- Differentiable Inference for Simons Observatory"  # noqa: E501
    )
    parser.add_argument("run_file", action="store",
                        help="A run configuration to analyze.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug logging.")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Make plots as you go.")

    args = parser.parse_args(argv)

    if args.plot:
        import matplotlib.pyplot as plt
        import cuneifit.plots

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, *args: x  # noqa: E731

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
    for name, conf in component_config.items():
        if hasattr(cuneifit, name):
            components.append(getattr(cuneifit, name)(conf))

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

    # Check out the starting parameters.
    if len(params_fix := run_config.get("restrict", {})) > 0:
        pipeline.restrict(params_fix)

    logger.info(pipeline.parameters)

    theta = pipeline.params_to_vec(run_config["start"])
    print(theta)

    if (min_args := run_config.get("minimizer", None)):
        logger.info("Optimizing parameters.")
        min_args["max_steps"] = min_args.get("max_steps", 100)
        min_args["learning_rate"] = min_args.get("learning_rate", 0.1)
        chain = pipeline.minimize(theta, tqdm=tqdm, **min_args)
        theta = chain[-1]

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
        fig, axes = cuneifit.plots.plot_fisher_triangle(pipeline, theta, F,
                                                        **plot_args)
        plt.savefig(os.path.join(dirname, "fisher.png"), bbox_inches="tight")

    if (chain_args := run_config.get("chain", None)):
        logger.info("Running chains.")
