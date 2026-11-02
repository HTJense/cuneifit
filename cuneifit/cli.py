import argparse
import blackjax
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
    bestfit = None

    if (min_args := run_config.get("minimizer", None)):
        logger.info("Optimizing parameters.")

        num_runs = min_args.get("runners", 1)
        method = min_args.get("method", "adam")
        patience = min_args.get("patience", 1)
        max_steps = min_args.get("max_steps", 10 + len(theta) * 5)
        start_spread = min_args.get("start_spread", 0.2)
        rtol = min_args.get("rtol", 1e-2)
        atol = min_args.get("atol", 1e-2)
        opt_args = min_args.get("args", {})

        logp, thetas = pipeline.minimize(theta, num_runs=num_runs, method=method,
                                         patience=patience, max_steps=max_steps,
                                         start_spread=start_spread, rtol=rtol,
                                         atol=atol, tqdm=tqdm, opt_args=opt_args)

        logp = np.asarray(logp)
        thetas = np.asarray(thetas)
        np.savetxt(output + ".minima", np.c_[logp, thetas], header = "log-posterior " + " ".join(pipeline.parameters), fmt=["%.5f"] + ["%.5e"] * len(pipeline.parameters))

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

        bestfit = theta.copy()
    else:
        F = pipeline.fisher(theta)
        cov = jnp.linalg.inv(F)

    np.savetxt(output + ".fisher_cov", cov,
               header=" ".join(pipeline.parameters), fmt="%.8e")

    if args.plot:
        cuneifit.plots.plot_fisher_triangle(pipeline, theta, F, **plot_args)
        plt.savefig(os.path.join(dirname, "fisher.pdf"), bbox_inches="tight")

    if (chain_args := run_config.get("chain", None)):
        logger.info("Running chains.")

        n_runners = chain_args.get("runners", 1)
        n_samples = chain_args["samples"]
        method = chain_args.get("method", "hmc")

        if not hasattr(blackjax, method):
            raise RuntimeError(f"Sampling method {method} not found in"
                               "blackjax.")

        start_spread = chain_args.get("start_spread", 0.0)
        warmup_steps = chain_args.get("warmup", 0)
        kernel_args = chain_args.get("args", {})

        logdensities, positions = pipeline.sample(
            theta, num_steps=n_samples, num_samplers=n_runners, method=method,
            warmup_steps=warmup_steps, kernel_args=kernel_args,
            start_spread=start_spread, tqdm=tqdm)

        logger.info("Writing chains to files.")

        for i in range(n_runners):
            np.savetxt(
                output + f".{i+1}.txt",
                np.c_[logdensities[i], positions[i]], delimiter="\t",
                fmt=["%.5f"] + ["%.5e"] * len(pipeline.parameters),
                header="log-posterior " + " ".join(pipeline.parameters)
            )

        if args.plot:
            # Create getdist triangle + 1D plots.
            logger.info("Creating posterior plots.")

            from getdist import plots, MCSamples

            settings = plots.GetDistPlotSettings()
            getdist_settings = run_config.get("getdist_settings", {})
            for k, v in getdist_settings.items():
                if hasattr(settings, k):
                    setattr(settings, k, v)

            labels = [plot_args.get("labels", {}).get(p, p)
                      for p in pipeline.parameters]
            chain = MCSamples(samples=np.asarray(positions),
                              loglikes=-np.asarray(logdensities),
                              names=pipeline.parameters, labels=labels)
            markers = {}
            if bestfit is not None:
                markers = {
                    p: bestfit[i] for i, p in enumerate(pipeline.parameters)
                }

            if (burn_in := getdist_settings.get("burn_in", 0.0)) > 0.0:
                chain.removeBurn(burn_in)

            g = plots.get_subplot_plotter(settings=settings)
            g.plots_1d([chain], markers=markers)
            g.export(os.path.join(dirname, "posteriors.pdf"))

            g = plots.get_subplot_plotter(settings=settings)
            g.triangle_plot([chain], filled=True, markers=markers)
            g.export(os.path.join(dirname, "triangle.pdf"))

    logger.info("Done.")
