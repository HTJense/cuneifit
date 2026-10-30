import argparse
import logging
import os

import jax
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np

import matplotlib.pyplot as plt
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
        description="Cuneifit --- Differentiable Inference for Simons Observatory"  # noqa: E501
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

        import blackjax
        import secrets
        n_runners = chain_args.get("n_runners", 1)
        n_samples = chain_args["n_samples"]
        method = chain_args.get("method", "hmc")

        if not hasattr(blackjax, method):
            raise RuntimeError(f"Sampling method {method} not found in"
                               "blackjax.")

        M = jnp.linalg.inv(F)
        rng_key = jax.random.key(secrets.randbits(32))

        if (warmup_steps := chain_args.get("warmup", None)) > 0:
            # Do a warmup run.
            logger.info(f"Running {warmup_steps} warm-up steps.")
            args = chain_args.get("args", {})

            warmup = blackjax.window_adaptation(
                getattr(blackjax, method), pipeline.logposterior,
                initial_inverse_mass_matrix=M, is_mass_matrix_diagonal=False,
                **args
            )

            rng_key, warmup_key = jax.random.split(rng_key, 2)
            with blackjax.progress_bar(label="Warm-up"):
                (state, parameters), _ = warmup.run(warmup_key, theta,
                                                    num_steps=warmup_steps)

            M = jnp.asarray(parameters["inverse_mass_matrix"])
            sampler = getattr(blackjax, method)(pipeline.logposterior,
                                                **parameters)
        else:
            # Don't do a warmup run - the user may provide some settings.
            args = chain_args.get("args", {})
            sampler = getattr(blackjax, method)(pipeline.logposterior, M,
                                                **args)
            state = sampler.init(theta)

        if n_runners == 1:
            # Run only a single chain.
            logger.info(f"Starting single {method} sampler.")

            if (scale := chain_args.get("start_spread", 0.0)) > 0.0:
                logger.info(f"Giving sampler {scale} sigma random offset.")
                L = jnp.linalg.cholesky(M)
                init_key, rng_key = jax.random.split(rng_key, 2)
                gn = jax.random.normal(init_key, shape=(state.position.size,))
                theta_start = state.position + scale * (L @ gn)
                state = sampler.init(theta_start)

            step = jax.jit(sampler.step)

            fp = open(output + ".1.txt", "w")
            fp.write("# log-posterior " + " ".join(pipeline.parameters) + "\n")

            for i in tqdm(range(n_samples), desc="Sampling"):
                key = jax.random.fold_in(rng_key, i)
                state, _ = step(key, state)

                fp.write(f"{state.logdensity:.3f}\t"
                         + "\t".join([f"{x:.8e}" for x in state.position])
                         + "\n")
                fp.flush()

            fp.close()
        else:
            # Run multiple chains in parallel using JAX sharding.
            devices = jax.devices()
            n_devices = len(devices)
            if n_runners % n_devices != 0:
                raise ValueError(
                    f"The number of chain runners = {n_runners} must be "
                    "divisible by the number of available devices = "
                    f"{n_devices}."
                )

            logger.info(f"Starting {n_runners} {method} samplers across "
                        f"{n_devices} devices.")

            # Create starting states.
            if (scale := chain_args.get("start_spread", 0.0)) > 0.0:
                logger.info(f"Giving samplers {scale} sigma random offset.")
                L = jnp.linalg.cholesky(M)
                init_key, rng_key = jax.random.split(rng_key, 2)
                gn = jax.random.normal(init_key,
                                       shape=(n_runners, state.position.size))
                theta_start = state.position + scale * (gn @ L.T)
                states = [
                    sampler.init(theta_start[i]) for i in range(n_runners)
                ]
            else:
                states = [
                    sampler.init(state.position) for _ in range(n_runners)
                ]

            states = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
            chain_keys = jax.random.split(rng_key, n_runners)

            # Initialize sharding setup.
            from jax.sharding import NamedSharding, PartitionSpec
            mesh = jax.make_mesh((n_devices,), ("chain",))
            chain_sharding = NamedSharding(mesh, PartitionSpec("chain"))
            chain_keys = jax.device_put(chain_keys, chain_sharding)
            states = jax.tree.map(lambda x: jax.device_put(x, chain_sharding),
                                  states)

            # The running function.
            def run_loop(key, kernel, state, num_samples):
                @jax.jit
                def step(state, key):
                    state, _ = kernel(key, state)
                    return state, (state.logdensity, state.position)

                sample_keys = jax.random.split(key, num_samples)
                _, states = jax.lax.scan(step, state, sample_keys)
                return states

            def shard_chains(keys, local_states):
                return jax.vmap(
                    lambda k, s: run_loop(k, sampler.step, s, n_samples),
                    in_axes=(0, 0),
                )(keys, local_states)

            state_spec = jax.tree.map(
                lambda _: PartitionSpec("chain"), states
            )

            parallel_sampling = jax.jit(
                jax.shard_map(
                    shard_chains, mesh=mesh,
                    in_specs=(PartitionSpec("chain"), state_spec),
                    out_specs=(
                        PartitionSpec("chain", None),
                        PartitionSpec("chain", None, None),
                    ),
                    check_vma=False,
                )
            )

            with blackjax.progress_bar(label="Sampling"):
                logdensities, positions = parallel_sampling(chain_keys, states)
                positions.block_until_ready()

            logdensities = np.asarray(logdensities)
            positions = np.asarray(positions)

            logger.info("Done. Writing to files.")

            for i in range(n_runners):
                np.savetxt(
                    output + f".{i+1}.txt",
                    np.c_[logdensities[i], positions[i]],
                    fmt=["%.3f"] + ["%.8e"] * len(pipeline.parameters),
                    header="log-posterior " + " ".join(pipeline.parameters)
                )

    logger.info("Done.")
