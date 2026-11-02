import jax
from jax.sharding import NamedSharding, PartitionSpec
import jax.numpy as jnp
import jax.scipy as jsc
import logging
import numpy as np
from typing import Callable, Optional

from .likelihood import Likelihood, GaussianLikelihood
from .prior import Prior
from .theory import Theory


logger = logging.getLogger(__name__)


class Pipeline:
    """ Posterior pipeline that automatically manages parameter and product
    dependencies.
    """
    def __init__(self, *components):
        """ Create a pipeline object.

        Parameters
        ----------
        components : list
            A list of components (i.e. Theory and Likelihood objects).

        prior : Optional[Prior]
            A prior contribution object.
        """
        used_components = []
        for comp in components:
            if (isinstance(comp, Prior) or isinstance(comp, Likelihood)
               or isinstance(comp, Theory)):
                used_components.append(comp)
            else:
                raise ValueError(f"What am I to do with {comp}?")

        components = used_components
        self.components = components
        self.priors = [comp for comp in self.components
                       if isinstance(comp, Prior)]
        self.theories = [comp for comp in self.components
                         if isinstance(comp, Theory)]
        self.likelihoods = [comp for comp in self.components
                            if isinstance(comp, Likelihood)]

        self.parameters_full = list(dict.fromkeys(
            p for comp in self.components for p in comp.parameters
        ))
        self.parameters = self.parameters_full.copy()
        self.theta_default = jnp.array([jnp.nan for _ in self.parameters_full])
        self.free_to_full = jnp.array(range(len(self.parameters)), dtype=int)

        self.prior = None
        for prior in self.priors:
            self.prior = prior + self.prior

        self.prior_indices = jnp.array([])
        self.prior_free_indices = jnp.array([])
        if self.prior is not None:
            self.prior_indices = jnp.array([
                self.parameters_full.index(i) for i in self.prior.parameters
            ], dtype=int)
            self.prior_free_indices = self.prior_indices.copy()

        self.th_indices = [
            jnp.array([
                self.parameters_full.index(i) for i in th.parameters
            ], dtype=int) for th in self.theories
        ]

        self.like_indices = [
            jnp.array([
                self.parameters_full.index(p) for p in like.parameters
            ], dtype=int) for like in self.likelihoods
        ]
        self.like_free_indices = [
            idx.copy() for idx in self.like_indices
        ]

        self.jacobians = None
        self.models = None

    def restrict(self, theta_fixed: dict):
        """ Mark any set of parameters as "fixed" for gradient computation.

        Sometimes, a set of parameters is degenerate, or too complicated, or
        unnecessary for the computation of the full posterior. In this case,
        *fixing* these parameters will make the Pipeline not request their
        values, and instead use the given fixed values from the last call
        to `restrict`.

        Parameters
        ----------
        theta_fixed : dict
            A dictionary of name-value pairs for parameters that should be kept
            fixed in this analysis.
        """
        self.parameters = self.parameters_full.copy()
        for p in theta_fixed:
            self.parameters.remove(p)

        self.theta_default = jnp.array([
            theta_fixed.get(p, jnp.nan) for p in self.parameters_full
        ])
        self.free_to_full = jnp.array([
            self.parameters_full.index(p) for p in self.parameters
        ], dtype=int)
        self.like_free_indices = [
            jnp.array([
                like.parameters.index(p)
                for p in like.parameters
                if p in self.parameters
            ], dtype=int)
            for like in self.likelihoods
        ]
        if self.prior is not None:
            self.prior_free_indices = jnp.array([
                self.parameters.index(p)
                for p in self.prior.parameters
                if p in self.parameters
            ], dtype=int)

        # Clear the Jacobians cache (they should be re-built).
        self.jacobians = None
        self.models = None

    def params_to_vec(self, params: dict) -> jnp.ndarray:
        """ Create a parameter vector from a dictionary.

        Parameters
        ----------
        params : dict
            A dictionary of parameters. The result is a vector that can be
            used directly as `theta` in other functions.
        """
        return jnp.array([params[p] for p in self.parameters])

    def vec_to_params(self, theta: jnp.ndarray) -> dict:
        """ Create a vector into a dictionary of parameters.

        Parameters
        ----------
        theta : jnp.ndarray
            A vector of values to be turned into a dictionary. The result is a
            dictionary with key-value pairs.
        """
        return {p: theta[i] for i, p in enumerate(self.parameters)}

    def expand(self, theta_free: jnp.ndarray) -> jnp.ndarray:
        if theta_free.shape == self.theta_default.shape:
            # This vector has already been expanded, so don't do it again!
            return theta_free

        return self.theta_default.at[self.free_to_full].set(theta_free)

    def compute(self, theta: jnp.ndarray | dict) -> dict:
        """ Iterate through all Theory objects and compute their products.

        Parameters
        ----------
        theta : jnp.ndarray | dict
            A list of scalar parameters used by this function. The meaning of
            each entry should match the name given by this class' `parameters`
            member. If presented as a dictionary of key-value pairs, this
            internally converts it
        """
        theta_full = self.expand(theta)
        products = {}
        left_to_compute = list(enumerate(self.theories))
        while len(left_to_compute) > 0:
            computed_something = False
            for i, th in left_to_compute:
                if all([x in products for x in th.inputs]):
                    th_theta = theta_full[self.th_indices[i]]
                    products |= th.compute(th_theta, **products)
                    left_to_compute.remove((i, th))
                    computed_something = True
                    break

            if not computed_something and len(left_to_compute) > 0:
                unresolved = []
                for name, th in left_to_compute:
                    missing = [x for x in th.inputs if x not in products]
                    if missing:
                        unresolved += f"{name} requires {', '.join(missing)}"
                raise RuntimeError("Failed to compute everything - "
                                   "Did you check your theory dependencies?"
                                   "Unresolved dependencies: "
                                   + "; ".join(unresolved))

        return products

    def logprior(self, theta: jnp.ndarray) -> float:
        """ Compute the log-prior contribution.

        Parameters
        ----------
        theta : jnp.ndarray
            A list of scalar parameters used by this function. The meaning of
            each entry should match the name given by this class' `parameters`
            member.
        """
        logpi = 0.0
        if self.prior is not None:
            theta_full = self.expand(theta)
            logpi = self.prior.logprior(theta_full[self.prior_indices])

        return logpi

    def loglike(self, theta: jnp.ndarray) -> float:
        """ Compute all likelihood values and yield the log-likelihood.

        Parameters
        ----------
        theta : jnp.ndarray
            A list of scalar parameters used by this function. The meaning of
            each entry should match the name given by this class' `parameters`
            member.
        """
        theta_full = self.expand(theta)
        products = self.compute(theta)

        loglike = jnp.sum(jnp.array([
            like.loglike(theta_full[idx], **products)
            for idx, like in zip(self.like_indices, self.likelihoods)
        ]))

        return loglike

    def logposterior(self, theta: jnp.ndarray) -> float:
        """ Alias for `logprior(theta) + loglike(theta)`.

        Parameters
        ----------
        theta : jnp.ndarray
            A list of scalar parameters used by this function. The meaning of
            each entry should match the name given by this class' `parameters`
            member.
        """
        return self.logprior(theta) + self.loglike(theta)

    def logposterior_dict(self, params: dict) -> float:
        """ Compute the log-posterior, using a dictionary as input. Useful for
            jax pytree functions.

        Parameters
        ----------
        params : dict
            A dictionary of parameters. Only the entries matching the
            Pipeline's `parameters` member will be used.
        """
        theta = self.params_to_vec(params)
        return self.logposterior(theta)

    def get_models(self) -> list[Callable]:
        """ Get a list of data model functions for all likelihoods (if such
            a function exists).
        """
        if self.models is None:
            def make_model(like, idx):
                def model(x):
                    x_f = self.expand(x)
                    products = self.compute(x_f)
                    return like.get_model(x_f[idx], **products)
                return model

            self.models = []

            for idx, like in zip(self.like_indices, self.likelihoods):
                self.models.append(jax.jit(make_model(like, idx)))

        return self.models

    def get_jacobians(self) -> list[Callable]:
        """ Get a list of model jacobian functions for all likelihoods (if such
            a function exists).
        """
        if self.jacobians is None:
            models = self.get_models()
            self.jacobians = []

            for idx, mod, like in zip(self.like_indices, models,
                                      self.likelihoods):
                self.jacobians.append(jax.jit(jax.jacfwd(mod)))

        return self.jacobians

    def fisher(self, theta: jnp.ndarray) -> jnp.ndarray:
        """ Compute the Fisher matrix of this pipeline at a given parameter
            set.

        Parameters
        ----------
        theta : jnp.ndarray
            A list of scalar parameters used by this function. The meaning of
            each entry should match the name given by this class' `parameters`
            member.
        """
        F = jnp.zeros((theta.size, theta.size))

        theta_full = self.expand(theta)
        products = self.compute(theta_full)
        jac = self.get_jacobians()

        if self.prior is not None:
            idx = self.prior_free_indices
            F_prior = self.prior.fisher(theta_full[self.prior_indices])
            F = F.at[jnp.ix_(idx, idx)].add(F_prior)

        for idx, J, f_idx, like in zip(self.like_indices, jac,
                                       self.like_free_indices,
                                       self.likelihoods):
            if isinstance(like, GaussianLikelihood):
                L = like.get_covariance_cholesky(theta_full[idx], **products)
                W = jsc.linalg.solve_triangular(L, J(theta), lower=True)

                F = F + W.T @ W
            else:
                F_like = like.fisher(theta_full[idx], **products)
                F = F.at[jnp.ix_(f_idx, f_idx)].add(
                    F_like[jnp.ix_(f_idx, f_idx)]
                )

        return F

    def minimize(self, theta_start: jnp.ndarray, num_runs: int = 1,
                 method: str = "adam", patience: int = 1,
                 min_steps: Optional[int] = None,
                 max_steps: Optional[int] = None, start_spread: float = 0.0,
                 rng_key=None, rtol: float = 0.0, atol: float = 1e-2,
                 tqdm=None, opt_args=None) -> jnp.ndarray | list[jnp.ndarray]:
        """ Minimize the log-posterior function using an optax algorithm (e.g.
        Adam).

        Parameters
        ----------
        theta_start : jnp.ndarray
            The starting point of the minimizer(s). By default, the first
            minimizer will start at this point, while all other minimzers will
            start at a random point near this starting point (controlled by
            the :code:`start_spread` parameter).
        num_runs : int
            How many minimzers to run. Defaults to only one runner.
        method : str
            The method to use. Must be a valid
            `optax <https://optax.readthedocs.io/en/latest/>`_ class. Defaults
            to use the Adam optimizer.
        min_steps : int | None
            The minimum number of steps to run each minimizer for. If set to
            some positive number, each minimizer will run for at least this
            many steps.
        max_steps : int | None
            The maximum number of steps to run each minimizer for. If set to
            :code:`None`, the minimizer will run until the log-posterior stops
            improving (see :code:`patience` parameter).
        start_spread : float
            If set to a number > 0, will start each minimizer after the first
            from a region around the starting point given by the fisher matrix
            around the starting point. The start_spread parameter scales the
            sampling range. Can help converge on a slightly better minimum.
        rng_key
            If given, a random number generator for :code:`jax`. Defaults to a
            system random seed.
        patience : int
            How many steps to wait before closing out of the loop if the
            log-posterior does not improve. The minimizer will stop if the
            log-posterior does not improve the best-fit beyond
            :math:`a + r |\\log(p)|_{\\rm best}` for :code:`patience` steps in
            a row.
        rtol : float
            The relative tolerance of the stopping criterion.
        atol : float
            The absolute tolerance of the stopping criterion.
        tqdm
        opt_args : dict
            Any arguments that get passed to the :code:`optax` optimizer class.
        """
        import optax

        if num_runs < 1:
            raise ValueError("num_runs must be at least 1.")
        if rng_key is None:
            import secrets
            rng_key = jax.random.key(secrets.randbits(32))
        if opt_args is None:
            opt_args = {}
        if min_steps is None:
            min_steps = 1
        if max_steps is None:
            max_steps = jnp.inf

        assert patience >= 1, "Patience should be at least 1."
        assert max_steps > min_steps, ("Maximum number of steps should be "
                                       "higher than minimum number.")

        starting_points = [theta_start] * num_runs

        if start_spread > 0.0:
            F = self.fisher(theta_start)
            cov = jnp.linalg.inv(F)
            L = jnp.linalg.cholesky(cov)
            init_key, rng_key = jax.random.split(rng_key, 2)
            gn = jax.random.normal(init_key,
                                   shape=(num_runs-1, theta_start.size))
            for i in range(1, num_runs):
                starting_points[i] = (starting_points[i]
                                      + start_spread * (L @ gn[i-1]))

        it_runs = range(num_runs)
        if num_runs > 1 and tqdm is not None:
            it_runs = tqdm(it_runs, desc="Minimizers")

        minima = []
        vgrad = jax.jit(jax.value_and_grad(
            lambda theta: -self.logposterior(theta)
        ))

        for i in it_runs:
            theta = starting_points[i]

            try:
                optimizer = getattr(optax, method)(**opt_args)
            except AttributeError:
                raise RuntimeError(f"Optimizer method {method} unknown. "
                                   "Please check the optax documentation for "
                                   "a valid method.")
            state = optimizer.init(theta)

            best = theta
            best_lp = -self.logposterior(theta)
            wait = 0
            steps = 0

            tbar = None
            if tqdm is not None:
                tbar = tqdm(total=max_steps, leave=False)

            while True:
                steps += 1
                v, g = vgrad(theta)

                if not jnp.isfinite(v):
                    logger.debug("Non-finite log-posterior.")
                    break
                if not jnp.all(jnp.isfinite(g)):
                    logger.debug("Non-finite gradient.")
                    break

                delta_logp = best_lp - v
                tol = rtol * jnp.abs(best_lp) + atol

                if v < best_lp:
                    best_lp = v
                    best = theta

                if delta_logp > tol:
                    wait = 0
                else:
                    wait += 1

                if steps >= min_steps:
                    if wait >= patience or steps >= max_steps:
                        logger.debug(f"Best-fit found: {best}.")
                        break

                updates, state = optimizer.update(g, state, theta)
                theta = optax.apply_updates(theta, updates)

                if not jnp.all(jnp.isfinite(theta)):
                    logger.debug("Non-finite theta.")
                    break

                if tbar is not None:
                    tbar.update()
                    tbar.set_description(f"log-post = {v:.3f} / "
                                         f"best = {best_lp:.3f}")

            if tbar is not None:
                tbar.close()

            minima.append((best_lp, best))

        thetas = [theta for _, theta in sorted(minima, key=lambda x: x[0])]
        logpost = [logp for logp, _ in sorted(minima, key=lambda x: x[0])]
        return logpost, thetas

    def sample(self, theta_start: jnp.ndarray, num_steps: int,
               num_samplers: int = 1, method: str = "nuts", rng_key=None,
               kernel_args=None, warmup_steps: int = 0,
               start_spread: float = 0.0,
               tqdm=None
               ) -> (list, list):
        """Sample a chain using a BlackJAX sampler (e.g. NUTS or HMC).

        Parameters
        ----------
        theta_start : jnp.ndarray
            The starting point of the samplers.
        warmup_steps : int
            How many warm-up steps are to be used to condition the sampler.
        num_steps : int
            How many steps each sampler should take.
        num_samplers : int
            The number of samplers to be run. If this number is bigger than 1
            and it does not divide the number of devices, this number gets
            increased until it does, otherwise JAX will not be able to shard
            this.
        start_spread : float
            If set to a number > 0, will start each sampler from a region
            around the starting point given by the fisher matrix at the
            starting point. The start_spread parameter scales the initial
            sampling range. Can help explore the local region better.
        method : str
            A `blackjax <https://blackjax-devs.github.io/blackjax/>`_ sampler
            class to be used. Defaults to use the No U-Turn Sampler (NUTS).
        kernel_args : dict
            If given and no warmup is used, use these parameters for
            initialization. See the :code:`blackjax` documentation for more
            information.
        rng_key
        tqdm
        """
        import blackjax

        if rng_key is None:
            import secrets
            rng_key = jax.random.key(secrets.randbits(32))
        if kernel_args is None:
            kernel_args = {}
        if tqdm is None:
            tqdm = lambda x, **kwargs: x  # noqa: E731

        F = self.fisher(theta_start)
        M = jnp.linalg.inv(F)

        logdensities = []
        positions = []

        if warmup_steps > 0:
            # Do a warmup run.
            logger.info(f"Running {warmup_steps} warm-up steps.")

            warmup = blackjax.window_adaptation(
                getattr(blackjax, method), self.logposterior,
                initial_inverse_mass_matrix=M, is_mass_matrix_diagonal=False,
                **kernel_args
            )

            rng_key, warmup_key = jax.random.split(rng_key, 2)
            with blackjax.progress_bar(label="Warm-up"):
                (state, parameters), _ = warmup.run(warmup_key, theta_start,
                                                    num_steps=warmup_steps)

            M = jnp.asarray(parameters["inverse_mass_matrix"])
            sampler = getattr(blackjax, method)(self.logposterior,
                                                **parameters)
        else:
            # Don't do a warmup run - the user may provide some settings.
            sampler = getattr(blackjax, method)(self.logposterior, M,
                                                **kernel_args)
            state = sampler.init(theta_start)

        if num_samplers == 1:
            # Run only a single chain.
            logger.info(f"Starting single {method} sampler.")

            if start_spread > 0.0:
                logger.info(f"Giving sampler {start_spread} sigma random "
                            "offset.")
                L = jnp.linalg.cholesky(M)
                init_key, rng_key = jax.random.split(rng_key, 2)
                gn = jax.random.normal(init_key, shape=(state.position.size,))
                theta_start = state.position + start_spread * (L @ gn)
                state = sampler.init(theta_start)

            step = jax.jit(sampler.step)

            for i in tqdm(range(num_steps), desc="Sampling"):
                key = jax.random.fold_in(rng_key, i)
                state, _ = step(key, state)
                logdensities.append(float(state.logdensity))
                positions.append(jnp.asarray(state.position))

            logdensities = np.asarray([logdensities])
            positions = np.asarray([positions])
        else:
            # Run multiple chains in parallel using JAX sharding.
            devices = jax.devices()
            n_devices = len(devices)
            if num_samplers % n_devices != 0:
                new_samplers = (
                    num_samplers + (n_devices - num_samplers % n_devices)
                )

                logger.warning(
                    f"The number of chain samplers = {num_samplers} must be "
                    "divisible by the number of available devices = "
                    f"{n_devices}. I am increasing the number of samplers to "
                    f"{new_samplers} to make it fit."
                )

                num_samplers = new_samplers

            logger.info(f"Starting {num_samplers} {method} samplers across "
                        f"{n_devices} devices.")

            # Create starting states.
            if start_spread > 0.0:
                logger.info(f"Giving samplers {start_spread} sigma random "
                            "offset.")
                L = jnp.linalg.cholesky(M)
                init_key, rng_key = jax.random.split(rng_key, 2)
                gn = jax.random.normal(
                    init_key, shape=(num_samplers, state.position.size)
                )
                theta_start = state.position + start_spread * (gn @ L.T)
                states = [
                    sampler.init(theta_start[i]) for i in range(num_samplers)
                ]
            else:
                states = [
                    sampler.init(state.position) for _ in range(num_samplers)
                ]

            states = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
            chain_keys = jax.random.split(rng_key, num_samplers)

            # Initialize sharding setup.
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
                    lambda k, s: run_loop(k, sampler.step, s, num_steps),
                    in_axes=(0, 0),
                )(keys, local_states)

            state_spec = jax.tree.map(
                lambda _: PartitionSpec("chain"), states
            )

            # The actual sharder: it maps from (starting) states to
            # an output array of [logp, theta] per chain.
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

            # Sampling!
            with blackjax.progress_bar(label="Sampling"):
                logdensities, positions = parallel_sampling(chain_keys, states)
                positions.block_until_ready()

            logdensities = np.asarray(logdensities)
            positions = np.asarray(positions)

        return logdensities, positions
