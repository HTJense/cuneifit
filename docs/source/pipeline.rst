==================
Analysis Pipelines
==================

.. _pipeline:

The Pipeline class provides a simple pythonic interface for doing analyses. In
general, a Pipeline is simply a set of likelihood and theory codes, optionally
with a set of (non-flat) priors.

The main way in which a pipeline can be initialized, is to simply pass a list
of components, e.g.:

.. code-block:: python

    like = dflike.Likelihood(...)
    theory = dflike.Theory(...)
    pipeline = dflike.Pipeline([like, theory])

If the likelihood and theory code both depend on some set of parameters, then
the pipeline object will automatically manage these parameters internally, and
expose only the parameters that are of interest. For example, if the theory
code requires some cosmological parameters, and the likelihood requires some
calibration factors, then the full set of parameters might be

.. code-block:: python

    >>> pipeline.parameters
    ["ombh2", "omch2", "H0", "cal90", "cal150"]

Passing a vector like :code:`theta = [0.022, 0.117, 67.5, 1.03, 0.992]` will
automatically be split up and sorted in the correct order for each of the
components.


Fixing Parameters
=================

If a likelihood or theory component relies on some parameters that you would
prefer to keep fixed during your analysis, then you can call the
:code:`restrict` function with a dictionary of parameters and their values. For
example, if you would like to keep both calibration parameters in the previous
example fixed to unity throughout the analysis, then you could do:

.. code-block:: python

    >>> pipeline.parameters
    ["ombh2", "omch2", "H0", "cal90", "cal150"]
    >>> pipeline.restrict({"cal90": 1.0, "cal150": 1.0})
    >>> pipeline.parameters
    ["ombh2", "omch2", "H0"]

Any of the functions from now on will only take these parameters as inputs, and
the *restricted* parameters will always be kept at the given fiducial values,
with their derivatives not propagated. A future call to :code:`restrict` will
**overwrite** the settings from previous calls.


Computing log-posteriors
========================

The pipeline has three main functions, to compute the log-likelihood, log-prior,
and log-posterior, with the latter being simply the sum of the first two. We
can trivially compute these:

.. code-block:: python

    >>> logp = pipeline.logposterior(theta)


Autodifferentiation
-------------------

The functions provided are all auto-differentiable through jax. For example, we
can compute the log-posterior at some point :math:`\theta`, but we can also
use jax to compute the gradient :math:`\nabla_\theta \ln P`

.. code-block:: python

    >>> logp = pipeline.logposterior(theta)
    >>> grad = jax.grad(pipeline.logposterior)
    >>> dlogp = grad(theta)

This function :code:`grad` will now be a :math:`\mathbb{R}^n \to \mathbb{R}^n`
function that gives the gradient of the log-posterior.

We can also access the model jacobians: for each likelihood, the data will be
somehow modelled by a function :math:`\mu(\theta)`, and this function is
differentiable and we can access its jacobian :math:`\mathbf{J}_\mu(\theta)`,
either manually or through the pipeline:

.. code-block:: python

    >>> jac = jax.jacfwd(like.get_model)
    >>> jacs = pipeline.get_jacobians()

Here, the function `jac` might depend on some computation products that need
to be manually added into the full jacobian, while the pipeline automatically
includes them: the final :code:`jacs` return value is a list of functions that
directly compute the full model jacobian.


Fisher forecasting
------------------

The use of autodifferentiation is clear when trying to compute fisher forecasts:
mathematically, the fisher matrix at some fiducial :math:`\theta` equals
:math:`\mathbf{F}_{ij} = -\partial^2 \ln P / \partial \theta_i \partial \theta_j|_{\theta}`.
Using autodifferentiation, we could simply evaluate the Hessian of the
log-posterior function, and in theory you can compute this easily as:

.. code-block:: python

    H = jax.hessian(pipeline.logposterior)
    F = -H(theta)

However, for some likelihood forms, the fisher matrix might be simpler to
compute. For example, for a Gaussian likelihood, the fisher matrix can be
approximated as :math:`\mathbf{F} = -\mathbf{J} \mathbf{\Sigma}^{-1} \mathbf{J}`,
where :math:`\mathbf{J}` is the model Jacobian. As such, a built-in
:code:`fisher` function exists that might optimize the full fisher computation:

.. code-block:: python

    F = pipeline.fisher(theta)

Note that a lot of these functions cache some results as well: calling them the
first time will make jax propagate derivatives, but the results are then jit
compiled and stored internally, re-using them the next function call.


.. rubric:: Module overview

.. autoclass:: dflike.pipeline.Pipeline
    :members:


Prior Contributions
===================

By default, we are not including any prior contributions to parameters. We can
add some priors with the :code:`Prior` class: it should simply be a list of
functions that take in parameters and return a log-posterior function. For
example, we can add a Gaussian prior to our calibration parameters as:

.. code-block:: python

    from jax import scipy as jsc

    prior = dflike.Prior([
        lambda cal90: jsc.norm.logpdf(cal90, loc=1.0, scale=0.01),
        lambda cal150: jsc.norm.logpdf(cal150, loc=1.0, scale=0.01)
    ])

    pipeline = dflike.Pipeline([prior, like, theory])

These functions should have the same parameter names as any likelihoods or
theory codes, but they can be functions of multiple parameters, for example if
we only have a relative calibration prior:

.. code-block:: python

    prior = dflike.Prior([
        lambda cal90, cal150: jsc.norm.logpdf(cal90 / cal150, loc=1.0, scale=0.01),
    ])


.. autoclass:: dflike.prior.Prior
    :members:
