====================
Cosmopower Emulators
====================


.. attention::

    When using this Cosmopower interface, please cite the following papers:

    A. Spurio Mancini et al., 2021
    (`COSMOPOWER: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys <https://arxiv.org/abs/2106.03846>`_)

    D. Piras, A. Spurio Mancini, 2023
    (`CosmoPower-JAX: high-dimensional Bayesian inference with differentiable cosmological emulators <https://arxiv.org/abs/2305.06347>`_)

    H. T. Jense et al., 2025
    (`A complete framework for cosmological emulation and inference with CosmoPower <https://doi.org/10.1093/rasti/rzaf002>`_)


Because of technical limitations of the pip version of Cosmopower and
Cosmopower-JAX, it is easiest to manually install the Cosmopower fork found on
`this repository <https://github.com/HTJense/cosmopower>`_. You can then use
the pip version of :code:`cosmopower-jax` as-is.


Loading Emulators
=================

The DFLike :code:`cosmopower` wrapper is built around the :code:`YAMLParser`
interface found in cosmopower. Pre-trained emulators can be readily loaded in
from their provided packaging metadata file.

.. hint::

    Several ready-to-use emulators for LCDM and extension models can be found
    `here <https://github.com/cosmopower-organization/jense_2024_emulators/tree/main>`_.


To instantiate the cosmopower code, simply create the parser object and pass
it to DFLike:

.. code-block:: python

    import cosmopower as cp

    parser = cp.YAMLParser("cp_config_file.yaml")
    emulators = dflike.Cosmopower(parser)


.. attention::

    Although the cosmopower package can handle emulators other than CMB multipole
    emulators, the wrapper built into DFLike is currently only capable of these
    kinds of emulators. It will not load or provide any other computation
    products at the moment.


At this point, similar to how other :ref:`theory codes<theory_objects>`
function, you can pass it to a pipeline, or manually check the code's desired
parameters and outputs:

.. code-block:: python

    theta = jnp.array([parameters[p] for p in emulators.parameters])
    cls = emulators.compute(theta)

    pipeline = dflike.Pipeline([emulators, ...])


Method overview
===============

.. autoclass:: dflike.cosmopower.Cosmopower
    :members:
