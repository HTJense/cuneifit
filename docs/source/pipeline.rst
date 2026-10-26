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

    pipeline.parameters
    >>> ["ombh2", "omch2", "H0", "cal90", "cal150"]

Passing a vector like :code:`theta = [0.022, 0.117, 67.5, 1.03, 0.992]` will
automatically be split up and sorted in the correct order for each of the
components.


.. rubric:: Module overview

.. autoclass:: dflike.pipeline.Pipeline
    :members:


Prior Contributions
===================

.. autoclass:: dflike.prior.Prior
    :members:
