============
Theory Codes
============

.. _theory_objects:

Sometimes, a parametric dependency for a Likelihood relates to a more
complicated relationship that is easier to abstract away into a different
object. For this, :code:`Theory` objects exist.

In general, a Theory class simply computes some quantity :math:`P(\theta; Q)`
based on some parameterization :math:`\theta` and optionally some other
computation products :math:`Q`. In the case where our Likelihood needs :math:`P`
for its computation, we thus need to compute
:math:`Q(\theta) \to P(\theta;Q) \to \ln\mathcal{L}(d|\theta,P,Q)`.

The :code:`Theory` interface ensures that the parameter and product dependency
is handled correctly by the :ref:`analysis pipeline <pipeline>`.


Custom Theory Objects
=====================

Similar to writing your own :ref:`custom likelihood <likelihood_custom>` codes,
you can create your own custom theory objects in the same manner:

.. code-block:: python

    class CustomTheory(Theory):
        def compute(self, theta, /, **kwargs):
            return {}

        @property
        def inputs(self):
            ...

        @property
        def outputs(self):
            ...

The :code:`compute` function works simlar to the :code:`loglike` function from
the :code:`Likelihood` class: it takes in a vector :math:`\theta`, but in this
case it returns a *dictionary* containing a series of computation products that
can be passed to other theory codes, or used in likelihoods.

Similar to the likelihood class, :ref:`scalar parameters <likelihood_params>`
can be requested by initializing a :code:`parameters` list, for example:

.. code-block:: python

    class CustomTheory(Theory):
        def __init__(self):
            self.parameters = ["a", "b"]

        def compute(self, theta, /, **kwargs):
            # theta will be a length-2 array with
            # a = theta[0] and b = theta[1].
            a = theta[0]
            b = theta[1]
            return {"P": a + b ** 2}


Handling Computation Outputs
----------------------------

From a technical standpoint, internally these Theory codes are used by
:ref:`analysis pipelines <pipeline>` by iterating over all Theory codes, and
computing their results one by one. However, at times the output of one Theory
object depends on the result of another Theory computation. In this case, you
have to flag that your Theory object depends on a different result using the
:code:`outputs` property. For example, in our earlier case where
:math:`P = P(Q(\theta_q);\theta_p)`:

.. code-block:: python

    class CustomTheory1(Theory):
        def compute(self, theta, /, **kwargs):
            # Compute something
            return {"Q": ...}

    class CustomTheory2(Theory):
        def compute(self, theta, /, *, Q, **kwargs):
            # Compute something
            return {"P": ...}

        @property
        def inputs(self):
            # Explicitly flag `Q` as an input for this code.
            return ["Q"]

In this case, the analysis pipeline will postpone the computation of *P* until
*Q* has been calculated, and then calls
:code:`custom_theory2.compute(theta_p, Q=Q)` after.


.. rubric:: Module overview

.. autoclass:: cuneifit.theory.Theory
    :members:
