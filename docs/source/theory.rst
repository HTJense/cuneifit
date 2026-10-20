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


Method overview
---------------

.. autoclass:: dflike.theory.Theory
    :members:
