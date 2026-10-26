===========
Likelihoods
===========

.. _likelihood_objects:

A **Likelihood** :math:`\mathcal{L}(d|\theta)` is any function denoting a
probability density relating the a model for some parameters :math:`\theta` to
a dataset :math:`d`.


Custom Likelihood Objects
=========================

.. _likelihood_custom:

It is possible to write your own likelihood objects and use them within DFLike:
to do so, simply import the `Likelihood` class and create your own object:

.. code-block:: python

    class CustomLikelihood(Likelihood):
        def loglike(self, theta, /, **kwargs):
            ...

        def fisher(self, theta, /, **kwargs):
            ...

Where the two functions `loglike` and `fisher` need to be implemented by the user.


Scalar Parameters
-----------------

.. _likelihood_params:

For both of these functions, it is possible to take in pure scalar parameters
in the form of some vector :math:`\theta`. When this function gets called, this
vector will contain a number of values referring to the parameters specified
in the likelihood instance's `parameters` class. So, if you want your custom
likelihood function to depend on some parameters `a`, `b`, and `c`, you would:

.. code-block:: python

    class CustomLikelihood(Likelihood):
        def __init__(self):
            self.parameters = ["a", "b", "c"]
        
        def loglike(self, theta, /, **kwargs):
            # Now, a = theta[0], b = theta[1], c = theta[2].
            ...

Using Computational Products
----------------------------

You can also have the likelihood functions depend on computational products
coming from :ref:`theory objects <theory_objects>`. The best code practice for
this is to use *keyword-only* arguments (this is what is done throughout
*DFLike* internally):

.. code-block:: python

    class CustomLikelihood(Likelihood):
        def loglike(self, theta, /, *, cls, **kwargs):
            # This function now should get called as:
            # custom_likelihood.loglike(theta, cls=cls)

Any arguments provided after :code:`theta` will always refer to computational
products. When using a :ref:`pipeline <pipeline>`, dependencies for these
products will be managed internally by the pipeline, and gradients are always
traced through these products.


.. rubric:: Module overview

.. autoclass:: dflike.likelihood.Likelihood
    :members:

.. autoclass:: dflike.likelihood.GaussianLikelihood
    :members:
