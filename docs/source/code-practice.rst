==============
Code Practices
==============

This page details some of the code practices used throughout the *Cuneifit*
package, and explains the best practices that can be used by you if you want
to release packages that interface with *Cuneifit*.


Position-only and Keyword-only Arguments
========================================

Many functions throughout the code are mathematically of the form
:math:`f = f(\theta; A)` with :math:`\theta` some scalar vector, and :math:`A`
some set of quantities (that may also depend on :math:`\theta`). To allow
for readability of the code, both in the definition of the function *f* and at
every place where *f* is called by the user, the use of the position-only and
keyword-only flags :code:`/` and :code:`*` are used.

In particular: for legibility reasons, we **always** assume that the **first**
argument of any differentiable function is the vector :math:`\theta`, and we
*always* want this to be a position-only argument, with all other arguments
as keyword-only arguments. As such, you should not call a differentiable
function like this:

.. code-block:: python

    result = f(theta=theta, A=A)

or like this:

.. code-block:: python

    result = f(theta, A)

But always like this:

.. code-block:: python

    result = f(theta, A=A)

This is enforced throughout the Cuneifit code base by using the special parameter
arguments :code:`/` and :code:`*`. Specifically, the function :code:`f` would
be defined as:

.. code-block:: python

    def f(theta, /, *, A, **kwargs):
        ...

which *disallows* the use of keyword-parameter usage for :code:`theta` and
disallows the position-parameter usage for :code:`A`.


Variable Keyword Arguments
==========================

Internally, the keyword-only arguments are often handled within *Cuneifit*
through dictionaries. This means for example that our function :code:`f` is
called as

.. code-block:: python

    products = {"A": A, ...}
    result = f(theta, **products)

Note that the dictionary :code:`products` may contain more entries than just
:code:`A`, and as such, the function :code:`f` should have a variable keywords
argument :code:`**kwargs` at the end of its function signature to capture these
computation products that may appear in the analysis. You should therefore
define your function as:

.. code-block:: python

    def f(theta, /, *, A, **kwargs):
        ...

and never as:

.. code-block:: python

    def f(theta, /, *, A):
        ...

