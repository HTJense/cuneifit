========
Cuneifit
========

.. image :: https://raw.githubusercontent.com/HTJense/cuneifit/refs/heads/main/docs/source/_static/banner.svg
   :target: https://cuneifit.readthedocs.io/
   :alt: No AI was used in the creation of this image, if you couldn't tell.

This is a collection of Differentiable Likelihoods and codes for Simons Observatory.

It currently contains a JAX implementation of `LAT_MFLike <https://github.com/simonsobs/LAT_MFLike/>`_ (and parts of `fgspectra <https://github.com/simonsobs/fgspectra>`_ which it relies on), as well a simple Gaussian Lensing likelihood based on the one from `SOLikeT <https://github.com/simonsobs/SOLikeT>`_.

*This code is in a highly volatile state and does not represent a finished product. Please take case and review your uses of data products and results, as exact numbers may be unreliable whilst the code is in development. Apologies for the fact that function signatures or class interfaces are constantly changing.*


Installation
============

You should be able to install the code by git cloning the repository and
installing the package.

.. code-block :: bash

    git clone git@github.com:HTJense/cuneifit.git
    cd cuneifit
    pip install .
