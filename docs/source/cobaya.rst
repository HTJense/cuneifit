======
Cobaya
======

.. attention::

    A separate installation of
    `cobaya <https://github.com/CobayaSampler/cobaya>`_ is needed for this
    content. Please refer to the
    `cobaya documentation <https://cobaya.readthedocs.io/en/latest/>`_ for the
    proper references when using this code. Always cite the sources for the
    data sets you are using.


Cuneifit comes with its own interface to the cobaya sampling software. Because
of differences between the underlying structure of cuneifit and cobaya, an
overview of the different interfaces is given here.

Most of the cobaya interfaces can be directly pointed to either a preset file
or be given a manual dictionary of entries.


General Structure
=================

To streamline the process, separate cobaya likelihood codes are created for
combinations of cuneifit likelihood and theory codes, whenever these codes
are necessarily coupled together. For example, the cuneifit
:code:`MultiFrequency` likelihood requires a :code:`BandpowerForeground` theory,
so the cobaya interface automatically packages these two codes together into
one.

The parameters are automatically inferred from the likelihood and theory
configurations. You will need to refer to the original (external) data or codes
to setup the correct parameter ranges and priors.


CMB-only likelihood
===================

.. code-block :: yaml

    cuneifit.cobaya.CMBLite:
        # A configuration file or preset.
        like_config:


Multi-frequency likelihood
==========================

.. code-block :: yaml

    cuneifit.cobaya.MultiFrequency:
        # A configuration file or preset for the likelihood.
        like_config:
        # A configuration file or preset for the foregrounds.
        fg_config:

This likelihood can also be accessed via the shorthand `cuneifit`.


Lensing likelihood
==================

.. code-block :: yaml

    cuneifit.cobaya.Lensing:
        # A configuration file for the lensing likelihood.
        like_config:
        # An optional configuration for the lensing corrections.
        # If left at null, no lensing corrections are introduced.
        corr_config: null

This likelihood can also be accessed via the shorthand `cuneifit.lensing`.
