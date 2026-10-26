===============
CMB Likelihoods
===============

This is a description of the CMB-only likelihood code.


The CMB-only likelihood is based on the
`ACT DR6 cmb-only likelihood <https://github.com/ACTCollaboration/DR6-ACT-lite>`_.
It models the data as a simple binned Gaussian signal, modified only by a
calibration uncertainty using a single global gain calibration
:math:`\mathrm{cal}_G`, and a pair of specific temperature-only and
polarization-only calibrations :math:`\mathrm{cal}_T`, :math:`\mathrm{cal}_P`.

The values given here match the :code:`act_dr6_cmbonly` preset.

.. code-block:: yaml

    # A sub-directory where the data should be stored.
    data_folder: ACTDR6CMBonly/v1.0
    # The name of the data file.
    input_file: dr6_data_cmbonly.fits

    # Which spectra are used in the likelihood, e.g. [tt, te, ee]
    requested_cls: [tt, te, ee]

    # The highest multipole requested from the theory code
    lmax: 6500

    # The range of scales to be loaded from the dataset, per cross-spectrum.
    scales:
      tt: [600, 6500]
      te: [600, 6500]
      ee: [600, 6500]

    # The name of the experiment stored inside the sacc file.
    experiment: dr6_cmb


.. rubric:: Module overview

.. autoclass:: dflike.cmb_like.CMBLite
    :members:
