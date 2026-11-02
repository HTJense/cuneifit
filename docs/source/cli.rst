==========================================
Running Pipelines through the Command Line
==========================================

Cuneifit comes with a lightweight command line interface (CLI) for performing
full analysis runs without having to write your own python scripts. This
interface is reminiscent of the interface used in cobaya, and provides a
minimal low-cost automated process to run your codes through several built-in
:code:`pipeline` functions that come with cuneifit.


Invoking the CLI
================

To start an analysis, simply invoke the :code:`cuneifit` command in your
terminal.

.. code-block :: bash

    usage: Cuneifit [-h] [--debug] [--plot] run_file

    Cuneifit --- Differentiable Inference for Simons Observatory

    positional arguments:
      run_file    A run configuration to analyze.

    options:
      -h, --help  show this help message and exit
      --debug     Enable debug logging.
      --plot      Make plots as you go.

As you can see, several command-line flags exist. The `debug` flag will enable
debug logging in the output. The `plot` command will make the pipeline create
several automated analysis plots. This flag requires :code:`matplotlib` and
:code:`getdist` to be installed to work.


The run configuration file
==========================

The run configuration is setup in a yaml file containing several blocks. An
annotated example configuration is given here.

.. important::

    To run the minimizer, make sure you have installed :code:`optax`.

    To run the full chain sampler, make sure you have installed
    :code:`blackjax`.
    Note that the sampler makes use of jax's sharding API. If you are running
    the code on a CPU, then XLA will not default to using multiple CPU cores
    at the same time. This can be enabled by setting the environment variable
    :code:`export XLA_FLAGS="--xla_force_host_platform_device_count=<count>"`
    with :code:`<count>` replaced by however many CPU cores you have.

A simple analysis
-----------------

.. code-block :: yaml

    # A simple path/to/output/filename string that points to where the
    # files should be saved. By default, the `path/to/output` directory will be
    # created and filled with the analysis products plus optional plots (if
    # enabled with the --plot flag).
    output: directory/analysis_name

    # A block that contains information about the (non-prior) components that
    # we want to include in our pipeline.
    components:
      # Each entry in this list refers to one theory or likelihood that is
      # to be included in the pipeline.

      # The key in each entry must match a Cuneifit class that can be
      # imported. Setting the value to a string assumes that the configuration
      # is loaded from either a local yaml file or a built-in preset.
      - CMBLite: act_dr6_cmbonly
      # Setting the value to a dictionary will assume you manually set all the
      # component's values.
      - Cosmopower:
          yaml_file: jense_2024_emulators/jense_2023_cmb_camb_lcdm.yaml
          root_dir: jense_2024_emulators/
      # It is also possible to load a preset and _overwrite_ some values, for
      # example to load a different dataset than a preset. This is done by
      # first setting the "preset" key, and then giving every value that has
      # to be overwritten.
      # - CMBLite:
      #     preset: act_dr6_cmbonly
      #     input_file: my_custom_dataset.fits

    # Priors get their own block. Each prior takes in a list of parameters that
    # this prior refers to, plus a built-in scipy probability distribution, and
    # that distribution's parameters.
    priors:
      - par: calG
        dist: norm
        loc: 1.0
        scale: 0.003

      - par: tau
        dist: norm
        loc: 0.06
        scale: 0.01

    # We can restrict several parameters by flagging them as "restricted":
    restrict:
      calT: 1.0

    # Next, our analysis needs a starting point. We do this with a "start"
    # block:
    start:
      ombh2: 0.022
      omch2: 0.120
      h: 0.67
      ns: 0.96
      logA: 3.05
      tau: 0.06
      calG: 1.0
      calP: 1.0

    # We can then run either a minimizer or a full chain, or both (in *that*
    # order!).

    # These arguments are passed almost exactly to the pipeline's minimize
    # function, so see that function's documentation for information on what
    # each setting does.
    minimizer:
      runners: 100
      method: adam

      args:
        learning_rate: 1.0e-2

    # If a minimizer is found, its result is used as the starting point for
    # running HMC chains. Again, these parameters are passed on almost exactly
    # to the pipeline's sample function.
    chain:
      method: nuts
      runners: 4
      warmup: 100
      samples: 1000

    # Finally, we can pass on some settings to Cuneifit's internal plotting
    # functions:
    plot_args:
      labels:
        ombh2: \Omega_b h^2
        omch2: \Omega_c h^2
        h: h
        ns: n_s
        logA: \log(10^{10} A_s)
        tau: \tau
        calG: \mathrm{cal}_{\rm dip}
        calP: \mathrm{p}_{\rm ACT}

    # And we can do the same for the getdist settings. Aside from the `burn_in`
    # setting, which sets how many samples are to be ignored by getdist,
    # all settings are passed directly on to getdist's GetDistPlotSettings
    # class.
    getdist_settings:
      burn_in: 0.3

      linewidth: 2.6
      linewidth_contour: 2.0
