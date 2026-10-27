=======================================
Installation Instructions and Tutorials
=======================================

You should be able to install the code by git cloning the repository and
installing the package.

.. code-block :: bash

    git clone git@github.com:HTJense/cuneifit.git
    cd cuneifit
    pip install .


Running the tests
=================

Cuneifit has a series of tests that can be run with :code:`pytest`. You can
simply invoke :code:`pytest` to run the unit tests, or you can run more
elaborate tests with some of the following options:

.. code-block ::

    --emu-dir="path"    Path to cosmopower emulators that you want to use in
                        your tests. If you set this path to an installation of
                        `this repository <https://github.com/cosmopower-organization/jense_2024_emulators>`_,
                        then the cosmopower tests will run.


Tutorials
=========

A series of example notebooks can be found in the
`notebooks folder <https://github.com/HTJense/cuneifit/tree/main/notebooks>`_.
These notebooks introduce you to the structure of the code by giving some
example likelihood calls, theory interfaces, analysis runs, and finally an
overview of the built-in analysis pipeline functionality.

The documentation provided here is more targeted at explaining the (internal)
python API. Snippets provided here show how python functions are written,
which functions are available, and how to extend the existing code base with
your own extensions (and best practices therefore).
