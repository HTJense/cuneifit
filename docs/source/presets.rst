==================================
Package Presets and Configurations
==================================

The theory and likelihood codes presented in here may require a configuration.


Using your own configurations on existing modules
=================================================

Replacing an existing configuration with a custom one is easily accessible
by creating a yaml file and loading it when initializing the object.

Files can be nested, for example if two configurations contain the same
sub-section, by using the :code:`!defaults` directive. For example, the file:

.. code-block:: yaml

    key1: value_a
    key2: !defaults [file_key2]

will try to open the file :code:`file_key2.yaml` in the same (working)
directory as the previous file, and paste its contents as a dictionary into
the :code:`key2` section.


Configuring your own code modules
=================================

The module :code:`config.py` contains several functions related to loading
configurations.


Parsing a configuration file
----------------------------

First, the function :code:`config_from_file` loads a configuration yaml file
(checking the local directory first, and the module presets directory second),
returning a dictionary containing the configuration yaml's contents. The
function will first simply try to open a file with the given file name, and if
this file does not exist, check if the file exists in the installed
:code:`cuneifit/presets` directory. In the latter case, a *.yaml* extension is
appended if it doesn't exist yet.

For example, modules can use configuration as:

.. code-block:: python

    class CustomTheory(Theory):
        def __init__(self, config: str | dict):
            if type(config) is str:
                self.config = cuneifit.config.config_from_file(config)
            elif type(config) is dict:
                self.config = config

            ...

Secondly, the function :code:`config_locate_data` takes in a file name or
path and returns the first path from a list of search locations where it
might exist. At first, the function will try to use the full path given, and if
it cannot locate the file there, try to use the
`cobaya package path <https://cobaya.readthedocs.io/en/latest/>`_ to locate it.


Including a preset in the package
---------------------------------

If you want to include a dataset or configuration that you created into the
package directory, you should locate your installation directory for *Cuneifit*
and go to the subdirectory :code:`cuneifit/presets`. Place your yaml file in
this directory, and any use of cuneifit will be able to access it. For example,
the included :code:`act_dr6.yaml` preset is accessible with
:code:`cuneifit.config.config_from_file("act_dr6")`.

