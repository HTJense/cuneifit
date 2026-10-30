import os
from typing import Optional
import yaml


config_extensions = [".yml", ".yaml"]
search_data_paths = []
search_config_paths = []
env_data_path = os.getenv("CUNEIFIT_DATA_DIR", None)
env_config_path = os.getenv("CUNEIFIT_CONFIG_DIR", None)


# This is copied from the way in which cobaya does it.
# It allows one to load a block from a different file.
class DefaultsLoader(yaml.SafeLoader):
    current_folder: Optional[str] = None
    yaml_root_name: Optional[str] = None


# Load files from default.
def _load_defaults(loader, node):
    if loader.current_folder is None:
        raise RuntimeError("Tried using !defaults without loading a file.")

    try:
        defaults_file = [loader.construct_scalar(node)]
    except yaml.constructor.ConstructorError:
        defaults_file = loader.construct_sequence(node)

    folder = loader.current_folder
    loaded_defaults = {}
    for dfile in defaults_file:
        df = os.path.abspath(os.path.join(folder, dfile))
        try:
            df += next(
                ext for ext in [""] + config_extensions
                if (os.path.basename(df) + ext
                    in os.listdir(os.path.dirname(df)))
            )
        except StopIteration:
            raise RuntimeError(f"File {dfile} not found in {folder}.")

        loaded_defaults |= load_config(df)

    loader.current_folder = folder
    return loaded_defaults


DefaultsLoader.add_constructor("!defaults", _load_defaults)


def add_data_path(path: str):
    """ Add a path to the data search order. Whenever a data file path needs to
    be resolved, and the file does not exist in the current working directory,
    the the given path will be searched as well.
    """
    search_data_paths.append(path)


def add_config_path(path: str):
    """ Add a path to the config search order. Whenever a config file path
    needs to be resolved, and the file does not exist in the current working
     directory, the the given path will be searched as well.
    """
    search_config_paths.append(path)


# Check if a configuration file exists here. Otherwise, load it from here.
def resolve_config(filename: str) -> str:
    from importlib.resources import files
    fn_options = []

    if not os.path.splitext(filename)[1]:
        for ext in config_extensions:
            fn_options.append(filename + ext)

    path_options = fn_options.copy()

    for extra_path in search_config_paths:
        for fn in fn_options:
            path_options.append(os.path.join(extra_path, fn))

    preset_dir = files("cuneifit.presets")
    for fn in fn_options:
        path_options.append(str(preset_dir / fn))

    for fn in path_options:
        if os.path.exists(fn):
            return fn

    errstr = (f"Failed to find config file {filename}.\n"
              "Search paths:\n\t" + "\n\t".join(path_options))
    raise OSError(errstr)


# Load a configuration file.
def load_config(filename: str) -> dict:
    with open(filename) as fp:
        text = fp.read()

    DefaultsLoader.current_folder = os.path.dirname(filename)
    DefaultsLoader.yaml_root_name = (
        os.path.splitext(os.path.basename(filename))[0]
    )
    return yaml.load(text, DefaultsLoader)


# Given a filename, explore several standardized file locations for
# the path. The order of earch
def resolve_data(filename: str) -> str:
    """ Given a filename, explore several file locations for the path.

    The order of the search is:
      1. The current work directory;
      2. Any path added to the search list using `config_add_path`;
      3. The path given by the environment variable $CUNEIFIT_DATA_DIR;
      4. The cobaya data directory.
    """
    from importlib.resources import files

    path_options = [filename]

    for extra_path in search_data_paths:
        path_options.append(os.path.join(extra_path, filename))

    preset_dir = files("cuneifit.data")
    path_options.append(str(preset_dir / filename))

    if env_data_path is not None:
        path_options.append(os.path.join(env_data_path, filename))

    try:
        from cobaya.tools import resolve_packages_path

        path_options.append(
            os.path.join(resolve_packages_path(), "data", filename)
        )
    except Exception:
        pass

    for fn in path_options:
        if os.path.exists(fn):
            return fn

    errstr = (f"Failed to find data file {filename}.\n"
              "Search paths:\n\t" + "\n\t".join(path_options))
    raise OSError(errstr)
