import os
from typing import Optional
import yaml


config_extensions = [".yml", ".yaml"]


# This is copied from the way in which cobaya does it.
# It allows one to load a block from a different file.
class DefaultsLoader(yaml.Loader):
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

    loader.current_folder = folder
    return loaded_defaults


DefaultsLoader.add_constructor("!defaults", _load_defaults)


# Check if a configuration file exists here. Otherwise, load it from here.
def config_resolve(filename: str) -> str:
    from importlib.resources import files
    if os.path.exists(filename):
        return filename

    preset_file = str(files("dflike.presets").joinpath(filename))
    if not os.path.splitext(preset_file)[1]:
        preset_file += ".yaml"
    if os.path.isfile(preset_file):
        return preset_file

    raise OSError(f"Failed to find file {filename}.")


# Load a configuration file.
def config_from_file(filename: str) -> dict:
    filename = config_resolve(filename)

    with open(filename) as fp:
        text = fp.read()

    DefaultsLoader.current_folder = os.path.dirname(filename)
    DefaultsLoader.yaml_root_name = (
        os.path.splitext(os.path.basename(filename))[0]
    )
    return yaml.load(text, DefaultsLoader)


# Given a filename, explore several standardized file locations for
# the path.
def config_locate_data(path: str) -> str:
    if os.path.exists(path):
        return path

    try:
        from cobaya.tools import resolve_packages_path

        if os.path.exists(
            fullpath := os.path.join(resolve_packages_path(), "data", path)
        ):
            return fullpath
    except Exception:
        pass

    raise OSError(f"Failed to find data path {path}.")
