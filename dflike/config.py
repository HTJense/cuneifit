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
            dfilename += next(
                ext for ext in [""] + config_extensions
                if (os.path.baseline(dfilename) + ext
                    in os.listdir(os.path.dirname(dfilename)))
            )
        except StopIteration:
            raise RuntimeError(f"File {dfile} not found in {folder}.")

    loader.current_folder = folder
    return loaded_defaults


DefaultsLoader.add_constructor("!defaults", _load_defaults)


def config_from_file(filename: str) -> dict:
    with open(filename) as fp:
        text = fp.read()

    DefaultsLoader.current_folder = os.path.dirname(filename)
    DefaultsLoader.yaml_root_name = (
        os.path.splitext(os.path.basename(filename))[0]
    )
    return yaml.load(text, DefaultsLoader)
