from abc import ABC, abstractmethod
import importlib
import importlib.util
import logging
import os
import sys

from . import config
from .prior import Prior


logger = logging.getLogger(__name__)


def import_custom_module(python_path: str, search_dirs: list[str]):
    """Locate a custom module, if the user is giving a path to import it from.
    This is used when trying to build CLI pipelines with custom import paths.
    
    For example:
      - CustomClass:
          python_path: mypkg.custom
    
    Will behave akin to `from mypkg.custom import CustomClass` before building
    the instance. The `search_dirs` list will be checked for the import dirs.
    """
    full_path = os.path.expanduser(python_path)

    if full_path.endswith(".py") or (os.sep in full_path):
        # This is a path to a python file we want to import.
        dirs = [""] if os.path.isabs(path) else search_dirs
        for dirname in dirs:
            if os.path.isfile(
                filename := os.path.abspath(os.path.join(dirname, full_path))
            ):
                break
        else:
            raise RuntimeError(f"Failed to find file {python_path}: searched "
                               f"in {dirs}")

        # Add the directory to our import path list.
        if (filedir := os.path.dirname(filename)) not in sys.path:
            sys.path.insert(0, filedir)

        # Add a unique name in front of the filename to prevent clashes.
        module_name = "cuneifit__" + re.sub(r"\W", "_", filename)
        if module_name in sys.modules:
            # This has already been imported, so we don't need to redo this.
            return sys.modules[module_name]

        # Module doesn't exist yet, import it.
        spec = importlib.util.spec_from_file_location(module_name, filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise

        logger.info(f"Loaded {python_path} from {module.__file__}")
        return module

    # The path was a single string without a file extension or directory in it.
    for d in reversed(search_dirs):
        # "reversed" so that local directories take precedence.
        if d not in sys.path:
            sys.path.insert(0, d)

    try:
        module = importlib.import_module(full_path)
    except ModuleNotFoundError as e:
        raise RuntimeError(f"Failed to import from {python_path}: searched in "
                           f"{search_dirs} and sys.path.") from e

    logger.info(f"Loaded {python_path} from {module.__file__}")
    return module


def build_component(component, conf, search_dirs=[]):
    import cuneifit

    python_path = None
    if isinstance(conf, dict) and "python_path" in conf:
        python_path = conf.pop("python_path")

    if python_path is None:
        cls = getattr(cuneifit, component, None)
        if cls is None:
            raise ImportError(f"Failed to find build-in component {component}: "
                              "Please check the spelling. If you intended to "
                              "import a custom class, set the `python_path` "
                              "attribute to the import path.")
    else:
        module = import_custom_module(python_path, search_dirs)
        cls = getattr(module, component, None)
        if cls is None:
            raise ImportError(f"Failed to find custom component {component}: "
                              f"Custom module {module.__name__} (imported "
                              f"from {module.__file__} has no class {name}.")
        # Include the module directory in the preset search path.
        config.add_config_path(
            os.path.dirname(os.path.abspath(module.__file__))
        )

    if not (isinstance(cls, type) and issubclass(cls, Component)):
        raise TypeError(
            f"Custom type {component} is not a cuneifit Component!"
        )

    component = cls(conf)
    conf = dict(component.config)
    if python_path is not None:
        conf["python_path"] = python_path

    return component, conf


class Component(ABC):
    parameters: list[str]
    default_priors: Prior | None = None
    default_restrictions: dict = {}

    def __init__(self, conf: str | dict | None = None, **kwargs):
        self.logger = logging.getLogger(type(self).__name__)
        self.config = config.load_config(conf, **kwargs)
        self.default_restrictions = self.config.get("restrict", {})

        if (priors := self.config.get("priors")) is not None:
            self.default_priors = Prior()

            for prior in priors:
                self.default_priors.add_prior(prior)

        self.initialize()

    def initialize(self):
        pass
