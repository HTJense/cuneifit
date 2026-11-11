from abc import ABC, abstractmethod
import logging

from .config import load_config


logger = logging.getLogger(__name__)


class Theory(ABC):
    """Any object that computes some intermediate product that is used by
    Likelihood objects."""
    parameters: list[str]
    default_restrictions: dict = {}

    def __init__(self, config: str | dict | None = None, **kwargs):
        self.config = load_config(config, **kwargs)
        self.default_restrictions = self.config.get("restrict", {})

    @abstractmethod
    def compute(self, theta, /, **kwargs) -> dict:
        """Compute this object's products.

        Parameters
        ----------
        theta : jnp.ndarray
            A list of scalar parameters used by this function. The meaning of
            each entry should match the name given by this class' `parameters`
            member.
        """
        ...

    @property
    @abstractmethod
    def inputs(self) -> tuple[str]:
        """Flags any other products that this Theory object can compute."""
        ...

    @property
    @abstractmethod
    def outputs(self) -> tuple[str]:
        """Flags any products that this Theory object computes."""
        ...
