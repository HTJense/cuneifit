from abc import ABC, abstractmethod
import logging

from .component import Component


class Theory(Component):
    """Any object that computes some intermediate product that is used by
    Likelihood objects."""
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
    def inputs(self) -> tuple[str]:
        """Flags any other products that this Theory object needs to ingest."""
        return ()

    @property
    @abstractmethod
    def outputs(self) -> tuple[str]:
        """Flags any products that this Theory object computes."""
        ...
