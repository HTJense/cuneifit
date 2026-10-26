from abc import ABC, abstractmethod


class Theory(ABC):
    """Any object that computes some intermediate product that is used by
    Likelihood objects."""
    parameters: list[str]

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
