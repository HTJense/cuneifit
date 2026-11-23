from abc import ABC, abstractmethod
import logging

from .config import load_config
from .prior import Prior


class Component(ABC):
    parameters: list[str]
    default_priors: Prior | None = None
    default_restrictions: dict = {}

    def __init__(self, config: str | dict | None = None, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.config = load_config(config, **kwargs)
        self.default_restrictions = self.config.get("restrict", {})

        if (priors := self.config.get("priors")) is not None:
            self.default_priors = Prior()

            for prior in priors:
                self.default_priors.add_prior(prior)

        self.initialize()

    def initialize(self):
        pass
