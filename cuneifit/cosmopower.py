from dataclasses import dataclass, field
from functools import partial

import jax
from jax import numpy as jnp

from .theory import Theory

T_CMB = 2.72548e6

@jax.tree_util.register_dataclass
@dataclass
class CMB:
    spectra: dict


def _emulators_to_jax(parser, desired=["tt", "te", "ee"]):
    """
        Convert the Cl emulators found in a Cosmopower.YAMLParser into
        ComsmoPower_JAX emulators.
    """
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

    emulators = parser.restore_networks()

    results = {}

    for xy in desired:
        if xy == "bb":
            emu = CosmoPowerJAX("cmb_ee")
        else:
            emu = CosmoPowerJAX(f"cmb_{xy}")

        emu.n_parameters = emulators[f"Cl/{xy}"].n_parameters
        emu.parameters = [str(x) for x in emulators[f"Cl/{xy}"].parameters]
        emu.modes = emulators[f"Cl/{xy}"].modes

        weights = [w.numpy().T for w in emulators[f"Cl/{xy}"].W]
        biases = [b.numpy() for b in emulators[f"Cl/{xy}"].b]
        alphas = [a.numpy() for a in emulators[f"Cl/{xy}"].alphas]
        betas = [b.numpy() for b in emulators[f"Cl/{xy}"].betas]

        emu.weights = list(zip(weights, biases))
        emu.hyper_params = list(zip(alphas, betas))
        # We handle this manually.
        emu.log = False

        if hasattr(emulators[f"Cl/{xy}"], "n_pcas"):
            # PCA data.
            emu.n_pcas = emulators[f"Cl/{xy}"].n_pcas
            emu.pca_matrix = emulators[f"Cl/{xy}"].pca_transform_matrix_
            emu.param_train_mean = emulators[f"Cl/{xy}"].parameters_mean_
            emu.param_train_std = emulators[f"Cl/{xy}"].parameters_std_
            emu.feature_train_mean = emulators[f"Cl/{xy}"].pca_mean_
            emu.feature_train_std = emulators[f"Cl/{xy}"].pca_std_
            emu.training_mean = emulators[f"Cl/{xy}"].features_mean_
            emu.training_std = emulators[f"Cl/{xy}"].features_std_
            emu.probe = "custom_pca"
        else:
            # Non-PCA data.
            emu.param_train_mean = emulators[f"Cl/{xy}"].parameters_mean.numpy()  # noqa: E501
            emu.param_train_std = emulators[f"Cl/{xy}"].parameters_std.numpy()
            emu.feature_train_mean = emulators[f"Cl/{xy}"].features_mean.numpy()  # noqa: E501
            emu.feature_train_std = emulators[f"Cl/{xy}"].features_std.numpy()
            emu.probe = "custom"

        results[xy] = emu

    return results


class Cosmopower(Theory):
    """A theory object capable of interfacing with Cosmopower-JAX.
    """
    def initialize(self):
        import cosmopower as cp

        self.parser = cp.YAMLParser(self.config["yaml_file"],
                               root_dir=self.config.get("root_dir", None))
        desired = self.config.get("desired", ["tt", "te", "ee"])
        self.lmax = self.config.get("lmax", 3000)

        self.parameters = []
        self.emulators = _emulators_to_jax(self.parser, desired=desired)
        for xy in self.emulators:
            for par in self.emulators[xy].parameters:
                if par not in self.parameters:
                    self.parameters.append(par)
        self.indices = {
            xy: jnp.array([
                self.parameters.index(i) for i in self.emulators[xy].parameters
            ])
            for xy in self.emulators
        }

    @partial(jax.jit, static_argnums=(0,))
    def get_cls(self, theta, /, **kwargs) -> dict:
        spectra = {}
        for xy in self.emulators:
            cl = self.emulators[xy].predict(theta[self.indices[xy]])

            if self.parser.is_log(f"Cl/{xy}"):
                cl = jnp.power(10., cl)

            if xy in ["tt", "te", "ee", "bb"]:
                cl = cl * (T_CMB ** 2.)

            cls = jnp.zeros(self.emulators[xy].modes.max()+1)
            spectra[xy] = cls.at[self.emulators[xy].modes].set(cl)

        return CMB(spectra=spectra)

    def compute(self, theta, /, **kwargs):
        return {"cls": self.get_cls(theta)}

    @property
    def inputs(self) -> tuple[str]:
        return ()

    @property
    def outputs(self) -> tuple[str]:
        return ("cls",)
