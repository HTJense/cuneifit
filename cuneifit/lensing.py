import numpy as np
import sacc
import os

import jax.numpy as jnp

from .config import resolve_config, load_config, resolve_data
from .likelihood import GaussianLikelihood


class Lensing(GaussianLikelihood):
    def __init__(self, config: str | dict | None = None, **kwargs):
        if type(config) is str:
            config = resolve_config(config)
            self.config = load_config(config)
        elif type(config) is dict:
            config = {**config, **kwargs}
            self.config = config
        elif config is None:
            self.config = kwargs
        else:
            raise TypeError("Configuration should be a string (filename) or "
                            f"dictionary, but was given {type(config)}.")

        data_path = resolve_data(self.config["data_folder"])
        data = sacc.Sacc.load_fits(os.path.join(data_path,
                                                self.config["data_file"]))

        self.parameters = []
        _, cl, ind = data.get_ell_cl("cl_00", "ck", "ck", return_cov=False,
                                     return_ind=True)
        self.data_vec = jnp.array(cl)
        bpw = data.get_bandpower_windows(ind)
        self.ells = jnp.array(bpw.values)
        self.lmax = int(self.ells.max())
        self.binning_matrix = jnp.array(bpw.weight.T)
        self.covariance = jnp.array(data.covariance.covmat[:, :])
        self.cov_L = jnp.linalg.cholesky(self.covariance)
        self.logp_const = -0.5 * (
            np.log(2. * np.pi) * len(self.data_vec)
            + np.linalg.slogdet(self.covariance)[1]
        )

    def get_covariance_cholesky(self, theta, **kwargs):
        return self.cov_L

    def bin_spectra(self, dlkk):
        model_vec = self.binning_matrix @ dlkk
        return model_vec

    def get_unbinned_model(self, theta, cls, corrections=None):
        dlkk = 2. * np.pi * cls["pp"][self.ells] / 4.
        if corrections is None:
            return dlkk
        return dlkk + corrections

    def get_model(self, theta, /, *, cls, corrections=None, **kwargs):
        model = self.get_unbinned_model(theta, cls=cls,
                                        corrections=corrections)
        return self.bin_spectra(model)


def get_cobaya_class():
    """
        This function allow one to import the lensing likelihood into cobaya
        using as `cuneifit.lensing`.
    """
    from .cobaya import Lensing as Lensing_cobaya

    return Lensing_cobaya
