import numpy as np
import jax.numpy as jnp

from .cmb_like import MultiFrequency


class Moped(MultiFrequency):
    def __init__(self, config: str | dict):
        super().__init__(config)

        self.moped_B = jnp.array(np.loadtxt(self.config["moped_file"]))
        self.covariance = jnp.eye(self.covariance.shape)
        self.logp_const = -0.5 * np.log(2. * np.pi) * len(self.data_vec)

    def get_model(self, theta, **kwargs):
        model = super().get_model(theta, **kwargs)
        return self.moped_B.T @ model
    
    def get_whitened_residual(self, theta, **kwargs):
        mu = self.get_model(theta, **kwargs)
        y = self.moped_B.T @ self.data_vec - mu
        return y


def get_cobaya_class():
    from .cobaya import Moped_cobaya

    return Moped_cobaya
