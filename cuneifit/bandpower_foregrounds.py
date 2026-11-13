import os

import numpy as np
import jax.numpy as jnp
import sacc
from scipy import constants

from .config import resolve_data
from .theory import Theory

from fgspectra import jax as fgm
from fgspectra import power as fgp
from fgspectra import frequency as fgf

T_CMB = 2.72548


def _cmb2bb(nu, T=T_CMB):
    x = nu * constants.h * 1e9 / (constants.k * T)
    return jnp.exp(x) * (nu * x / jnp.expm1(x)) ** 2.


def _trapz_weights(nu):
    d = jnp.diff(nu)
    return jnp.concatenate([d[:1] / 2, (d[1:] + d[:-1]) / 2, d[-1:] / 2])


class BandpowerForegrounds(Theory):
    def __init__(self, config: str | dict | None, **kwargs):
        super().__init__(config, **kwargs)

        self.lmax = self.config["lmax"]
        self.ells = jnp.arange(2, self.lmax+1)
        self.experiments = self.config["experiments"]

        data_path = resolve_data(self.config["data_folder"])
        s = sacc.Sacc.load_fits(os.path.join(data_path,
                                             self.config["input_file"]))

        self.tracers = {}
        for name, tracer in s.tracers.items():
            self.tracers[name] = {
                "nu": tracer.nu,
                "bp": tracer.bandpass,
                "beam": tracer.beam.T
            }

        if self.config["top_hat_band"] is None:
            self.load_bandpass_from_file()
        else:
            self.build_bandpass()

        if self.config["beam_profile"] is None:
            self.init_beam_flat()
        elif "beam_from_file" in self.config["beam_profile"]:
            self.init_beam_from_file()

        self.parameters = []
        for exp in self.experiments:
            self.parameters.append(f"bandint_shift_{exp}")
        self.bp_index = jnp.array(
            [self.parameters.index(f"bandint_shift_{exp}")
             for exp in self.experiments])

        self.build_foreground_model(self.config["components"])
        self.bp_w = [
            _trapz_weights(nu) if len(nu) > 1 else jnp.ones(1)
            for nu in self.nu
        ]

    def load_bandpass_from_file(self):
        self.nu = [self.tracers[exp + "_s0"]["nu"].astype(float)
                   for exp in self.experiments]
        self.bp = [self.tracers[exp + "_s0"]["bp"].astype(float)
                   for exp in self.experiments]

    def build_bandpass(self):
        self.nu = []
        self.bp = []

        nsteps = self.config["top_hat_band"]["nsteps"]
        bandwidth = self.config["top_hat_band"]["bandwidth"]
        if type(bandwidth) is not list:
            bandwidth = [bandwidth for _ in self.experiments]
        nu_mid = self.config["bandint_freqs"]

        for i, (num, bw) in enumerate(zip(nu_mid, bandwidth)):
            self.nu.append(np.linspace(num - bw / 2., num + bw / 2., nsteps))
            self.bp.append(np.ones((nsteps,)))

    def init_beam_flat(self):
        for i, (exp, bp, nu) in enumerate(zip(self.experiments, self.bp,
                                              self.nu)):
            bp_beam = bp[:, None] * np.ones((1, len(self.ells)))

            self.nu[i] = jnp.array(nu)
            if len(nu) > 1:
                bp_beam = bp_beam / np.trapezoid(bp_beam, nu, axis=0)

            self.bp[i] = jnp.array(bp_beam)

    def init_beam_from_file(self):
        for i, (exp, bp, nu) in enumerate(zip(self.experiments, self.bp,
                                              self.nu)):
            beam = self.tracers[exp + "_s0"]["beam"]
            bp_beam = bp[:, None] * beam[:, self.ells]

            self.nu[i] = jnp.array(nu)
            if len(nu) > 1:
                bp_beam = bp_beam / np.trapezoid(bp_beam, nu, axis=0)

            self.bp[i] = jnp.array(bp_beam)

    def build_foreground_model(self, config):
        # TODO: cleanup this function >_<
        self.foreground_components = {cl: [] for cl in config}
        self.fg_indices = {cl: [] for cl in config}
        defaults = self.config["normalisation"]

        for cl in config:
            cl_indices = []

            for component in config[cl]:
                model_config = config[cl][component]
                mod_name = list(model_config.keys())[0]
                mod = getattr(fgm, mod_name)
                model_products = []
                product_list = (model_config[mod_name] or [])
                for product in product_list:
                    tmpl_name = list(product.keys())[0]
                    settings = product[tmpl_name]
                    settings = defaults | (settings or {})
                    if hasattr(fgp, tmpl_name):
                        template = getattr(fgp, tmpl_name)
                    elif hasattr(fgf, tmpl_name):
                        template = getattr(fgf, tmpl_name)
                    else:
                        raise ImportError(f"Failed to find {tmpl_name} among "
                                          "Cl/SED templates. Check spelling?")

                    model_products.append(template(**settings))
                model = mod(*model_products)

                n_req = model.n
                n_prov = len(model_config["params"])

                assert n_req == n_prov, (f"Configuration provided {n_prov} "
                                         "parameters for component "
                                         f"{component}, but {n_req} are "
                                         "required.")

                self.foreground_components[cl].append(model)

                indices = []
                for param in model_config["params"]:
                    if param not in self.parameters:
                        self.parameters.append(param)
                    indices.append(self.parameters.index(param))
                cl_indices.append(jnp.array(indices))

            self.fg_indices[cl] = cl_indices

    def apply_bandpass_shifts(self, bandint_theta):
        nus = []
        bps = []

        for i, (nu, bp0, w) in enumerate(zip(self.nu, self.bp, self.bp_w)):
            nub = nu + bandint_theta[i]
            nus.append(nub)
            if len(nub) > 1:
                wg = w * _cmb2bb(nub)
                bps.append((wg, bp0, wg @ bp0))
            else:
                bps.append((w, bp0, jnp.ones(bp0.shape[1])))

        return nus, bps

    def get_foregrounds(self, theta):
        nu, bp = self.apply_bandpass_shifts(theta[self.bp_index])

        foregrounds = {
            "experiments": self.experiments
        }

        for cl in self.foreground_components:
            fg_total = jnp.zeros((self.lmax+1, len(self.experiments),
                                  len(self.experiments)))

            for idx, fg in zip(self.fg_indices[cl],
                               self.foreground_components[cl]):
                theta_fg = theta[idx]
                fg_total = fg_total.at[self.ells].add(
                    fg(nu, bp, self.ells, theta_fg)
                )

            foregrounds[cl] = fg_total

        return foregrounds

    def compute(self, theta, /, **kwargs):
        return {"foregrounds": self.get_foregrounds(theta)}

    @property
    def inputs(self):
        return ()

    @property
    def outputs(self):
        return ("foregrounds",)
