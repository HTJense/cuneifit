import cuneifit
import utilities
import jax.numpy as jnp
import os
import pytest  # noqa: F401


parser_filename = "jense_2023_cmb_camb_lcdm.yaml"


def test_act_dr6_mflike_preset(dr6_dir):
    cuneifit.config.config_add_path(dr6_dir)
    dr6 = cuneifit.MultiFrequency("act_dr6")
    cuneifit.BandpowerForegrounds("act_dr6_foregrounds", dr6)


def test_act_dr6_lite_preset(dr6_dir):
    cuneifit.config.config_add_path(dr6_dir)
    cuneifit.CMBLite("act_dr6_cmbonly")


def test_act_dr6_mflike_fiducial(emu_dir, dr6_dir):
    import cosmopower as cp

    cuneifit.config.config_add_path(dr6_dir)

    parser = cp.YAMLParser(os.path.join(emu_dir, parser_filename),
                           root_dir=emu_dir)
    cmb = cuneifit.Cosmopower(parser)
    dr6 = cuneifit.MultiFrequency("act_dr6")
    fg = cuneifit.BandpowerForegrounds("act_dr6_foregrounds", dr6)
    pipeline = cuneifit.Pipeline([cmb, dr6, fg])

    theta = jnp.array([utilities.test_dr6_parameters[p]
                       for p in pipeline.parameters])

    assert jnp.isclose(pipeline.logposterior(theta),
                       utilities.test_dr6_logpost)

    theta_cmb = jnp.array([utilities.test_dr6_parameters[p]
                           for p in cmb.parameters])
    cls = cmb.compute(theta_cmb)
    theta_fg = jnp.array([utilities.test_dr6_parameters[p]
                          for p in fg.parameters])
    fgs = fg.compute(theta_fg)
    theta_like = jnp.array([utilities.test_dr6_parameters[p]
                            for p in dr6.parameters])
    chi2 = dr6.chisquare(theta_like, **cls, **fgs)

    assert jnp.isclose(chi2, utilities.test_dr6_chi2)


def test_act_dr6_lite_fiducial(emu_dir, dr6_dir):
    import cosmopower as cp
    import jax.numpy as jnp

    cuneifit.config.config_add_path(dr6_dir)

    parser = cp.YAMLParser(os.path.join(emu_dir, parser_filename),
                           root_dir=emu_dir)
    cmb = cuneifit.Cosmopower(parser)
    dr6 = cuneifit.CMBLite("act_dr6_cmbonly")
    pipeline = cuneifit.Pipeline([cmb, dr6])

    theta = jnp.array([utilities.test_dr6_lite_parameters[p]
                       for p in pipeline.parameters])

    assert jnp.isclose(pipeline.logposterior(theta),
                       utilities.test_dr6_lite_logpost)

    theta_cmb = jnp.array([utilities.test_dr6_lite_parameters[p]
                           for p in cmb.parameters])
    cls = cmb.compute(theta_cmb)
    theta_like = jnp.array([utilities.test_dr6_lite_parameters[p]
                            for p in dr6.parameters])
    chi2 = dr6.chisquare(theta_like, **cls)

    assert jnp.isclose(chi2, utilities.test_dr6_lite_chi2)
