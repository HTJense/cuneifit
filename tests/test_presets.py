import cuneifit
import utilities
import jax.numpy as jnp
import os
import pytest  # noqa: F401

try:
    import cosmopower as cp  # noqa: F401
    import cosmopower_jax.cosmopower_jax as cpj  # noqa: F401
    CP_IMPORT_OK = True
except Exception:
    CP_IMPORT_OK = False

parser_filename = "jense_2023_cmb_camb_lcdm.yaml"


def test_act_dr6_mflike_preset(is_github_workflow, dr6_dir):
    if is_github_workflow:
        pytest.skip("Skipping DR6 mflike test on a github workflow.")

    cuneifit.config.add_data_path(dr6_dir)
    dr6 = cuneifit.MultiFrequency("act_dr6")
    cuneifit.BandpowerForegrounds(dr6, "act_dr6_foregrounds")


def test_act_dr6_lite_preset(dr6_dir):
    cuneifit.config.add_data_path(dr6_dir)
    cuneifit.CMBLite("act_dr6_cmbonly")


@pytest.mark.skipif(not CP_IMPORT_OK, reason="Cosmopower not installed.")
def test_act_dr6_mflike_fiducial(is_github_workflow, emu_dir, dr6_dir):
    if is_github_workflow:
        pytest.skip("Skipping DR6 mflike test on a github workflow.")

    cuneifit.config.add_data_path(dr6_dir)

    cmb = cuneifit.Cosmopower(yaml_file=os.path.join(emu_dir, parser_filename),
                              root_dir=emu_dir)
    dr6 = cuneifit.MultiFrequency("act_dr6")
    fg = cuneifit.BandpowerForegrounds(dr6, "act_dr6_foregrounds")
    pipeline = cuneifit.Pipeline(cmb, dr6, fg)

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


@pytest.mark.skipif(not CP_IMPORT_OK, reason="Cosmopower not installed.")
def test_act_dr6_lite_fiducial(emu_dir, dr6_dir):
    import jax.numpy as jnp

    cuneifit.config.add_data_path(dr6_dir)

    cmb = cuneifit.Cosmopower(yaml_file=os.path.join(emu_dir, parser_filename),
                              root_dir=emu_dir)
    dr6 = cuneifit.CMBLite("act_dr6_cmbonly")
    pipeline = cuneifit.Pipeline(cmb, dr6)

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
