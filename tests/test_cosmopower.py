import os
import pytest


try:
    import cosmopower as cp  # noqa: F401
    import cosmopower_jax.cosmopower_jax as cpj  # noqa: F401
except Exception as e:
    pytest.skip(
        f"Failed to import cosmopower: {str(e)}.",
        allow_module_level=True,
    )


parser_filename = "jense_2023_cmb_camb_lcdm.yaml"


def test_cosmopower_parser(emu_dir):
    """Test whether we can load data for cosmopower.
       (This is not really testing this package, but rather testing whether the
       cosmopower dependency works).
    """
    cp.YAMLParser(os.path.join(emu_dir, parser_filename), root_dir=emu_dir)


def test_cosmopower_emulators(emu_dir):
    """Test whether we can load emulators into jax.
    """
    import cuneifit

    cuneifit.Cosmopower(yaml_file=os.path.join(emu_dir, parser_filename),
                        root_dir=emu_dir)
    cuneifit.Cosmopower(yaml_file=os.path.join(emu_dir, parser_filename),
                        root_dir=emu_dir, desired=["tt"])
    cuneifit.Cosmopower(yaml_file=os.path.join(emu_dir, parser_filename),
                        root_dir=emu_dir, desired=["tt", "te", "ee",
                                                   "bb", "pp"])


def test_cosmopower_evaluate(emu_dir):
    """Test whether we can evaluate emulators.
    """
    import cuneifit
    import utilities
    import jax.numpy as jnp

    cosmopower = cuneifit.Cosmopower(yaml_file=os.path.join(emu_dir,
                                                            parser_filename),
                                     root_dir=emu_dir, desired=["tt"])

    theta = jnp.array([utilities.test_cosmopower_parameters[p]
                       for p in cosmopower.parameters])
    cosmopower.compute(theta)
