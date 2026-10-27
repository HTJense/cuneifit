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
    """
    cp.YAMLParser(os.path.join(emu_dir, parser_filename), root_dir=emu_dir)
