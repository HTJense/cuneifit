import os
import pytest

COSMOPOWER_SKIP = True

try:
    import cosmopower as cp  # noqa: F401
    import cosmopower_jax.cosmopower_jax as cpj  # noqa: F401

    COSMOPOWER_SKIP = False
except Exception:
    pass


parser_filename = "jense_2023_cmb_camb_lcdm.yaml"


@pytest.mark.skipif(COSMOPOWER_SKIP, reason="Cosmopower not installed")
def test_cosmopower_parser(emu_dir):
    """Test whether we can load data for cosmopower.
    """
    cp.YAMLParser(os.path.join(emu_dir, parser_filename), root_dir=emu_dir)
