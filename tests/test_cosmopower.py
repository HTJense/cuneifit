import pytest
from pathlib import Path

COSMOPOWER_SKIP = True

try:
    import cosmopower as cp  # noqa: F401
    import cosmopower_jax.cosmopower_jax as cpj  # noqa: F401

    COSMOPOWER_SKIP = False
except Exception:
    pass


parser_filename = "jense_2023_cmb_camb_lcdm.yaml"


@pytest.mark.skipif(COSMOPOWER_SKIP, reason="Cosmopower not installed")
def test_cosmopower_parser(request):
    root_dir = request.config.getoption("--emu-dir")
    cp.YAMLParser(parser_filename, root_dir=root_dir)
