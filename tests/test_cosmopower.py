import pytest

COSMOPOWER_SKIP = True

try:
    import cosmopower as cp  # noqa: F401
    import cosmopower_jax.cosmopower_jax as cpj  # noqa: F401

    COSMOPOWER_SKIP = False
except Exception:
    pass


parser_filename = "jense_2023_cmb_camb_lcdm.yaml"
parser_rootdir = "jense_2024_emulators/"


@pytest.mark.skipif(COSMOPOWER_SKIP, reason="Cosmopower not installed")
def test_cosmopower_parser():
    cp.YAMLParser(parser_filename, rootdir=parser_rootdir)
