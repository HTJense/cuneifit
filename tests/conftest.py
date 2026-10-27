import os
from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption("--emu-dir", action="store", default=None,
                     help="Path to the emulator directory.")
    parser.addoption("--dr6-dir", action="store", default=None,
                     help="Path to the ACT DR6 data.")


@pytest.fixture(scope="session")
def emu_dir(request):
    emu_dir = request.config.getoption("--emu-dir")

    if emu_dir is None:
        pytest.skip("Skipping test: Emulator dir was not provided")

    return str(Path(emu_dir).resolve())


@pytest.fixture(scope="session")
def dr6_dir(request):
    dr6_dir = (request.config.getoption("--dr6-dir") or
               os.getenv("CUNEIFIT_DATA_DIR", None))

    if dr6_dir is None:
        pytest.skip("Skipping test: ACT DR6 dir was not provided")

    dr6_dir = str(Path(dr6_dir).resolve())
    os.environ["CUNEIFIT_DATA_DIR"] = dr6_dir
    return dr6_dir
