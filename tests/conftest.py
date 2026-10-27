import pytest
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--emu-dir", action="store", default=None,
                     help="Path to the emulator directory.")


@pytest.fixture(scope="session")
def emu_dir(request):
    emu_dir = request.config.getoption("--emu-dir")

    if emu_dir is None:
        pytest.skip("Skipping test: Emulator dir was not provided")

    return emu_dir
