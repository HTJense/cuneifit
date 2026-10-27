import pytest
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption("--emu-dir", action="store", default="jense_2024_emulators",
                     help="Path to the emulator directory.")

def pytest_configure(config):
    config.emu_dir = Path(config.getoption("--emu-dir")).resolve()
