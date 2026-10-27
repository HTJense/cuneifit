import cuneifit
import pytest  # noqa: F401
import pathlib

rootdir = pathlib.Path(__file__).parent.resolve() / "test_yamls"


def test_load_yaml():
    result = cuneifit.config.config_from_file(str(rootdir / "test1.yaml"))

    assert result["x"] == 1
    assert result["y"] == 2


def test_nested_yamls():
    result = cuneifit.config.config_from_file(str(rootdir / "test2.yaml"))
    old_data = cuneifit.config.config_from_file(str(rootdir / "test1.yaml"))

    for key, value in old_data.items():
        assert result["old_data"][key] == value


def test_recursive_nested_yamls():
    result = cuneifit.config.config_from_file(str(rootdir / "test3.yaml"))

    for key, value in result["older_data"]["old_data"].items():
        assert result["older_data"][key] == value
