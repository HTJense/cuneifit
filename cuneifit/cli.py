import argparse
from . import config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="Cuneifit",
        description="Cuneifit --- Differentiable Inference for Simons Observatory"  # noqa: E501
    )
    parser.add_argument("run_file", action="store",
                        help="A run configuration to analyze.")

    args = parser.parse_args(argv)

    run_config = config.
