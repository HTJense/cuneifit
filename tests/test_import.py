import pytest  # noqa: F401


def test_import():
    """Test whether the package imports.
    """
    import cuneifit  # noqa: F401


def test_jax():
    """Test whether jax imports (mostly a deployment test).
    """
    import jax  # noqa: F401
    from jax import numpy as jnp  # noqa: F401
    from jax import scipy as jsc  # noqa: F401
