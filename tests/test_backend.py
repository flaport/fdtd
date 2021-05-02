""" Test the FDTD backends """

## Imports

# 3rd party
import pytest

# This library
import fdtd

try:
    from fdtd.backend import NumpyBackend, TorchBackend
    skip_test_backend = False
except ImportError:
    skip_test_backend = True


## Tests


@pytest.mark.skipif(skip_test_backend, reason="PyTorch is not installed.")
def test_set_backend():
    fdtd.set_backend("torch")
    assert isinstance(fdtd.backend, TorchBackend)
    fdtd.set_backend("numpy")
    assert isinstance(fdtd.backend, NumpyBackend)

