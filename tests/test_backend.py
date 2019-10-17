""" Test the FDTD backends """

## Imports

# 3rd party
import pytest

# This library
import fdtd
from fdtd.backend import NumpyBackend, TorchBackend


## Tests


def test_set_backend():
    fdtd.set_backend("torch")
    assert isinstance(fdtd.backend, TorchBackend)
    fdtd.set_backend("numpy")
    assert isinstance(fdtd.backend, NumpyBackend)
