""" Often used pytest fixtures for the fdtd library """

## Imports
import fdtd
import pytest
from fdtd.backend import backend_names





## Fixtures


@pytest.fixture
def grid():
    grid = fdtd.Grid(
        shape=(10, 10, 10),
        grid_spacing=100e-9,
        permittivity=1.0,
        permeability=1.0,
        courant_number=None,  # calculate the courant number
    )
    return grid


@pytest.fixture
def periodic_boundary():
    periodic_boundary = fdtd.PeriodicBoundary(name="periodic_boundary")
    return periodic_boundary


@pytest.fixture
def pml():
    pml = fdtd.PML(name="PML")
    return pml


# Perform tests over all backends when pytest called with --all_backends
# and function name has "all_bends" in it.
# Function must have (, backends) in args and
# rename to all_bds for consistency?
# but "all_bends" is, frankly, hilarious
def backend_parametrizer(metafunc):
    # called once per each test function
    # see https://docs.pytest.org/en/6.2.x/example/parametrize.html
    if("all_bends" in metafunc.function.__name__):
        if(metafunc.config.getoption("all_backends")):
            funcarglist = backend_names
            argnames = sorted(funcarglist[0])
            metafunc.parametrize(
                argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
            )
        else:
            funcarglist = [backend_names[0]]
            argnames = sorted(funcarglist[0])
            metafunc.parametrize(
                argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
            )
