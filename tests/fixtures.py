""" Often used pytest fixtures for the fdtd library """

## Imports
import fdtd
import pytest

## Fixtures

backend_names = [dict(bnd="numpy"), dict(bnd="torch")]

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


#Perform tests over entire
def backend_parametrizer(metafunc):

    # called once per each test function
    # params = {
    #     "test_current_detector": backends,
    # }
    if("all_bends" in metafunc.function.__name__):
        funcarglist = backend_names
    else:
        raise KeyError()

    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )
