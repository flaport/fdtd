""" Test the FDTD sources """


## Imports
import fdtd
import pytest

from fixtures import grid, pml, periodic_boundary
from fdtd.sources import SoftArbitraryPointSource
from fdtd.sources import PlaneSource
## Tests


def test_SoftArbitraryPointSource(grid):
    pass

def test_PlaneSource_polarization_error(grid):
    with pytest.raises(ValueError):
        grid[0, :, :] = PlaneSource(polarization='x')
    with pytest.raises(ValueError):
        grid[:, 0, :] = PlaneSource(polarization='y')
    with pytest.raises(ValueError):
        grid[:, :, 0] = PlaneSource(polarization='z')

def test_PlaneSource_polarization_inference(grid):
    ps = PlaneSource(polarization='y')
    grid[0, :, :] = ps
    assert ps._Epol == 1 and ps._Hpol == 2
    ps = PlaneSource(polarization='z')
    grid[0, :, :] = ps
    assert ps._Epol == 2 and ps._Hpol == 1
    ps = PlaneSource(polarization='x')
    grid[:, 0, :] = ps
    assert ps._Epol == 0 and ps._Hpol == 2
    ps = PlaneSource(polarization='z')
    grid[:, 0, :] = ps
    assert ps._Epol == 2 and ps._Hpol == 0
    ps = PlaneSource(polarization='x')
    grid[:, :, 0] = ps
    assert ps._Epol == 0 and ps._Hpol == 1
    ps = PlaneSource(polarization='y')
    grid[:, :, 0] = ps
    assert ps._Epol == 1 and ps._Hpol == 0
