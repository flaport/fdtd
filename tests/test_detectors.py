""" Test the FDTD detectors """

## Imports
import fdtd
from fdtd.backend import backend as bd
from fdtd.waveforms import normalized_gaussian_pulse
from fdtd.conversions import *
from fdtd.sources import SoftArbitraryPointSource
from fdtd.detectors import CurrentDetector, BlockDetector
from fdtd.constants import X, Y, Z

import pytest
from fixtures import grid, pml, periodic_boundary
import matplotlib.pyplot as plt

## Tests

x_ = 4
y_ = 4
z_ = 4

# theory-driven development? there must be a more efficient way to test this kind of code...

# if/when detectors get polarizations,
# test in three axes - spin the grid to make sure everything's consistent

# each detector starts out with all the parameters empty ([])
# this causes edge cases when no time steps have passed, and should get explicit test cases


# @pytest.mark.skipif(skip_test_backend, reason="PyTorch is not installed.")
def test_DC_absorbing_current_detector_all_bends(grid, pml, backends):
    fdtd.set_backend(backends)

    # we maybe want our test cases to be orthogonal:
    # The test case with the patch antenna also relies on other modules like
    # the electric field detector & PEC, seems like a smaller test could be in order

    # However, because of the overlapping object issue, seems like this'll be harder.

    # This doesn't seem to work properly, not sure why-
    # see issue #11, will look into it.

    conductivity = 1e9
    # absorb = bd.ones((grid.Nx,grid.Ny,grid.Nz))
    # absorb[x_,y_,z_] = 0
    grid[x_ - 1 : x_ + 1, y_ - 1 : y_ + 1, z_ - 1 : z_ + 1] = fdtd.AbsorbingObject(
        permittivity=1.0, conductivity=conductivity
    )

    # AbsorbingObject doesn't create a magnetic field?
    # How could that ever cause a current density?

    # grid[x_,y_,z_] = fdtd.PointSource(period=500)
    cd = fdtd.CurrentDetector(name="Dave")
    grid[x_, y_, z_] = cd
    grid[x_:x_, y_:y_, z_:z_] = fdtd.BlockDetector()
    # FIXME: AbsorbingObject must be added last for some reason,

    n = 1000
    # grid.run(total_time=n)
    for _ in range(n):
        grid.update_E()
        grid.E[x_, y_, z_, Z] = normalized_gaussian_pulse(
            grid.time_passed,
            (n / 10.0) * grid.time_step,
            center=(n / 2) * grid.time_step,
        )
        grid.update_H()
        grid.time_steps_passed += 1

    #
    current_time_history = bd.array(grid.detectors[0].I).reshape(n)
    electric_field_time_history = bd.array(grid.detectors[1].E).reshape(n, 3)[:, Z]
    # print(grid.E[x_,y_,z_,Z])
    #
    # print(current_time_history)
    voltage = electric_field_time_history * grid.grid_spacing
    resistivity = 1.0 / conductivity
    resistance = (resistivity * grid.grid_spacing) / (
        grid.grid_spacing * grid.grid_spacing
    )
    expected_current = voltage / resistance
    # print(expected_current)
    # print(current_time_history)
    # plt.plot(current_time_history)
    # plt.show()
    # plt.plot(electric_field_time_history)


def test_CurrentDetector_shape(grid):
    edetector = BlockDetector()
    cdetector = CurrentDetector()
    grid[4, 4, 4] = cdetector
    grid[4, 4, 5] = edetector
    grid.run(10)
    # at the moment, edetector has an extra dimension for polarization
    # which isn't included in cdetector.
    assert bd.array(cdetector.I).shape[0:2] == bd.array(edetector.E).shape[0:2]
    assert bd.array(cdetector.I).shape[0] == 10


def test_SAPS_detector_register(grid):
    detector = SoftArbitraryPointSource(bd.zeros(1), impedance=50.0)
    grid[4, 4, 4] = detector
    grid.run(100)
    assert isinstance(grid.detectors[0], CurrentDetector)
