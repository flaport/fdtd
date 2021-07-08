""" Test the FDTD detectors """

#To run tests with conda, python -m pytest
# to specify a test,  -k "test_current_detector".
# To view output, -rA
# to run with all backends, --all_backends.

## Imports
import fdtd
import pytest
from fixtures import grid, pml, periodic_boundary, backend_parametrizer
from fdtd.grid import d_
## Tests

x_ = 4
y_ = 4
z_ = 4


pytest_generate_tests = backend_parametrizer # perform tests over all backends

# @pytest.mark.skipif(skip_test_backend, reason="PyTorch is not installed.")
def test_current_detector_all_bends(grid, pml, backends):
    fdtd.set_backend(backends)

    grid[x_,y_,z_] = fdtd.PointSource()
    grid[x_,y_,z_] = fdtd.CurrentDetector(name="Dave")
    grid[x_:x_,y_:y_,z_:z_] = fdtd.BlockDetector()
    grid[x_,y_,z_] = fdtd.AbsorbingObject(permittivity=1.0, conductivity=1.0)
    # FIXME: AbsorbingObject must be added last for some reason,

    grid.run(total_time=100)
    #
    current_time_history = grid.detectors[0].I
    electric_field_time_history = grid.detectors[1].E[:][d_.Z]
    # print(grid.E[x_,y_,z_,d_.Z])
    print(current_time_history[-1])
    print(electric_field_time_history)
