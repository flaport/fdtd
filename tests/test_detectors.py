""" Test the FDTD detectors """

# To run tests in a conda env, use python -m pytest in the root
# to specify a test,  -k "test_current_detector"
# To view output, -rA
# to run with all backends, --all_backends.
# with --all_backends, --maxfail=1 is recommended to avoid blowing up the scrollback!

## Imports
import fdtd
from fdtd.backend import backend as bd
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
    cd = fdtd.CurrentDetector(name="Dave")
    grid[x_,y_,z_] = cd
    grid[x_:x_,y_:y_,z_:z_] = fdtd.BlockDetector()
    conductivity = 1.0
    grid[x_,y_,z_] = fdtd.AbsorbingObject(permittivity=1.0, conductivity=conductivity)
    # FIXME: AbsorbingObject must be added last for some reason,


    n = 100
    grid.run(total_time=n)
    #
    current_time_history = grid.detectors[0].I
    electric_field_time_history = bd.array(grid.detectors[1].E).reshape(n,3)[:,d_.Z]
    # print(grid.E[x_,y_,z_,d_.Z])
    print(cd.single_point_current(x_,y_,z_))
    print(max(electric_field_time_history))

    print(current_time_history)

    voltage = max(electric_field_time_history)*grid.grid_spacing
    resistivity = 1.0/conductivity
    resistance = (resistivity * grid.grid_spacing) / (grid.grid_spacing*grid.grid_spacing)
    expected_current = voltage / resistance
    print(expected_current)
