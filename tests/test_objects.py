""" Test the FDTD objects """


## Imports
import fdtd
import pytest

## Tests


#
# def test_object_grid_overlap(grid, pml):
#     grid[x_-1:x_+1,y_-1:y_+1,z_-1:z_+1] = fdtd.AbsorbingObject(permittivity=1.0,
#     conductivity=conductivity)
#     # grid[x_,y_,z_] = fdtd.PointSource(period=500)
#
