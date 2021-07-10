""" Test the FDTD boundaries """

## Imports

# 3rd party
import pytest

# fdtd
import fdtd

# fixtures
from fixtures import grid, pml, periodic_boundary
from fdtd.backend import backend as bd

from fdtd.boundaries import DomainBorderPML
## Tests


def test_periodic_boundary_in_grid_boundary_list(grid, periodic_boundary):
    grid[0, :, :] = periodic_boundary
    assert periodic_boundary in grid.boundaries


def test_periodic_boundary_raises_error_when_indexed_with_slice(
    grid, periodic_boundary
):
    with pytest.raises(IndexError):
        grid[0:2, :, :] = periodic_boundary


def test_periodic_boundary_placed_in_middle_of_grid(grid, periodic_boundary):
    with pytest.raises(IndexError):
        grid[2, :, :] = periodic_boundary


def test_pml_in_grid_boundary_list(grid, pml):
    grid[0:3, :, :] = pml
    assert pml in grid.boundaries


def test_pml_placed_in_middle_of_grid(grid, pml):
    with pytest.raises(IndexError):
        grid[2:4, :, :] = pml

def test_pml_in_grid_boundary_list(grid, pml):
    grid[0:3, :, :] = pml
    assert pml in grid.boundaries

def test_DomainPML_exception(grid, pml):
    with pytest.raises(IndexError):
        DomainBorderPML(grid, grid.Nx//2+1)

def test_DomainPML_success(grid, pml):
    DomainBorderPML(grid, 3)

def test_DomainPML_bonehead_indexing():
    #make sure I did the indexing properly!
    #no overlaps.
    test_grid = bd.zeros((9,9,9))
    border_cells = 4

    #top and bottom
    test_grid[:, : ,0:border_cells] += 1
    test_grid[:, : ,-border_cells:] += 1

    #left and right
    test_grid[0:border_cells, :, border_cells:-border_cells] += 1
    test_grid[-border_cells:, :, border_cells:-border_cells] += 1

    #front and back
    test_grid[border_cells:-border_cells, 0:border_cells, border_cells:-border_cells] += 1
    test_grid[border_cells:-border_cells, -border_cells:, border_cells:-border_cells] += 1

    comparison_grid = bd.ones((9,9,9))
    comparison_grid[4,4,4] = 0

    assert (test_grid == comparison_grid).all()

# test non-unique names
