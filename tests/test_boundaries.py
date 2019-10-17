""" Test the FDTD boundaries """

## Imports

# 3rd party
import pytest

# fdtd
import fdtd

# fixtures
from fixtures import grid, pml, periodic_boundary

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
