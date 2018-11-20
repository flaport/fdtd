## Imports

# typing
from numbers import Number
from typing import Union

# relative
from .grid import Grid
from .backend import backend as bd


## Types
number_list_or_slice = Union[Number, list, slice]

## Detector
class Detector:
    """ an FDTD Detector """

    def __init__(
        self, x: number_list_or_slice, y: number_list_or_slice, z: number_list_or_slice
    ):
        """ Create a detector """
        self.grid = None  # will be registered later
        self.x = x
        self.y = y
        self.z = z
        self.E = []
        self.H = []

    def register_grid(self, grid: Grid):
        """ Register a grid to the boundary """
        self.grid = grid

        if isinstance(self.x, Number):
            self.x = [grid._handle_distance(self.x)]
        elif not isinstance(self.x, slice):
            self.x = [grid._handle_distance(xx) for xx in self.x]

        if isinstance(self.y, Number):
            self.y = [grid._handle_distance(self.y)]
        elif not isinstance(self.y, slice):
            self.y = [grid._handle_distance(yy) for yy in self.y]

        if isinstance(self.z, Number):
            self.z = [grid._handle_distance(self.z)]
        elif not isinstance(self.z, slice):
            self.z = [grid._handle_distance(zz) for zz in self.z]

    def detect_E(self):
        """ detect the electric field at a certain location in the grid """
        E = self.grid.E[self.x, self.y, self.z]
        self.E.append(E)

    def detect_H(self):
        """ detect the magnetic field at a certain location in the grid """
        H = self.grid.H[self.x, self.y, self.z]
        self.H.append(H)

    def __repr__(self):
        return self.__class__.__name__.lower()
