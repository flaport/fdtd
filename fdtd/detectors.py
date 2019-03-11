""" FDTD Detectors

Detectors for the FDTD Grid. Available Detectors:

 - Detector

"""

## Imports

# typing
from .typing import ListOrSlice

# relative
from .grid import Grid

## Detector
class Detector:
    """ an FDTD Detector """

    def __init__(self, name=None):
        """ Create a detector

        Args:
            name: str=None: name of the Detector
        """
        self.grid = None  # will be registered later
        self.E = []
        self.H = []
        self.name = name

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        """ Register a grid to the detector

        Args:
            grid: the grid to place the detector into
            x: the x-location in the grid
            y: the y-location in the grid
            z: the z-location in the grid
        """
        self.grid = grid
        self.grid._detectors.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )
        self.x = x
        self.y = y
        self.z = z

    def detect_E(self):
        """ detect the electric field at a certain location in the grid """
        E = self.grid.E[self.x, self.y, self.z]
        self.E.append(E)

    def detect_H(self):
        """ detect the magnetic field at a certain location in the grid """
        H = self.grid.H[self.x, self.y, self.z]
        self.H.append(H)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"
