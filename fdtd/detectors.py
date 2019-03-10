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
        self,
        grid: Grid,
        x: number_list_or_slice = None,
        y: number_list_or_slice = None,
        z: number_list_or_slice = None,
    ):
        """ Register a grid to the detector
        
        Args:
            grid: Grid: the grid to place the detector into 
            x: slice = None: the x-location in the grid
            y: slice = None: the y-location in the grid
            z: slice = None: the z-location in the grid
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

        if isinstance(x, Number):
            self.x = [grid._handle_distance(x)]
        elif not isinstance(x, slice):
            self.x = [grid._handle_distance(xx) for xx in x]
        else:
            self.x = x

        if isinstance(y, Number):
            self.y = [grid._handle_distance(y)]
        elif not isinstance(y, slice):
            self.y = [grid._handle_distance(yy) for yy in y]
        else:
            self.y = y

        if isinstance(z, Number):
            self.z = [grid._handle_distance(z)]
        elif not isinstance(z, slice):
            self.z = [grid._handle_distance(zz) for zz in z]
        else:
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
