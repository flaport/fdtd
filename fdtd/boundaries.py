## Imports

# typing
from numbers import Number

# relative
from .grid import Grid
from .backend import backend as bd


## Boundary Conditions [base class]
class Boundary:
    """ an FDTD Boundary [base class] """

    def __init__(self):
        """ Create a boundary """
        self.grid = None  # will be registered later

    def register_grid(self, grid: Grid):
        """ Register a grid to the boundary """
        self.grid = grid

    def update_phi_E(self):
        """ Update convolution [phi_E]

        Note:
            this method is called *before* the electric field is updated
        """
        pass

    def update_phi_H(self):
        """ Update convolution [phi_H]

        Note:
            this method is called *before* the magnetic field is updated
        """
        pass

    def update_E(self):
        """ Update electric field of the grid

        Note:
            this method is called *after* the grid fields are updated
        """
        pass

    def update_H(self):
        """ Update magnetic field of the grid

        Note:
            this method is called *after* the grid fields are updated
        """
        pass

    def __repr__(self):
        return self.__class__.__name__.lower()


## Periodic Boundaries

# Periodic Boundaries in the X-direction
class PeriodicBoundaryX(Boundary):
    def update_E(self):
        """ Update electric field such that periodic boundary conditions in the
        X-direction apply """
        self.grid.E[-1, :, :, 0] = self.grid.E[0, :, :, 0]
        self.grid.E[0, :, :, 1] = self.grid.E[-1, :, :, 1]
        self.grid.E[0, :, :, 2] = self.grid.E[-1, :, :, 2]

    def update_H(self):
        """ Update magnetic field such that periodic boundary conditions in the
        X-directions apply """
        self.grid.H[0, :, :, 0] = self.grid.H[-1, :, :, 0]
        self.grid.H[-1, :, :, 1] = self.grid.H[0, :, :, 1]
        self.grid.H[-1, :, :, 2] = self.grid.H[0, :, :, 2]


# Periodic Boundaries in the Y-direction
class PeriodicBoundaryY(Boundary):
    def update_E(self):
        """ Update electric field such that periodic boundary conditions in the
        Y-direction apply """
        self.grid.E[:, 0, :, 0] = self.grid.E[:, -1, :, 0]
        self.grid.E[:, -1, :, 1] = self.grid.E[:, 0, :, 1]
        self.grid.E[:, 0, :, 2] = self.grid.E[:, -1, :, 2]

    def update_H(self):
        """ Update magnetic field such that periodic boundary conditions in the
        Y-direction apply """
        self.grid.H[:, -1, :, 0] = self.grid.H[:, 0, :, 0]
        self.grid.H[:, 0, :, 1] = self.grid.H[:, -1, :, 1]
        self.grid.H[:, -1, :, 2] = self.grid.H[:, 0, :, 2]


# Periodic Boundaries in the Z-direction
class PeriodicBoundaryZ(Boundary):
    def update_E(self):
        """ Update electric field such that periodic boundary conditions in the
        Z-direction apply """
        self.grid.E[:, :, 0, 0] = self.grid.E[:, :, -1, 0]
        self.grid.E[:, :, 0, 1] = self.grid.E[:, :, -1, 1]
        self.grid.E[:, :, -1, 2] = self.grid.E[:, :, 0, 2]

    def update_H(self):
        """ Update magnetic field such that periodic boundary conditions in the
        Z-direction apply """
        self.grid.H[:, :, -1, 0] = self.grid.H[:, :, 0, 0]
        self.grid.H[:, :, -1, 1] = self.grid.H[:, :, 0, 1]
        self.grid.H[:, :, 0, 2] = self.grid.H[:, :, -1, 2]


