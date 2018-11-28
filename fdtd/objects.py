## Imports

# typing
from numbers import Number

# relative
from .grid import Grid
from .backend import backend as bd
from .backend import Tensorlike


## Object
class Object:
    """ An object to place in the grid """

    def __init__(self, permittivity: Tensorlike, x: slice, y: slice, z: slice):
        """ Create an object """
        self.grid = None
        self.x = x
        self.y = y
        self.z = z
        self._permittivity = permittivity

    def register_grid(self, grid: Grid):
        """ Register a grid to the object """
        self.grid = grid

        self.x = slice(
            grid._handle_distance(self.x.start),
            grid._handle_distance(self.x.stop),
            None,
        )
        self.y = slice(
            grid._handle_distance(self.y.start),
            grid._handle_distance(self.y.stop),
            None,
        )
        self.z = slice(
            grid._handle_distance(self.z.start),
            grid._handle_distance(self.z.stop),
            None,
        )

        self.Nx = abs(self.x.stop - self.x.start)
        self.Ny = abs(self.y.stop - self.y.start)
        self.Nz = abs(self.z.stop - self.z.start)

        self.inverse_permittivity = (
            bd.ones((self.Nx, self.Ny, self.Nz, 3))/self._permittivity
        )

        if self.Nx > 1:
            self.inverse_permittivity[-1,:,:,0] = self.grid.inverse_permittivity[-1,self.y,self.z,0]
        if self.Ny > 1:
            self.inverse_permittivity[:,-1,:,1] = self.grid.inverse_permittivity[self.x,-1,self.z,1]
        if self.Nz > 1:
            self.inverse_permittivity[:,:,-1,2] = self.grid.inverse_permittivity[self.x,self.y,-1,1]

        self.grid.inverse_permittivity[self.x, self.y, self.z] = 0

    def update_E(self, curl_H):
        loc = (self.x, self.y, self.z)
        self.grid.E[loc] += (
            self.grid.courant_number * self.inverse_permittivity * curl_H[loc]
        )

    def update_H(self, curl_E):
        pass
