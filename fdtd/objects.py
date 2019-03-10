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

    def __init__(self, permittivity: Tensorlike, name: str = None):
        """ Create an object """
        self.grid = None
        self.name = name
        self._permittivity = permittivity

    def _register_grid(self, grid: Grid, x: slice, y: slice, z: slice):
        """ Register a the object to the grid
        
        Args:
            grid: Grid: the grid to register the object into
            x: slice = None: the x-location of the object in the grid
            y: slice = None: the y-location of the object in the grid
            z: slice = None: the z-location of the object in the grid
        """
        self.grid = grid
        self.grid._objects.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        self.x = slice(
            grid._handle_distance(x.start), grid._handle_distance(x.stop), None
        )
        self.y = slice(
            grid._handle_distance(y.start), grid._handle_distance(y.stop), None
        )
        self.z = slice(
            grid._handle_distance(z.start), grid._handle_distance(z.stop), None
        )

        self.Nx = abs(self.x.stop - self.x.start)
        self.Ny = abs(self.y.stop - self.y.start)
        self.Nz = abs(self.z.stop - self.z.start)

        self.inverse_permittivity = (
            bd.ones((self.Nx, self.Ny, self.Nz, 3)) / self._permittivity
        )

        # set the permittivity values of the object at its border to be equal
        # to the grid permittivity. This way, the object is made symmetric.
        if self.Nx > 1:
            self.inverse_permittivity[-1, :, :, 0] = self.grid.inverse_permittivity[
                -1, self.y, self.z, 0
            ]
        if self.Ny > 1:
            self.inverse_permittivity[:, -1, :, 1] = self.grid.inverse_permittivity[
                self.x, -1, self.z, 1
            ]
        if self.Nz > 1:
            self.inverse_permittivity[:, :, -1, 2] = self.grid.inverse_permittivity[
                self.x, self.y, -1, 2
            ]

        self.grid.inverse_permittivity[self.x, self.y, self.z] = 0

    def update_E(self, curl_H):
        """ custom update equations for inside the object
        
        Args:
            curl_H: the curl of magnetic field in the grid.

        """
        loc = (self.x, self.y, self.z)
        self.grid.E[loc] += (
            self.grid.courant_number * self.inverse_permittivity * curl_H[loc]
        )

    def update_H(self, curl_E):
        """ custom update equations for inside the object
        
        Args:
            curl_E: the curl of electric field in the grid.

        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"


class AnisotropicObject(Object):
    """ An object with anisotropic permittivity tensor """

    def _register_grid(
        self, grid: Grid, x: slice = None, y: slice = None, z: slice = None
    ):
        """ Register a grid to the object 
        
        Args:
            grid: Grid: the grid to register the object into
            x: slice = None: the x-location of the object in the grid
            y: slice = None: the y-location of the object in the grid
            z: slice = None: the z-location of the object in the grid
        """
        super()._register_grid(grid=grid, x=x, y=y, z=z)
        eye = bd.zeros((self.Nx * self.Ny * self.Nz, 3, 3))
        eye[:, range(3), range(3)] = 1.0
        self.inverse_permittivity = bd.reshape(
            bd.reshape(self.inverse_permittivity, (-1, 1, 3)) * eye,
            (self.Nx, self.Ny, self.Nz, 3, 3),
        )

    def update_E(self, curl_H):
        """ custom update equations for inside the anisotropic object
        
        Args:
            curl_H: the curl of magnetic field in the grid.

        """
        loc = (self.x, self.y, self.z)
        self.grid.E[loc] += bd.reshape(
            self.grid.courant_number
            * bd.bmm(
                bd.reshape(self.inverse_permittivity, (-1, 3, 3)),
                bd.reshape(curl_H[loc], (-1, 3, 1)),
            ),
            (self.Nx, self.Ny, self.Nz, 3),
        )

    def update_H(self, curl_E):
        """ custom update equations for inside the anisotropic object
        
        Args:
            curl_E: the curl of electric field in the grid.

        """
        pass
