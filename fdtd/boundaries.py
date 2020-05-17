""" Boundaries for the FDTD Grid.

Available Boundaries:

 - PeriodicBoundary
 - PML

"""
## Imports

# typing
from .typing import Tensorlike, ListOrSlice, IntOrSlice

# relative
from .grid import Grid
from .backend import backend as bd


## Boundary Conditions [base class]
class Boundary:
    """ an FDTD Boundary [base class] """

    def __init__(self, name: str = None):
        """ Create a boundary

        Args:
            name: name of the boundary
        """
        self.grid = None  # will be registered later
        self.name = name

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        """ Register a grid to the boundary

        Args:
            grid: the grid to register the boundary to
            x: the x-location of the boundary
            y: the y-location of the boundary
            z: the z-location of the boundary

        Note:
            This is a helper method. To register the boundary to the grid,
            index the grid and assign to it:
            grid[0,:,:] = Boundary(name="boundary_name")
        """
        self.grid = grid
        self.grid.boundaries.append(self)
        self.x = self._handle_slice(x)
        self.y = self._handle_slice(y)
        self.z = self._handle_slice(z)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

    def _handle_slice(self, s: ListOrSlice) -> IntOrSlice:
        if isinstance(s, list):
            if len(s) > 1:
                raise ValueError(
                    "Use slices or single numbers to index the grid for a boundary"
                )
            return s[0]
        if isinstance(s, slice):
            if (
                s.start is not None
                and s.stop is not None
                and (s.start == s.stop or abs(s.start - s.stop) == 1)
            ):
                return s.start
            return s
        raise ValueError("Invalid grid indexing used for boundary")

    def update_phi_E(self):
        """ Update convolution [phi_E]

        Note:
            this method is called *before* the electric field is updated
        """

    def update_phi_H(self):
        """ Update convolution [phi_H]

        Note:
            this method is called *before* the magnetic field is updated
        """

    def update_E(self):
        """ Update electric field of the grid

        Note:
            this method is called *after* the grid fields are updated
        """

    def update_H(self):
        """ Update magnetic field of the grid

        Note:
            this method is called *after* the grid fields are updated
        """

    def __repr__(self):
        return f"PML(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"

        def _handle_slice(s):
            return (
                str(s)
                .replace("slice(", "")
                .replace(")", "")
                .replace(", ", ":")
                .replace("None", "")
            )

        x = _handle_slice(self.x)
        y = _handle_slice(self.y)
        z = _handle_slice(self.z)
        s += f"        @ x={x}, y={y}, z={z}".replace(":,", ",")
        if s[-1] == ":":
            s = s[:-1]
        return s + "\n"


## Periodic Boundaries
class PeriodicBoundary(Boundary):
    """ An FDTD Periodic Boundary

    Note:
        Registering a periodic boundary to the grid will change the periodic
        boundary in one of its subclasses: ``_PeriodicBoundaryX``,
        ``_PeriodicBoundaryY`` or ``_PeriodicBoundaryY``, depending on the
        position in the grid.
    """

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        super()._register_grid(grid=grid, x=x, y=y, z=z)

        if self.x == 0 or self.x == -1:
            self.__class__ = _PeriodicBoundaryX  # subclass of PeriodicBoundary
            if hasattr(grid, "_xlow_boundary") or hasattr(grid, "_xhigh_boundary"):
                raise AttributeError("grid already has an xlow/xhigh boundary!")
            setattr(grid, "_xlow_boundary", self)
            setattr(grid, "_xhigh_boundary", self)
        elif self.y == 0 or self.y == -1:
            self.__class__ = _PeriodicBoundaryY  # subclass of PeriodicBoundary
            if hasattr(grid, "_ylow_boundary") or hasattr(grid, "_yhigh_boundary"):
                raise AttributeError("grid already has an ylow/yhigh boundary!")
            setattr(grid, "_ylow_boundary", self)
            setattr(grid, "_yhigh_boundary", self)
        elif self.z == 0 or self.z == -1:
            self.__class__ = _PeriodicBoundaryZ  # subclass of PeriodicBoundary
            if hasattr(grid, "_zlow_boundary") or hasattr(grid, "_zhigh_boundary"):
                raise AttributeError("grid already has an zlow/zhigh boundary!")
            setattr(grid, "_zlow_boundary", self)
            setattr(grid, "_zhigh_boundary", self)
        else:
            raise IndexError(
                "A periodic boundary should be placed at the boundary of the "
                "grid using a single index (either 0 or -1)"
            )


# Periodic Boundaries in the X-direction
class _PeriodicBoundaryX(PeriodicBoundary):
    def update_E(self):
        """ Update electric field such that periodic boundary conditions in the
        X-direction apply """
        self.grid.E[0, :, :, :] = self.grid.E[-1, :, :, :]

    def update_H(self):
        """ Update magnetic field such that periodic boundary conditions in the
        X-directions apply """
        self.grid.H[-1, :, :, :] = self.grid.H[0, :, :, :]


# Periodic Boundaries in the Y-direction
class _PeriodicBoundaryY(PeriodicBoundary):
    def update_E(self):
        """ Update electric field such that periodic boundary conditions in the
        Y-direction apply """
        self.grid.E[:, 0, :, :] = self.grid.E[:, -1, :, :]

    def update_H(self):
        """ Update magnetic field such that periodic boundary conditions in the
        Y-direction apply """
        self.grid.H[:, -1, :, :] = self.grid.H[:, 0, :, :]


# Periodic Boundaries in the Z-direction
class _PeriodicBoundaryZ(PeriodicBoundary):
    def update_E(self):
        """ Update electric field such that periodic boundary conditions in the
        Z-direction apply """
        self.grid.E[:, :, 0, :] = self.grid.E[:, :, -1, :]

    def update_H(self):
        """ Update magnetic field such that periodic boundary conditions in the
        Z-direction apply """
        self.grid.H[:, :, -1, :] = self.grid.H[:, :, 0, :]


## Perfectly Matched Layer (PML)


class PML(Boundary):
    """ A perfectly matched layer (PML)

    a PML is an impedence-matched area at the boundary of the grid for which
    all fields incident perpendicular to the area are absorbed without
    reflection.

    Note:
        Registering a PML to the grid will monkeypatch the PML to become one of
        its subclasses: ``_PMLXlow``, ``_PMLYlow`` or ``_PMLZlow``,
        ``_PMLXhigh``, ``_PMLYhigh``, ``_PMLZhigh`` depending on the position
        in the grid.
    """

    def __init__(self, a: float = 1e-8, name: str = None):
        """ Perfectly Matched Layer

        Args:
            a: stability parameter
            name: name of the PML
        """
        super().__init__(name=name)

        # TODO: to make this a PML parameter, the *normal* curl equations need to be updated
        self.k = 1.0
        self.thickness = 0

        self.a = a

    def _set_locations(self):
        """ helper function

        sets:
            self.loc: the location of the PML
            self.locx: the location of the PML (only x coordinates)
            self.locy: the location of the PML (only y coordinates)
            self.locz: the location of the PML (only z coordinates)
        """
        raise NotImplementedError

    def _set_shape(self):
        """ helper function

        sets:
            self.shape: the shape of the PML
        """
        raise NotImplementedError

    def _set_sigmaE(self):
        """ helper function

        sets:
            self.sigmaE: the electric conductivity (responsible for the absorption) of
                the PML
        """
        raise NotImplementedError

    def _set_sigmaH(self):
        """ helper function

        sets:
            self.sigmaH: the magnetic conductivity (responsible for the absorption) of
                the PML
        """
        raise NotImplementedError

    def _sigma(self, vect: Tensorlike):
        """ create a cubicly increasing profile for the conductivity """
        return 40 * vect ** 3 / (self.thickness + 1) ** 4

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):

        super()._register_grid(grid=grid, x=x, y=y, z=z)

        if (
            (self.x.start is None or self.x.start == 0)
            and (self.x.stop is not None)
            and (self.x.stop > 0)
        ):
            self.__class__ = _PMLXlow
            if hasattr(grid, "_xlow_boundary"):
                raise AttributeError("grid already has an xlow boundary!")
            setattr(grid, "_xlow_boundary", self)
            self._calculate_parameters(thickness=self.x.stop)
        elif (
            (self.x.start is not None) and (self.x.stop is None) and (self.x.start < 0)
        ):
            self.__class__ = _PMLXhigh
            if hasattr(grid, "_xhigh_boundary"):
                raise AttributeError("grid already has an xhigh boundary!")
            setattr(grid, "_xhigh_boundary", self)
            self._calculate_parameters(thickness=-self.x.start)
        elif (
            (self.y.start is None or self.y.start == 0)
            and (self.y.stop is not None)
            and (self.y.stop > 0)
        ):
            self.__class__ = _PMLYlow
            if hasattr(grid, "_ylow_boundary"):
                raise AttributeError("grid already has an ylow boundary!")
            setattr(grid, "_ylow_boundary", self)
            self._calculate_parameters(thickness=self.y.stop)
        elif (
            (self.y.start is not None) and (self.y.stop is None) and (self.y.start < 0)
        ):
            self.__class__ = _PMLYhigh
            if hasattr(grid, "_yhigh_boundary"):
                raise AttributeError("grid already has an yhigh boundary!")
            setattr(grid, "_yhigh_boundary", self)
            self._calculate_parameters(thickness=-self.y.start)
        elif (
            (self.z.start is None or self.z.start == 0)
            and (self.z.stop is not None)
            and (self.z.stop > 0)
        ):
            self.__class__ = _PMLZlow
            if hasattr(grid, "_zlow_boundary"):
                raise AttributeError("grid already has an zlow boundary!")
            setattr(grid, "_zlow_boundary", self)
            self._calculate_parameters(thickness=self.z.stop)
        elif (
            (self.z.start is not None) and (self.z.stop is None) and (self.z.start < 0)
        ):
            self.__class__ = _PMLZhigh
            if hasattr(grid, "_zhigh_boundary"):
                raise AttributeError("grid already has an zhigh boundary!")
            setattr(grid, "_zhigh_boundary", self)
            self._calculate_parameters(thickness=-self.z.start)
        else:
            raise IndexError(
                "not a valid slice for a PML. Make sure the slice is at the border of the PML"
            )

    def _handle_slice(self, s: ListOrSlice) -> slice:
        if isinstance(s, list):
            raise ValueError("One can only use slices to index the grid for a PML")
        if isinstance(s, slice):
            return s
        raise ValueError("Invalid grid indexing used for boundary")

    def _calculate_parameters(self, thickness: int = 10):
        """ Calculate the parameters for the PML

        Args:
            thickness=10: The thickness of the PML. The thickness can be specified as
                integer [gridpoints] or as float [meters].
        """

        self.thickness = thickness

        # set orientation dependent parameters: (different for x, y, z-PML)
        # NOTE: these methods are implemented by the subclasses of PML.
        self._set_locations()
        self._set_shape()
        self._set_sigmaE()
        self._set_sigmaH()

        # set the other parameters
        Nx, Ny, Nz = self.shape  # is defined by _set_shape()
        self.phi_E = bd.zeros((Nx, Ny, Nz, 3))
        self.phi_H = bd.zeros((Nx, Ny, Nz, 3))
        self.psi_Ex = bd.zeros((Nx, Ny, Nz, 3))
        self.psi_Ey = bd.zeros((Nx, Ny, Nz, 3))
        self.psi_Ez = bd.zeros((Nx, Ny, Nz, 3))
        self.psi_Hx = bd.zeros((Nx, Ny, Nz, 3))
        self.psi_Hy = bd.zeros((Nx, Ny, Nz, 3))
        self.psi_Hz = bd.zeros((Nx, Ny, Nz, 3))

        self.bE = bd.exp(-(self.sigmaE / self.k + self.a) * self.grid.courant_number)
        self.cE = (
            (self.bE - 1.0)
            * self.sigmaE  # is defined by _set_sigmaE()
            / (self.sigmaE * self.k + self.a * self.k ** 2)
        )

        self.bH = bd.exp(-(self.sigmaH / self.k + self.a) * self.grid.courant_number)
        self.cH = (
            (self.bH - 1.0)
            * self.sigmaH  # is defined by _set_sigmaH()
            / (self.sigmaH * self.k + self.a * self.k ** 2)
        )

    def update_E(self):
        """ Update electric field of the grid

        Note:
            this method is called *after* the electric field is updated
        """
        self.grid.E[self.loc] += (
            self.grid.courant_number
            * self.grid.inverse_permittivity[self.loc]
            * self.phi_E
        )

    def update_H(self):
        """ Update magnetic field of the grid

        Note:
            this method is called *after* the magnetic field is updated
        """
        self.grid.H[self.loc] -= (
            self.grid.courant_number
            * self.grid.inverse_permeability[self.loc]
            * self.phi_H
        )

    def update_phi_E(self):
        """ Update convolution [phi_E]

        Note:
            this method is called *before* the electric field is updated
        """
        self.psi_Ex *= self.bE
        self.psi_Ey *= self.bE
        self.psi_Ez *= self.bE

        c = self.cE
        Hx = self.grid.H[self.locx]
        Hy = self.grid.H[self.locy]
        Hz = self.grid.H[self.locz]

        self.psi_Ex[:, 1:, :, 1] += (Hz[:, 1:, :] - Hz[:, :-1, :]) * c[:, 1:, :, 1]
        self.psi_Ex[:, :, 1:, 2] += (Hy[:, :, 1:] - Hy[:, :, :-1]) * c[:, :, 1:, 2]

        self.psi_Ey[:, :, 1:, 2] += (Hx[:, :, 1:] - Hx[:, :, :-1]) * c[:, :, 1:, 2]
        self.psi_Ey[1:, :, :, 0] += (Hz[1:, :, :] - Hz[:-1, :, :]) * c[1:, :, :, 0]

        self.psi_Ez[1:, :, :, 0] += (Hy[1:, :, :] - Hy[:-1, :, :]) * c[1:, :, :, 0]
        self.psi_Ez[:, 1:, :, 1] += (Hx[:, 1:, :] - Hx[:, :-1, :]) * c[:, 1:, :, 1]

        self.phi_E[..., 0] = self.psi_Ex[..., 1] - self.psi_Ex[..., 2]
        self.phi_E[..., 1] = self.psi_Ey[..., 2] - self.psi_Ey[..., 0]
        self.phi_E[..., 2] = self.psi_Ez[..., 0] - self.psi_Ez[..., 1]

    def update_phi_H(self):
        """ Update convolution [phi_H]

        Note:
            this method is called *before* the magnetic field is updated
        """
        self.psi_Hx *= self.bH
        self.psi_Hy *= self.bH
        self.psi_Hz *= self.bH

        c = self.cH
        Ex = self.grid.E[self.locx]
        Ey = self.grid.E[self.locy]
        Ez = self.grid.E[self.locz]

        self.psi_Hx[:, :-1, :, 1] += (Ez[:, 1:, :] - Ez[:, :-1, :]) * c[:, :-1, :, 1]
        self.psi_Hx[:, :, :-1, 2] += (Ey[:, :, 1:] - Ey[:, :, :-1]) * c[:, :, :-1, 2]

        self.psi_Hy[:, :, :-1, 2] += (Ex[:, :, 1:] - Ex[:, :, :-1]) * c[:, :, :-1, 2]
        self.psi_Hy[:-1, :, :, 0] += (Ez[1:, :, :] - Ez[:-1, :, :]) * c[:-1, :, :, 0]

        self.psi_Hz[:-1, :, :, 0] += (Ey[1:, :, :] - Ey[:-1, :, :]) * c[:-1, :, :, 0]
        self.psi_Hz[:, :-1, :, 1] += (Ex[:, 1:, :] - Ex[:, :-1, :]) * c[:, :-1, :, 1]

        self.phi_H[..., 0] = self.psi_Hx[..., 1] - self.psi_Hx[..., 2]
        self.phi_H[..., 1] = self.psi_Hy[..., 2] - self.psi_Hy[..., 0]
        self.phi_H[..., 2] = self.psi_Hz[..., 0] - self.psi_Hz[..., 1]


class _PMLXlow(PML):
    """ A perfectly matched layer to place where X is low. """

    def _set_locations(self):
        self.loc = (slice(None, self.thickness), slice(None), slice(None), slice(None))
        self.locx = (slice(None, self.thickness), slice(None), slice(None), 0)
        self.locy = (slice(None, self.thickness), slice(None), slice(None), 1)
        self.locz = (slice(None, self.thickness), slice(None), slice(None), 2)

    def _set_shape(self):
        self.shape = (self.thickness, self.grid.Ny, self.grid.Nz)

    def _set_sigmaE(self):
        sigma = self._sigma(bd.arange(self.thickness - 0.5, -0.5, -1.0))
        self.sigmaE = bd.zeros((self.thickness, self.grid.Ny, self.grid.Nz, 3))
        self.sigmaE[:, :, :, 0] = sigma[:, None, None]

    def _set_sigmaH(self):
        sigma = self._sigma(bd.arange(self.thickness - 1.0, 0, -1.0))
        self.sigmaH = bd.zeros((self.thickness, self.grid.Ny, self.grid.Nz, 3))
        self.sigmaH[:-1, :, :, 0] = sigma[:, None, None]


class _PMLXhigh(PML):
    """ A perfectly matched layer to place where X is high. """

    def _set_locations(self):
        self.loc = (slice(-self.thickness, None), slice(None), slice(None), slice(None))
        self.locx = (slice(-self.thickness, None), slice(None), slice(None), 0)
        self.locy = (slice(-self.thickness, None), slice(None), slice(None), 1)
        self.locz = (slice(-self.thickness, None), slice(None), slice(None), 2)

    def _set_shape(self):
        self.shape = (self.thickness, self.grid.Ny, self.grid.Nz)

    def _set_sigmaE(self):
        sigma = self._sigma(bd.arange(0.5, self.thickness + 0.5, 1.0))
        self.sigmaE = bd.zeros((self.thickness, self.grid.Ny, self.grid.Nz, 3))
        self.sigmaE[:, :, :, 0] = sigma[:, None, None]

    def _set_sigmaH(self):
        sigma = self._sigma(bd.arange(1.0, self.thickness, 1.0))
        self.sigmaH = bd.zeros((self.thickness, self.grid.Ny, self.grid.Nz, 3))
        self.sigmaH[:-1, :, :, 0] = sigma[:, None, None]


class _PMLYlow(PML):
    """ A perfectly matched layer to place where Y is low. """

    def _set_locations(self):
        self.loc = (slice(None), slice(None, self.thickness), slice(None))
        self.locx = (slice(None), slice(None, self.thickness), slice(None), 0)
        self.locy = (slice(None), slice(None, self.thickness), slice(None), 1)
        self.locz = (slice(None), slice(None, self.thickness), slice(None), 2)

    def _set_shape(self):
        self.shape = (self.grid.Nx, self.thickness, self.grid.Nz)

    def _set_sigmaE(self):
        sigma = self._sigma(bd.arange(self.thickness - 0.5, -0.5, -1.0))
        self.sigmaE = bd.zeros((self.grid.Nx, self.thickness, self.grid.Nz, 3))
        self.sigmaE[:, :, :, 1] = sigma[None, :, None]

    def _set_sigmaH(self):
        sigma = self._sigma(bd.arange(self.thickness - 1.0, 0, -1.0))
        self.sigmaH = bd.zeros((self.grid.Nx, self.thickness, self.grid.Nz, 3))
        self.sigmaH[:, :-1, :, 1] = sigma[None, :, None]


class _PMLYhigh(PML):
    """ A perfectly matched layer to place where Y is high. """

    def _set_locations(self):
        self.loc = (slice(None), slice(-self.thickness, None), slice(None), slice(None))
        self.locx = (slice(None), slice(-self.thickness, None), slice(None), 0)
        self.locy = (slice(None), slice(-self.thickness, None), slice(None), 1)
        self.locz = (slice(None), slice(-self.thickness, None), slice(None), 2)

    def _set_shape(self):
        self.shape = (self.grid.Nx, self.thickness, self.grid.Nz)

    def _set_sigmaE(self):
        sigma = self._sigma(bd.arange(0.5, self.thickness + 0.5, 1.0))
        self.sigmaE = bd.zeros((self.grid.Nx, self.thickness, self.grid.Nz, 3))
        self.sigmaE[:, :, :, 1] = sigma[None, :, None]

    def _set_sigmaH(self):
        sigma = self._sigma(bd.arange(1.0, self.thickness, 1.0))
        self.sigmaH = bd.zeros((self.grid.Nx, self.thickness, self.grid.Nz, 3))
        self.sigmaH[:, :-1, :, 1] = sigma[None, :, None]


class _PMLZlow(PML):
    """ A perfectly matched layer to place where Z is low. """

    def _set_locations(self):
        self.loc = (slice(None), slice(None), slice(None, self.thickness), slice(None))
        self.locx = (slice(None), slice(None), slice(None, self.thickness), 0)
        self.locy = (slice(None), slice(None), slice(None, self.thickness), 1)
        self.locz = (slice(None), slice(None), slice(None, self.thickness), 2)

    def _set_shape(self):
        self.shape = (self.grid.Nx, self.grid.Ny, self.thickness)

    def _set_sigmaE(self):
        sigma = self._sigma(bd.arange(self.thickness - 0.5, -0.5, -1.0))
        self.sigmaE = bd.zeros((self.grid.Nx, self.grid.Ny, self.thickness, 3))
        self.sigmaE[:, :, :, 2] = sigma[None, None, :]

    def _set_sigmaH(self):
        sigma = self._sigma(bd.arange(self.thickness - 1.0, 0, -1.0))
        self.sigmaH = bd.zeros((self.grid.Nx, self.grid.Ny, self.thickness, 3))
        self.sigmaH[:, :, :-1, 2] = sigma[None, None, :]


class _PMLZhigh(PML):
    """ A perfectly matched layer to place where Z is high. """

    def _set_locations(self):
        self.loc = (slice(None), slice(None), slice(-self.thickness, None), slice(None))
        self.locx = (slice(None), slice(None), slice(-self.thickness, None), 0)
        self.locy = (slice(None), slice(None), slice(-self.thickness, None), 1)
        self.locz = (slice(None), slice(None), slice(-self.thickness, None), 2)

    def _set_shape(self):
        self.shape = (self.grid.Nx, self.grid.Ny, self.thickness)

    def _set_sigmaE(self):
        sigma = self._sigma(bd.arange(0.5, self.thickness + 0.5, 1.0))
        self.sigmaE = bd.zeros((self.grid.Nx, self.grid.Ny, self.thickness, 3))
        self.sigmaE[:, :, :, 2] = sigma[None, None, :]

    def _set_sigmaH(self):
        sigma = self._sigma(bd.arange(1.0, self.thickness, 1.0))
        self.sigmaH = bd.zeros((self.grid.Nx, self.grid.Ny, self.thickness, 3))
        self.sigmaH[:, :, :-1, 2] = sigma[None, None, :]
