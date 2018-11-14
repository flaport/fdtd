## Imports

# typing
from numbers import Number
from .backend import Tensorlike

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
        self.grid.E[0, :, :, :] = self.grid.E[-1, :, :, :]

    def update_H(self):
        """ Update magnetic field such that periodic boundary conditions in the
        X-directions apply """
        self.grid.H[-1, :, :, :] = self.grid.H[0, :, :, :]


# Periodic Boundaries in the Y-direction
class PeriodicBoundaryY(Boundary):
    def update_E(self):
        """ Update electric field such that periodic boundary conditions in the
        Y-direction apply """
        self.grid.E[:, 0, :, :] = self.grid.E[:, -1, :, :]

    def update_H(self):
        """ Update magnetic field such that periodic boundary conditions in the
        Y-direction apply """
        self.grid.H[:, -1, :, :] = self.grid.H[:, 0, :, :]


# Periodic Boundaries in the Z-direction
class PeriodicBoundaryZ(Boundary):
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
    """ A perfectly matched layer is an impedence-matched area at the boundary of the
    grid for which all fields incident perpendicular to the area are absorbed without
    reflection.
    """

    def __init__(self, thickness: Number = 10, a: float = 1e-8):
        """ Perfectly Matched Layer

        Args:
            thickness: The thickness of the PML. The thickness can be specified as
                integer [gridpoints] or as float [meters].
            a = 1e-8: stability parameter
        """
        self.grid = None  # will be set later

        # TODO: to make this a PML parameter, the *normal* curl equations need to be updated
        self.k = 1.0

        self.a = a
        self.thickness = thickness

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

    def register_grid(self, grid: Grid):
        """ Register a grid for the PML

        Args:
            grid: The grid to register the PML in
        """
        self.grid = grid
        self.thickness = self.grid._handle_distance(self.thickness)

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


class PMLXlow(PML):
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


class PMLXhigh(PML):
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


class PMLYlow(PML):
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


class PMLYhigh(PML):
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


class PMLZlow(PML):
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


class PMLZhigh(PML):
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
