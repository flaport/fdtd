## Imports

# typing
from typing import Tuple
from numbers import Number

# Other
from backend import set_backend
from backend import backend as bd

from tqdm import tqdm

## Constants
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

## Functions
def curl_E(E):
    """ Transforms an E-type field into an H-type field by performing a curl operation """
    ret = bd.zeros(E.shape)
    ret[:, :-1, 0] = E[:, 1:, 2] - E[:, :-1, 2]
    ret[:, -1, 0] = -E[:, -1, 2]
    ret[:-1, :, 1] = -E[1:, :, 2] + E[:-1, :, 2]
    ret[-1, :, 1] = E[-1, :, 2]
    ret[:-1, :, 2] = E[1:, :, 1] - E[:-1, :, 1]
    ret[-1, :, 2] = -E[-1, :, 1]
    ret[:, :-1, 2] -= E[:, 1:, 0] - E[:, :-1, 0]
    ret[:, -1, 2] -= -E[:, -1, 0]
    return ret


def curl_H(H):
    """ Transforms an H-type field into an E-type field by performing a curl operation """
    ret = bd.zeros(H.shape)
    ret[:, 1:, 0] = H[:, 1:, 2] - H[:, :-1, 2]
    ret[:, 0, 0] = H[:, 0, 2]
    ret[1:, :, 1] = -(H[1:, :, 2] - H[:-1, :, 2])
    ret[0, :, 1] = -H[0, :, 2]
    ret[1:, :, 2] = H[1:, :, 1] - H[:-1, :, 1]
    ret[0, :, 2] = H[0, :, 1]
    ret[:, 1:, 2] -= H[:, 1:, 0] - H[:, :-1, 0]
    ret[:, 0, 2] -= H[:, 0, 0]
    return ret


## FDTD Grid Class
class Grid:
    def __init__(
        self,
        shape: Tuple[Number, Number],
        grid_spacing: float = 50e-9,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        courant_number: float = 0.7,
    ):
    '''
    Args:
        shape: shape of the FDTD grid.
        grid_spacing = 50e-9: distance between the grid cells.
        permittivity = 1.0: the relative permittivity of the background.
        permeability = 1.0: the relative permeability of the background.
        courant_number = 0.7: the courant number of the FDTD simulation. The timestep of
            the simulation will be derived from this number using the CFL-condition.
    '''
        # simulation constants
        self.courant_number = float(courant_number)
        self.grid_spacing = float(grid_spacing)
        self.timestep = self.courant_number * self.grid_spacing / SPEED_LIGHT

        # save grid shape as integers
        self.Nx, self.Ny = self._handle_shape(shape)

        # save electric and magnetic field
        self.E = bd.zeros((self.Nx, self.Ny, 3))
        self.H = bd.zeros((self.Nx, self.Ny, 3))

        # save the inverse of the relative permittiviy and the relative permeability
        # as a diagonal matrix
        self.inverse_permittivity = bd.ones((self.Nx, self.Ny, 3)) / permittivity
        self.inverse_permeability = bd.ones((self.Nx, self.Ny, 3)) / permeability

        # save current time index
        self.timesteps_passed = 0

    @staticmethod
    def _handle_shape(shape: Tuple[Number, Number]) -> Tuple[int, int]:
        ''' validate the grid shape and transform to a length-2 tuple of ints '''
        if len(shape) != 2:
            raise ValueError(
                f"invalid grid shape {shape}\n"
                f"grid shape should be a 2D tuple containing floats or ints"
            )
        x, y = shape
        if isinstance(x, float):
            x = int(x / self.grid_spacing + 0.5)
        if isinstance(y, float):
            x = int(x / self.grid_spacing + 0.5)
        return x, y

    @property
    def x(self) -> int:
        ''' get the number of grid cells in the x-direction '''
        return self.Nx * self.grid_spacing

    @property
    def y(self) -> int:
        ''' get the number of grid cells in the y-direction '''
        return self.Ny * self.grid_spacing

    @property
    def shape(self) -> Tuple[int,int]:
        ''' get the shape of the FDTD grid '''
        return (self.Nx, self.Ny)

    @property
    def time_passed(self) -> float:
        ''' get the total time passed '''
        return self.timesteps_passed * self.timestep

    def run(self, total_time: Number, progress_bar: bool = True):
        ''' run an FDTD simulation.

        Args:
            total_time: the total time for the simulation to run.
            progress_bar = True: choose to show a progress bar during simulation

        '''
        if isinstance(total_time, float):
            total_time /= self.timestep
        time = range(0, int(total_time), 1)
        if progress_bar:
            time = tqdm(time)
        for _ in time:
            self.step()

    def step(self):
        ''' do a single FDTD step by first updating the electric field and then
        updating the magnetic field
        '''
        self.update_E()
        self.update_H()
        self.timesteps_passed += 1

    def update_E(self):
        ''' update the electric field by using the curl of the magnetic field '''
        curl = curl_H(self.H)
        self.E += self.courant_number * self.inverse_permittivity * curl

        # add source (dummy for now)
        self.E[self.Nx // 2, self.Ny // 2, 2] = 1

    def update_H(self):
        ''' update the magnetic field by using the curl of the electric field '''
        curl = curl_E(self.E)
        self.H -= self.courant_number * self.inverse_permeability * curl

    def reset(self):
        ''' reset the grid by setting all fields to zero '''
        self.H *= 0.0
        self.E *= 0.0
        self.timesteps_passed *= 0
