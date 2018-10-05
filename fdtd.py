## Imports

# typing
from typing import Tuple
from numbers import Number

# Other
import numpy as np
from tqdm import tqdm

## Constants
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

## Functions
def curl_E(E):
    """ Transforms an E-type field into an H-type field by performing a curl operation """
    ret = np.zeros(E.shape)
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
    ret = np.zeros(H.shape)
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
        # simulation constants
        self.courant_number = float(courant_number)
        self.grid_spacing = float(grid_spacing)
        self.timestep = self.courant_number * self.grid_spacing / SPEED_LIGHT

        # save grid shape as integers
        self.Nx, self.Ny = self._handle_shape(shape)

        # save electric and magnetic field
        self.E = np.zeros((self.Nx, self.Ny, 3))
        self.H = np.zeros((self.Nx, self.Ny, 3))

        # save the inverse of the relative permittiviy and the relative permeability
        # as a diagonal matrix
        self.inverse_permittivity = np.ones((self.Nx, self.Ny, 3)) / permittivity
        self.inverse_permeability = np.ones((self.Nx, self.Ny, 3)) / permeability

        # save current time index
        self.timesteps_passed = 0

    @staticmethod
    def _handle_shape(shape: Tuple[Number, Number]):
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
    def x(self):
        return self.Nx * self.grid_spacing

    @property
    def y(self):
        return self.Ny * self.grid_spacing

    @property
    def shape(self):
        return (self.Nx, self.Ny)

    @property
    def time_passed(self):
        return self.timesteps_passed * self.timestep

    @property
    def permittivity(self):
        return 1.0 / self.permittivity

    @property
    def permeability(self):
        return 1.0 / self.permeability

    def run(self, total_time: Number, progress_bar: bool = True):
        if isinstance(total_time, float):
            total_time /= self.timestep
        time = range(0, int(total_time), 1)
        if progress_bar:
            time = tqdm(time)
        for _ in time:
            self.step()

    def step(self):
        self.update_E()
        self.update_H()
        self.timesteps_passed += 1

    def update_E(self):
        # normal update equation
        self.E += self.courant_number * self.inverse_permittivity * curl_H(self.H)
        # add source (dummy for now)
        self.E[self.Nx // 2, self.Ny // 2, 2] = 1

    def update_H(self):
        # normal update equation
        self.H -= self.courant_number * self.inverse_permeability * curl_E(self.E)
        # add source (None for now)
        # self.H[...]

    def reset(self):
        self.H *= 0.0
        self.E *= 0.0
        self.timesteps_passed *= 0


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    grid = Grid((400, 400))

    start = time.time()
    grid.run(20, progress_bar=False)
    print(f"time elapsed: {time.time()-start}")

    plt.imshow(grid.E[..., 2])
    plt.show()
