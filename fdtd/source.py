## Imports

# other
from math import pi, sin

# typing
from typing import Tuple
from numbers import Number

# relatvie
from .backend import Tensorlike
from .backend import backend as bd


## Source class
class Source:
    def __init__(
        self,
        p0: Tuple[Number, Number, Number],
        p1: Tuple[Number, Number, Number],
        period: Number = 1,
        power: float = 1.0,
        phase_shift: float = 0.0,
    ):
        """ Create a Source with a gaussian profile spanning from `p0` to `p1`.

        Args:
            p0: First point of the source. The coordinates can be specified as integers
                [gridpoints] or as floats [meters].
            p1: Last point of the source. The coordinates can be specified as integers
                [gridpoints] or as floats [meters].
            period = 1: The period of the source. The period can be specified as integer
                [timesteps] or as float [seconds]
            power = 1.0: The power of the source
            phase_shift = 0.0: The phase offset of the source.

        Note:
            The initialization of the source is not finished before it is registered on the
            grid. This is don automatically by setting the source to be an attribute of the
            grid.

        """
        self.grid = None
        self.period = period
        self.power = power
        self.phase_shift = phase_shift
        self.p0 = p0
        self.p1 = p1

    def register_grid(self, grid: "Grid"):
        """ Register a grid for the source.

        Args:
            grid: The grid to register the source in.

        """
        self.grid = grid
        if isinstance(self.period, float):
            self.period = int(self.period / self.grid.timestep + 0.5)
        self.p0 = self.grid._handle_tuple(self.p0)
        self.p1 = self.grid._handle_tuple(self.p1)
        self.x, self.y, self.z = self._get_location(self.p0, self.p1)
        L = self.x.shape[0]

        amplitude = (
            self.power * self.grid.inverse_permittivity[self.x, self.y, self.z, 2]
        ) ** 0.5
        self.vect = bd.array(
            (self.x - self.x[L // 2]) ** 2
            + (self.y - self.y[L // 2]) ** 2
            + (self.z - self.z[L // 2]) ** 2,
            bd.float,
        )
        self.profile = bd.exp(-self.vect ** 2 / (2 * (0.5 * self.vect.max()) ** 2))
        self.profile /= self.profile.sum()
        self.profile *= amplitude

    @staticmethod
    def _get_location(
        p0: Tuple[int, int, int], p1: Tuple[int, int, int]
    ) -> Tuple[Tensorlike, Tensorlike, Tensorlike]:
        """ get the line of points spanning between `p0` and `p1`.

        Args:
            p0: The first point of the source specified as a tuple of gridpoint coordinates
            p1: The second point of the source specified as a tuple of gridpoint coordinates

        Returns:
            x, y, z: the x, y and z coordinates of the source

        """
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        m = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        x = bd.array(bd.linspace(x0, x1, m, endpoint=False), bd.int)
        y = bd.array(bd.linspace(y0, y1, m, endpoint=False), bd.int)
        z = bd.array(bd.linspace(z0, z1, m, endpoint=False), bd.int)
        return x, y, z

    def source_E(self):
        """ Add the source to the electric field """
        q = self.grid.timesteps_passed
        vect = self.profile * sin(2 * pi * q / self.period + self.phase_shift)
        self.grid.E[self.x, self.y, self.z, 2] += vect

    def source_H(self):
        """ Add the source to the magnetic field """
        pass

    def __repr__(self):
        return f"Source(p0={self.p0}, p1={self.p1}, period={self.period}, power={self.power})"
