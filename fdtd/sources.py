""" Sources are objects that inject power into the grid.

Available sources:

- PointSource
- LineSource

"""
## Imports

# other
from math import pi, sin, cos

# typing
from .typing import Tuple, Number, ListOrSlice, List

# relatvie
from .grid import Grid
from .backend import backend as bd


# For Hanning window pulses
def hanning(f, t, n):
    return (1 / 2) * (1 - cos(f * t / n)) * (sin(f * t))


## PointSource class
class PointSource:
    """ A source placed at a single point (grid cell) in the grid """

    def __init__(
        self,
        period: Number = 15,
        power: float = 1.0,
        phase_shift: float = 0.0,
        name: str = None,
        pulse: bool = False,
        cycle: int = 5,
        hanning_dt: float = 10.0,
    ):
        """Create a LineSource with a gaussian profile

        Args:
            period: The period of the source. The period can be specified
                as integer [timesteps] or as float [seconds]
            power: The power of the source
            phase_shift: The phase offset of the source.
            name: name of the source.
            pulse: Set True to use a Hanning window pulse instead of continuous wavefunction.
            cycle: cycles for Hanning window pulse.
            hanning_dt: timestep used for Hanning window pulse width (optional).

        """
        self.grid = None
        self.period = period
        self.power = power
        self.phase_shift = phase_shift
        self.name = name
        self.pulse = pulse
        self.cycle = cycle
        self.frequency = 1.0 / period
        self.hanning_dt = hanning_dt if hanning_dt is not None else 0.5 / self.frequency

    def _register_grid(self, grid: Grid, x: Number, y: Number, z: Number):
        """Register a grid for the source.

        Args:
            grid: the grid to place the source into.
            x: The x-location of the source in the grid
            y: The y-location of the source in the grid
            z: The z-location of the source in the grid

        Note:
            As its name suggests, this source is a POINT source.
            Hence it should be placed at a single coordinate tuple
            int the grid.
        """
        self.grid = grid
        self.grid.sources.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        try:
            (x,), (y,), (z,) = x, y, z
        except (TypeError, ValueError):
            raise ValueError("a point source should be placed on a single grid cell.")
        self.x, self.y, self.z = grid._handle_tuple((x, y, z))
        self.period = grid._handle_time(self.period)
        self.amplitude = (
            self.power * self.grid.inverse_permittivity[self.x, self.y, self.z, 2]
        ) ** 0.5

    def update_E(self):
        """ Add the source to the electric field """
        q = self.grid.time_steps_passed
        # if pulse
        if self.pulse:
            t1 = int(2 * pi / (self.frequency * self.hanning_dt / self.cycle))
            if q < t1:
                src = self.amplitude * hanning(
                    self.frequency, q * self.hanning_dt, self.cycle
                )
            else:
                # src = - self.grid.E[self.x, self.y, self.z, 2]
                src = 0
        # if not pulse
        else:
            src = self.amplitude * sin(2 * pi * q / self.period + self.phase_shift)
        self.grid.E[self.x, self.y, self.z, 2] += src

    def update_H(self):
        """ Add the source to the magnetic field """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(period={self.period}, "
            f"power={self.power}, phase_shift={self.phase_shift}, "
            f"name={repr(self.name)})"
        )

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"{self.x}"
        y = f"{self.y}"
        z = f"{self.z}"
        s += f"        @ x={x}, y={y}, z={z}\n"
        return s


## LineSource class
class LineSource:
    """ A source along a line in the FDTD grid """

    def __init__(
        self,
        period: Number = 15,
        power: float = 1.0,
        phase_shift: float = 0.0,
        name: str = None,
        pulse: bool = False,
        cycle: int = 5,
        hanning_dt: float = 10.0,
    ):
        """Create a LineSource with a gaussian profile

        Args:
            period: The period of the source. The period can be specified
                as integer [timesteps] or as float [seconds]
            power: The power of the source
            phase_shift: The phase offset of the source.
            pulse: Set True to use a Hanning window pulse instead of continuous wavefunction.
            cycle: cycles for Hanning window pulse.
            hanning_dt: timestep used for Hanning window pulse width (optional).

        """
        self.grid = None
        self.period = period
        self.power = power
        self.phase_shift = phase_shift
        self.name = name
        self.pulse = pulse
        self.cycle = cycle
        self.frequency = 1.0 / period
        self.hanning_dt = hanning_dt if hanning_dt is not None else 0.5 / self.frequency

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        """Register a grid for the source.

        Args:
            grid: the grid to place the source into.
            x: The x-location of the source in the grid
            y: The y-location of the source in the grid
            z: The z-location of the source in the grid

        Note:
            As its name suggests, this source is a LINE source.
            Hence the source spans the diagonal of the cube
            defined by the slices in the grid.
        """
        self.grid = grid
        self.grid.sources.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        self.x, self.y, self.z = self._handle_slices(x, y, z)

        self.period = grid._handle_time(self.period)
        amplitude = (
            self.power * self.grid.inverse_permittivity[self.x, self.y, self.z, 2]
        ) ** 0.5

        L = len(self.x)
        vect = bd.array(
            (bd.array(self.x) - self.x[L // 2]) ** 2
            + (bd.array(self.y) - self.y[L // 2]) ** 2
            + (bd.array(self.z) - self.z[L // 2]) ** 2,
            bd.float,
        )

        self.profile = bd.exp(-(vect ** 2) / (2 * (0.5 * vect.max()) ** 2))
        self.profile /= self.profile.sum()
        self.profile *= amplitude

    def _handle_slices(
        self, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ) -> Tuple[List, List, List]:
        """Convert slices in the grid to lists

        This is necessary to make the source span the diagonal of the volume
        defined by the slices.

        Args:
            x: The x-location of the volume in the grid
            y: The y-location of the volume in the grid
            z: The z-location of the volume in the grid

        Returns:
            x, y, z: the x, y and z coordinates of the source as lists

        """

        # if list-indices were chosen:
        if isinstance(x, list) and isinstance(y, list) and isinstance(z, list):
            if len(x) != len(y) or len(y) != len(z) or len(z) != len(x):
                raise IndexError(
                    "sources require grid to be indexed with slices or equal length list-indices"
                )
            x = [self.grid._handle_distance(_x) for _x in x]
            y = [self.grid._handle_distance(_y) for _y in y]
            z = [self.grid._handle_distance(_z) for _z in z]
            return x, y, z

        # if a combination of list-indices and slices were chosen,
        # convert the list-indices to slices.
        # TODO: maybe issue a warning here?
        if isinstance(x, list):
            x = slice(
                self.grid._handle_distance(x[0]),
                self.grid._handle_distance(x[-1]),
                None,
            )
        if isinstance(y, list):
            y = slice(
                self.grid._handle_distance(y[0]),
                self.grid._handle_distance(y[-1]),
                None,
            )
        if isinstance(z, list):
            z = slice(
                self.grid._handle_distance(z[0]),
                self.grid._handle_distance(z[-1]),
                None,
            )

        # if we get here, we can assume slices:
        x0 = self.grid._handle_distance(x.start if x.start is not None else 0)
        y0 = self.grid._handle_distance(y.start if y.start is not None else 0)
        z0 = self.grid._handle_distance(z.start if z.start is not None else 0)
        x1 = self.grid._handle_distance(x.stop if x.stop is not None else self.grid.Nx)
        y1 = self.grid._handle_distance(y.stop if y.stop is not None else self.grid.Ny)
        z1 = self.grid._handle_distance(z.stop if z.stop is not None else self.grid.Nz)

        # we can now convert these coordinates into index lists
        m = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        if m < 2:
            raise ValueError("a LineSource should consist of at least two gridpoints")
        x = [v.item() for v in bd.array(bd.linspace(x0, x1, m, endpoint=False), bd.int)]
        y = [v.item() for v in bd.array(bd.linspace(y0, y1, m, endpoint=False), bd.int)]
        z = [v.item() for v in bd.array(bd.linspace(z0, z1, m, endpoint=False), bd.int)]

        return x, y, z

    def update_E(self):
        """ Add the source to the electric field """
        q = self.grid.time_steps_passed
        # if pulse
        if self.pulse:
            t1 = int(2 * pi / (self.frequency * self.hanning_dt / self.cycle))
            if q < t1:
                vect = self.profile * hanning(
                    self.frequency, q * self.hanning_dt, self.cycle
                )
            else:
                # src = - self.grid.E[self.x, self.y, self.z, 2]
                vect = self.profile * 0
        # if not pulse
        else:
            vect = self.profile * sin(2 * pi * q / self.period + self.phase_shift)
        # do not use list indexing here, as this is much slower especially for torch backend
        # DISABLED: self.grid.E[self.x, self.y, self.z, 2] = vect
        for x, y, z, value in zip(self.x, self.y, self.z, vect):
            self.grid.E[x, y, z, 2] += value

    def update_H(self):
        """ Add the source to the magnetic field """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(period={self.period}, "
            f"power={self.power}, phase_shift={self.phase_shift}, "
            f"name={repr(self.name)})"
        )

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"[{self.x[0]}, ... , {self.x[-1]}]"
        y = f"[{self.y[0]}, ... , {self.y[-1]}]"
        z = f"[{self.z[0]}, ... , {self.z[-1]}]"
        s += f"        @ x={x}, y={y}, z={z}\n"
        return s


## PlaneSource class
class PlaneSource:
    """ A source along a plane in the FDTD grid """

    def __init__(
        self,
        period: Number = 15,
        power: float = 1.0,
        phase_shift: float = 0.0,
        name: str = None,
    ):
        """Create a LineSource with a gaussian profile

        Args:
            period: The period of the source. The period can be specified
                as integer [timesteps] or as float [seconds]
            power: The power of the source
            phase_shift: The phase offset of the source.

        """
        self.grid = None
        self.period = period
        self.power = power
        self.phase_shift = phase_shift
        self.name = name

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        """Register a grid for the source.

        Args:
            grid: the grid to place the source into.
            x: The x-location of the source in the grid
            y: The y-location of the source in the grid
            z: The z-location of the source in the grid

        Note:
            As its name suggests, this source is a LINE source.
            Hence the source spans the diagonal of the cube
            defined by the slices in the grid.
        """
        self.grid = grid
        self.grid.sources.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        self.x, self.y, self.z = self._handle_slices(x, y, z)

        self.period = grid._handle_time(self.period)
        amplitude = (
            self.power * self.grid.inverse_permittivity[self.x, self.y, self.z, 2]
        ) ** 0.5

        x = bd.arange(self.x.start, self.x.stop, 1) - (self.x.start + self.x.stop) // 2
        y = bd.arange(self.y.start, self.y.stop, 1) - (self.y.start + self.y.stop) // 2
        z = bd.arange(self.z.start, self.z.stop, 1) - (self.z.start + self.z.stop) // 2
        xvec, yvec, zvec = bd.broadcast_arrays(
            x[:, None, None], y[None, :, None], z[None, None, :]
        )
        _xvec = bd.array(xvec, float)
        _yvec = bd.array(yvec, float)
        _zvec = bd.array(zvec, float)

        print(
            [
                self.x.stop - self.x.start,
                self.y.stop - self.y.start,
                self.z.stop - self.z.start,
            ]
        )

        if self.x.stop - self.x.start == 1:
            profile = bd.exp(
                -(_yvec ** 2 + _zvec ** 2)
                / (2 * (0.5 * min(_yvec.max(), _zvec.max())) ** 2)
            )
        elif self.y.stop - self.y.start == 1:
            profile = bd.exp(
                -(_zvec ** 2 + _xvec ** 2)
                / (2 * (0.5 * min(_zvec.max(), _xvec.max())) ** 2)
            )
        elif self.z.stop - self.z.start == 1:
            profile = bd.exp(
                -(_xvec ** 2 + _yvec ** 2)
                / (2 * (0.5 * min(_xvec.max(), _yvec.max())) ** 2)
            )

        self.profile = amplitude * profile / profile.sum()

    def _handle_slices(
        self, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ) -> Tuple[List, List, List]:
        """Validate slices and calculate center of plane

        Args:
            x: The x-location of the volume in the grid
            y: The y-location of the volume in the grid
            z: The z-location of the volume in the grid

        Returns:
            x, y, z: the x, y and z coordinates of the source as slices

        """
        # ensure all slices
        if not isinstance(x, slice):
            x = slice(
                self.grid._handle_distance(x), self.grid._handle_distance(x) + 1, None
            )
        if not isinstance(y, slice):
            y = slice(
                self.grid._handle_distance(y), self.grid._handle_distance(y) + 1, None
            )
        if not isinstance(z, slice):
            z = slice(
                self.grid._handle_distance(z), self.grid._handle_distance(z) + 1, None
            )

        # if we get here, we can assume slices:
        x0 = self.grid._handle_distance(x.start if x.start is not None else 0)
        y0 = self.grid._handle_distance(y.start if y.start is not None else 0)
        z0 = self.grid._handle_distance(z.start if z.start is not None else 0)
        x1 = self.grid._handle_distance(x.stop if x.stop is not None else self.grid.Nx)
        y1 = self.grid._handle_distance(y.stop if y.stop is not None else self.grid.Ny)
        z1 = self.grid._handle_distance(z.stop if z.stop is not None else self.grid.Nz)

        # make sure all slices have a start, stop and no step:
        x = (
            slice(x0, x1)
            if x0 < x1
            else (slice(x1, x0) if x0 > x1 else slice(x0, x0 + 1))
        )
        y = (
            slice(y0, y1)
            if y0 < y1
            else (slice(y1, y0) if y0 > y1 else slice(y0, y0 + 1))
        )
        z = (
            slice(z0, z1)
            if z0 < z1
            else (slice(z1, z0) if z0 > z1 else slice(z0, z0 + 1))
        )

        if [x.stop - x.start, y.stop - y.start, z.stop - z.start].count(0) > 0:
            raise ValueError(
                "Given location for PlaneSource results in slices of length 0!"
            )
        if [x.stop - x.start, y.stop - y.start, z.stop - z.start].count(1) == 0:
            raise ValueError("Given location for PlaneSource is not a 2D plane!")
        if [x.stop - x.start, y.stop - y.start, z.stop - z.start].count(1) > 1:
            raise ValueError(
                "Given location for PlaneSource should have no more than one dimension in which it's flat.\n"
                "Use a LineSource for lower dimensional sources."
            )

        return x, y, z

    def update_E(self):
        """ Add the source to the electric field """
        q = self.grid.time_steps_passed
        vect = self.profile * sin(2 * pi * q / self.period + self.phase_shift)
        self.grid.E[self.x, self.y, self.z, 2] = vect

    def update_H(self):
        """ Add the source to the magnetic field """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(period={self.period}, "
            f"power={self.power}, phase_shift={self.phase_shift}, "
            f"name={repr(self.name)})"
        )

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"[{self.x.start}, ... , {self.x.stop}]"
        y = f"[{self.y.start}, ... , {self.y.stop}]"
        z = f"[{self.z.start}, ... , {self.z.stop}]"
        s += f"        @ x={x}, y={y}, z={z}\n"
        return s
