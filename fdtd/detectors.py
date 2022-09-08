""" Detectors for the FDTD Grid.

Available Detectors:

 - LineDetector

"""

## Imports

# typing
from .typing_ import ListOrSlice, Tuple, List

# relative
from .grid import Grid
from .backend import backend as bd
from .constants import X, Y, Z

## Detector
class LineDetector:
    """ A detector along a line in the FDTD grid """

    def __init__(self, name=None):
        """Create a line detector

        Args:
            name: name of the Detector

        """
        self.grid = None
        self.E = []
        self.H = []
        self.name = name

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        """Register a grid to the detector

        Args:
            grid: the grid to place the detector into
            x: the x-location of the detector in the grid
            y: the y-location of the detector in the grid
            z: the z-location of the detector in the grid

        Note:
            As its name suggests, this detector is a LINE detector.
            Hence the detector spans the diagonal of the cube
            defined by the slices in the grid.
        """
        self.grid = grid
        self.grid.detectors.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        self.x, self.y, self.z = self._handle_slices(x, y, z)

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
            return x, y, z

        # if a combination of list-indices and slices were chosen,
        # convert the list-indices to slices.
        # TODO: maybe issue a warning here?
        if isinstance(x, list):
            x = slice(x[0], x[-1], None)
        if isinstance(y, list):
            y = slice(y[0], y[-1], None)
        if isinstance(z, list):
            z = slice(z[0], z[-1], None)

        # if we get here, we can assume slices:
        x0 = x.start if x.start is not None else 0
        y0 = y.start if y.start is not None else 0
        z0 = z.start if z.start is not None else 0
        x1 = x.stop if x.stop is not None else self.grid.Nx
        y1 = y.stop if y.stop is not None else self.grid.Ny
        z1 = z.stop if z.stop is not None else self.grid.Nz

        # we can now convert these coordinates into index lists
        m = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))
        x = [v.item() for v in bd.array(bd.linspace(x0, x1, m, endpoint=False), bd.int)]
        y = [v.item() for v in bd.array(bd.linspace(y0, y1, m, endpoint=False), bd.int)]
        z = [v.item() for v in bd.array(bd.linspace(z0, z1, m, endpoint=False), bd.int)]
        return x, y, z

    def detect_E(self):
        """ detect the electric field at a certain location in the grid """
        # TODO: there is a performance bottleneck here (indexing with lists)
        E = self.grid.E[self.x, self.y, self.z]
        self.E.append(E)

    def detect_H(self):
        """ detect the magnetic field at a certain location in the grid """
        # TODO: there is a performance bottleneck here (indexing with lists)
        H = self.grid.H[self.x, self.y, self.z]
        self.H.append(H)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"[{self.x[0]}, ... , {self.x[-1]}]"
        y = f"[{self.y[0]}, ... , {self.y[-1]}]"
        z = f"[{self.z[0]}, ... , {self.z[-1]}]"
        s += f"        @ x={x}, y={y}, z={z}\n"
        return s

    def detector_values(self):
        """ outputs what detector detects """
        return {"E": self.E, "H": self.H}


# is the "detector" paradigm necessary? Can we just flag a segment of the base mesh to be
# stored per timestep?

## BlockDetector
class BlockDetector:
    """ A detector along a block in the FDTD grid """

    """ Basic copy of LineDetector code, changed detect functions """

    def __init__(self, name=None):
        """Create a block detector

        Args:
            name: name of the Detector

        """
        self.grid = None
        self.E = []
        self.H = []
        self.name = name

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        """Register a grid to the detector

        Args:
            grid: the grid to place the detector into
            x: the x-location of the detector in the grid
            y: the y-location of the detector in the grid
            z: the z-location of the detector in the grid

        Note:
            As its name suggests, this detector is a BLOCK detector.
            Hence the detector spans the VOLUME of the cube
            defined by the slices in the grid.
        """
        self.grid = grid
        self.grid.detectors.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        self.x, self.y, self.z = self._handle_slices(x, y, z)

    def _handle_slices(
        self, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ) -> Tuple[List, List, List]:
        """Convert slices in the grid to lists

        This is necessary to make the source span the volume
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
            return x, y, z

        # if a combination of list-indices and slices were chosen,
        # convert the list-indices to slices.
        # TODO: maybe issue a warning here?
        if isinstance(x, list):
            x = slice(x[0], x[-1], None)
        if isinstance(y, list):
            y = slice(y[0], y[-1], None)
        if isinstance(z, list):
            z = slice(z[0], z[-1], None)

        # if we get here, we can assume slices:
        x0 = x.start if x.start is not None else 0
        y0 = y.start if y.start is not None else 0
        z0 = z.start if z.start is not None else 0
        x1 = x.stop if x.stop is not None else self.grid.Nx
        y1 = y.stop if y.stop is not None else self.grid.Ny
        z1 = z.stop if z.stop is not None else self.grid.Nz

        # we can now convert these coordinates into index lists
        x = [v.item() for v in bd.arange(x0, x1 + 1)]
        y = [v.item() for v in bd.arange(y0, y1 + 1)]
        z = [v.item() for v in bd.arange(z0, z1 + 1)]
        return x, y, z

    def detect_E(self):
        """ detect the electric field at a certain location in the grid """
        # TODO: there is a performance bottleneck here (indexing with lists)
        E = []
        for i, row in enumerate(self.x):
            E.append([])
            for j, col in enumerate(self.y):
                E[i].append([])
                for pillar in self.z:
                    E[i][j].append(self.grid.E[row, col, [pillar]][0])
        # E = self.grid.E[self.x, self.y, self.z]
        self.E.append(E)

    def detect_H(self):
        """ detect the magnetic field at a certain location in the grid """
        # TODO: there is a performance bottleneck here (indexing with lists)
        H = []
        for i, row in enumerate(self.x):
            H.append([])
            for j, col in enumerate(self.y):
                H[i].append([])
                for pillar in self.z:
                    H[i][j].append(self.grid.H[row, col, [pillar]][0])
        # H = self.grid.H[self.x, self.y, self.z]
        self.H.append(H)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"[{self.x[0]}, ... , {self.x[-1]}]"
        y = f"[{self.y[0]}, ... , {self.y[-1]}]"
        z = f"[{self.z[0]}, ... , {self.z[-1]}]"
        s += f"        @ x={x}, y={y}, z={z}\n"
        return s

    def detector_values(self):
        """ outputs what detector detects """
        return {"E": self.E, "H": self.H}


## CurrentDetector
class CurrentDetector:
    """ A current detector. """

    """



    """

    # Alternative APIs:
    #
    # 1. Current detection could be a standard detector output,
    # if every h-field probe pointed two ways.
    # 2. No current detector class at all. SoftArbitraryPointSource
    # makes two H-detectors. (can't test separated then).
    # Don't know if this op over a volume is valid. Should be.
    # TODO: Detector geometry as an argument?
    # Lots of interesting I detector geometries that might be worth implementing,
    # ("integrated current loop"), this could be a flag here.
    #
    # Existing sources are all z-polarized; E-detectors output all the polarizations.
    # Unlike other Detectors, this does not yet output a polarized current!
    #
    # *Source* polarization is probably an important feature for 3D electrical work
    # not sure how important it is for optical stuff

    def __init__(self, name=None):
        """Create a block detector

        Args:
            name: name of the Detector

        """
        self.grid = None
        self.orientation = None
        self.I = []
        self.name = name

    def _register_grid(
        self, grid: Grid, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ):
        """Register a grid to the detector

        Args:
            grid: the grid to place the detector into
            x: the x-location of the detector in the grid
            y: the y-location of the detector in the grid
            z: the z-location of the detector in the grid
            orientation: a unit vector (a length 1) describing the orientation?

        Note:
            As its name suggests, this detector is a BLOCK detector.
            Hence the detector spans the VOLUME of the cube
            defined by the slices in the grid.
        """
        self.grid = grid
        self.grid.detectors.append(self)
        if self.name is not None:
            if not hasattr(grid, self.name):
                setattr(grid, self.name, self)
            else:
                raise ValueError(
                    f"The grid already has an attribute with name {self.name}"
                )

        self.x, self.y, self.z = self._handle_slices(x, y, z)

    def _handle_slices(
        self, x: ListOrSlice, y: ListOrSlice, z: ListOrSlice
    ) -> Tuple[List, List, List]:
        """Convert slices in the grid to lists

        This is necessary to make the source span the volume
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
            return x, y, z

        # if a combination of list-indices and slices were chosen,
        # convert the list-indices to slices.
        # TODO: maybe issue a warning here?
        if isinstance(x, list):
            x = slice(x[0], x[-1], None)
        if isinstance(y, list):
            y = slice(y[0], y[-1], None)
        if isinstance(z, list):
            z = slice(z[0], z[-1], None)

        # if we get here, we can assume slices:
        x0 = x.start if x.start is not None else 0
        y0 = y.start if y.start is not None else 0
        z0 = z.start if z.start is not None else 0
        x1 = x.stop if x.stop is not None else self.grid.Nx
        y1 = y.stop if y.stop is not None else self.grid.Ny
        z1 = z.stop if z.stop is not None else self.grid.Nz

        # we can now convert these coordinates into index lists
        x = [v.item() for v in bd.arange(x0, x1 + 1)]
        y = [v.item() for v in bd.arange(y0, y1 + 1)]
        z = [v.item() for v in bd.arange(z0, z1 + 1)]
        return x, y, z

    def detect_E(self):
        """ detect the electric field at a certain location in the grid """

    def single_point_current(self, px, py, pz):
        """

        Only Z-polarized for now. Can probably do a cross product to get arbitrary polarizations

        ^
        |
        |
        X---->

        TODO: FIXME: IMPORTANT:
        material magnetic permeability? find test cases!



        Implements the first correction from [Fang 1994] (two
        cells are spatially averaged to account for Yee cell half-step inaccuracies),
        but not the second one (minor loss of accuracy).

        Jiayuan Fang, Danwei Xue.
        Precautions in the calculation of impedance in FDTD computations.
        Proceedings of IEEE Antennas and Propagation Society International Symposium
        and URSI National Radio Science Meeting, vol. 3, 1994, p. 1814–7 vol.3.
        https://doi.org/10.1109/APS.1994.408185.


        Luebbers RJ, Langdon HS.
        A simple feed model that reduces time steps needed for
        FDTD antenna and microstrip calculations.
        IEEE Trans Antennas Propagat 1996;44:1000–5.
        https://doi.org/10.1109/8.504308.

        """

        # [Luebbers 1996 1992]
        # /sqrt(mu_0)s removed!

        # option for unit factor here? Units will get very complicated otherwise

        current_vector_1 = (
            (self.grid.H[px, py - 1, pz, X]) - (self.grid.H[px, py, pz, X])
        ) * self.grid.grid_spacing
        current_vector_2 = (
            (self.grid.H[px, py, pz, Y]) - (self.grid.H[px - 1, py, pz, Y])
        ) * self.grid.grid_spacing

        current_1 = current_vector_1 + current_vector_2
        # current_1 = float(current.cpu())

        current_vector_1 = (
            (self.grid.H[px, py - 1, pz - 1, X]) - (self.grid.H[px, py, pz - 1, X])
        ) * self.grid.grid_spacing
        current_vector_2 += (
            (self.grid.H[px, py, pz - 1, Y]) - (self.grid.H[px - 1, py, pz - 1, Y])
        ) * self.grid.grid_spacing
        # current_2 = float(current_2.cpu())
        current_2 = current_vector_1 + current_vector_2

        I = (current_1 + current_2) / 2.0

        return I

    def detect_H(self):

        # should detector outputs be bd.array() by default?
        # for that matter, should they always be converted to numpy?

        I = []
        for i, row in enumerate(self.x):
            I.append([])
            for j, col in enumerate(self.y):
                I[i].append([])
                for k, pillar in enumerate(self.z):
                    I[i][j].append([])
                    I[i][j][k] = self.single_point_current(row, col, pillar)

        self.I.append(I)

    # can these functions be templated somehow?
    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)})"

    def __str__(self):
        s = "    " + repr(self) + "\n"
        x = f"[{self.x[0]}, ... , {self.x[-1]}]"
        y = f"[{self.y[0]}, ... , {self.y[-1]}]"
        z = f"[{self.z[0]}, ... , {self.z[-1]}]"
        s += f"        @ x={x}, y={y}, z={z}\n"
        return s

    def detector_values(self):
        """ outputs what detector detects """
        return {"I": self.I}
