""" The FDTD LGrid

The grid is the core of the FDTD Library. It is where everything comes
together and where the biggest part of the calculations are done.

"""

## Imports

# standard library
import os
from os import path, makedirs, chdir, remove
from subprocess import check_call, CalledProcessError
from glob import glob
from datetime import datetime

# 3rd party
from tqdm import tqdm
from numpy import savez

# typing
from .typing_ import Tuple, Number, Tensorlike

# relative
from .backend import backend as bd
from . import constants as const

## Functions
def curl_E(E: Tensorlike) -> Tensorlike:
    """Transforms an E-type field into an H-type field by performing a curl
    operation

    Args:
        E: Electric field to take the curl of (E-type field located on the
           edges of the grid cell [integer gridpoints])

    Returns:
        The curl of E (H-type field located on the faces of the grid [half-integer grid points])
    """
    curl = bd.zeros(E.shape)

    curl[:, :-1, :, 0] += E[:, 1:, :, 2] - E[:, :-1, :, 2]
    curl[:, :, :-1, 0] -= E[:, :, 1:, 1] - E[:, :, :-1, 1]

    curl[:, :, :-1, 1] += E[:, :, 1:, 0] - E[:, :, :-1, 0]
    curl[:-1, :, :, 1] -= E[1:, :, :, 2] - E[:-1, :, :, 2]

    curl[:-1, :, :, 2] += E[1:, :, :, 1] - E[:-1, :, :, 1]
    curl[:, :-1, :, 2] -= E[:, 1:, :, 0] - E[:, :-1, :, 0]

    return curl


def curl_H(H: Tensorlike) -> Tensorlike:
    """Transforms an H-type field into an E-type field by performing a curl
    operation

    Args:
        H: Magnetic field to take the curl of (H-type field located on half-integer grid points)

    Returns:
        The curl of H (E-type field located on the edges of the grid [integer grid points])

    """
    curl = bd.zeros(H.shape)

    curl[:, 1:, :, 0] += H[:, 1:, :, 2] - H[:, :-1, :, 2]
    curl[:, :, 1:, 0] -= H[:, :, 1:, 1] - H[:, :, :-1, 1]

    curl[:, :, 1:, 1] += H[:, :, 1:, 0] - H[:, :, :-1, 0]
    curl[1:, :, :, 1] -= H[1:, :, :, 2] - H[:-1, :, :, 2]

    curl[1:, :, :, 2] += H[1:, :, :, 1] - H[:-1, :, :, 1]
    curl[:, 1:, :, 2] -= H[:, 1:, :, 0] - H[:, :-1, :, 0]

    return curl


def div_E(E: Tensorlike) -> Tensorlike:
    """Computes the divergence of an E-type field located on the edges of the grid cells
    Args:
        E: E-type field of shape (x, y, z, 3) 
    Returns:
        The divergence of E, a scalar field of shape (x, y, z, 1)

    """
    
    div = bd.zeros(E.shape[:-1] + (1,))
    
    div[:, :, :-1, 0] += E[:, :, 1:, 0] - E[:, :, :-1, 0]
    div[:, :-1, :, 0] += E[:, 1:, :, 1] - E[:, :-1, :, 1]
    div[:-1, :, :, 0] += E[1:, :, :, 2] - E[:-1, :, :, 2]
    
    return div

def div_H(H: Tensorlike) -> Tensorlike:
    """Computes the divergence of an H-type field located on the faces of the grid
    Args:
        H: H-type field of shape (x, y, z, 3) 
    Returns:
        The divergence of H, a scalar field of shape (x, y, z, 1)
        """
        
    div = bd.zeros(H.shape[:-1]+(1,))
    
    div[:, :-1, :, 0] += H[:, :-1, :, 2] - H[:, 1:, :, 2]
    div[:, :, :-1, 0] -= H[:, :, :-1, 1] - H[:, :, 1:, 1]
    div[:-1, :, :, 0] += H[:-1, :, :, 0] - H[1:, :, :, 0]
    
    return div


def grad_E(E: Tensorlike) -> Tensorlike:
    """Computes the gradient of a scalar field
    Args:
        E: Electric field of shape (x, y, z, 3)
    Returns:
        The gradient of E, a vector field of shape (x, y, z, 3) located on the edges of the grid cells
    """
    
    grad = bd.zeros((*E.shape[:-1], 3))
    
    grad[:-1, :, :, 0] = E[1:, :, :, 0] - E[:-1, :, :, 0]
    grad[:, :-1, :, 1] = E[:, 1:, :, 0] - E[:, :-1, :, 0]
    grad[:, :, :-1, 2] = E[:, :, 1:, 0] - E[:, :, :-1, 0]
    
    return grad



def grad_H(H: Tensorlike) -> Tensorlike:
    """Computes the gradient of a scalar field
    Args:
        H: Magnetic field of shape (x, y, z, 3)
    Returns:
        The gradient of H, a vector field of shape (x, y, z, 3) located on the faces of the grid
    
    """

    grad = bd.zeros((*H.shape[:-1], 3))
    
    grad[:, :-1, :, 0] = H[:, :-1, :, 0] - H[:, 1:, :, 0]
    grad[:-1, :, :, 1] = H[:-1, :, :, 0] - H[1:, :, :, 0]
    grad[:, :, :-1, 2] = H[:, :, :-1, 0] - H[:, :, 1:, 0]
    
    return grad
    


## FDTD LGrid Class
class LGrid:
    """The FDTD LGrid

    The LGrid is the core of the FDTD Library. It is where everything comes
    together and where the biggest part of the calculations are done.
    
    This implementation implements fields based on the Laplace operator
    rather than Maxwells equations

    """

    from .visualization import visualize

    def __init__(
        self,
        shape: Tuple[Number, Number, Number],
        grid_spacing: float = 155e-9,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        courant_number: float = None,
    ):
        """
        Args:
            shape: shape of the FDTD grid.
            grid_spacing: distance between the grid cells.
            permittivity: the relative permittivity of the background.
            permeability: the relative permeability of the background.
            courant_number: the courant number of the FDTD simulation.
                Defaults to the inverse of the square root of the number of
                dimensions > 1 (optimal value). The timestep of the simulation
                will be derived from this number using the CFL-condition.
        """
        # save the grid spacing
        self.grid_spacing = float(grid_spacing)

        # save grid shape as integers
        self.Nx, self.Ny, self.Nz = self._handle_tuple(shape)

        # dimension of the simulation:
        self.D = int(self.Nx > 1) + int(self.Ny > 1) + int(self.Nz > 1)

        # courant number of the simulation (optimal value)
        max_courant_number = float(self.D) ** (-0.5)
        if courant_number is None:
            # slight stability factor added
            self.courant_number = 0.99 * max_courant_number
        elif courant_number > max_courant_number:
            raise ValueError(
                f"courant_number {courant_number} too high for "
                f"a {self.D}D simulation"
            )
        else:
            self.courant_number = float(courant_number)

        # timestep of the simulation
        self.time_step = self.courant_number * self.grid_spacing / const.c

        # save electric and magnetic field
        #self.E          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        #self.H          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        self.k          = float(8.987551787368176e+16)
        
        self.C          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        self.P          = bd.zeros((self.Nx, self.Ny, self.Nz, 1))
        self.A          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))

        self.E          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.H          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))

        self.F          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.I          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        self.T          = bd.zeros((self.Nx, self.Ny, self.Nz, 1))
        self.Chi        = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        self.Eambda     = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.Psi        = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        self.Y          = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        
        

        # save the inverse of the relative permittiviy and the relative permeability
        # these tensors can be anisotropic!

        if bd.is_array(permittivity) and len(permittivity.shape) == 3:
            permittivity = permittivity[:, :, :, None]
        self.inverse_permittivity = bd.ones((self.Nx, self.Ny, self.Nz, 3)) / float(
            permittivity
        )

        if bd.is_array(permeability) and len(permeability.shape) == 3:
            permeability = permeability[:, :, :, None]
        self.inverse_permeability = bd.ones((self.Nx, self.Ny, self.Nz, 3)) / float(
            permeability
        )

        # save current time index
        self.time_steps_passed = 0

        # dictionary containing the sources:
        self.sources = []

        # dictionary containing the boundaries
        self.boundaries = []

        # dictionary containing the detectors
        self.detectors = []

        # dictionary containing the objects in the grid
        self.objects = []

        # folder path to store the simulation
        self.folder = None

    def _handle_distance(self, distance: Number) -> int:
        """ transform a distance to an integer number of gridpoints """
        if not isinstance(distance, int):
            return int(float(distance) / self.grid_spacing + 0.5)
        return distance

    def _handle_time(self, time: Number) -> int:
        """ transform a time value to an integer number of timesteps """
        if not isinstance(time, int):
            return int(float(time) / self.time_step + 0.5)
        return time

    def _handle_tuple(
        self, shape: Tuple[Number, Number, Number]
    ) -> Tuple[int, int, int]:
        """ validate the grid shape and transform to a length-3 tuple of ints """
        if len(shape) != 3:
            raise ValueError(
                f"invalid grid shape {shape}\n"
                f"grid shape should be a 3D tuple containing floats or ints"
            )
        x, y, z = shape
        x = self._handle_distance(x)
        y = self._handle_distance(y)
        z = self._handle_distance(z)
        return x, y, z

    def _handle_slice(self, s: slice) -> slice:
        """ validate the slice and transform possibly float values to ints """
        start = (
            s.start
            if not isinstance(s.start, float)
            else self._handle_distance(s.start)
        )
        stop = (
            s.stop if not isinstance(s.stop, float) else self._handle_distance(s.stop)
        )
        step = (
            s.step if not isinstance(s.step, float) else self._handle_distance(s.step)
        )
        return slice(start, stop, step)

    def _handle_single_key(self, key):
        """ transform a single index key to a slice or list """
        try:
            len(key)
            return [self._handle_distance(k) for k in key]
        except TypeError:
            if isinstance(key, slice):
                return self._handle_slice(key)
            else:
                return [self._handle_distance(key)]
        return key

    @property
    def x(self) -> int:
        """ get the number of grid cells in the x-direction """
        return self.Nx * self.grid_spacing

    @property
    def y(self) -> int:
        """ get the number of grid cells in the y-direction """
        return self.Ny * self.grid_spacing

    @property
    def z(self) -> int:
        """ get the number of grid cells in the y-direction """
        return self.Nz * self.grid_spacing

    @property
    def shape(self) -> Tuple[int, int, int]:
        """ get the shape of the FDTD grid """
        return (self.Nx, self.Ny, self.Nz)

    @property
    def time_passed(self) -> float:
        """ get the total time passed """
        return self.time_steps_passed * self.time_step

    def run(self, total_time: Number, progress_bar: bool = True):
        """run an FDTD simulation.

        Args:
            total_time: the total time for the simulation to run.
            progress_bar: choose to show a progress bar during
                simulation

        """
        if isinstance(total_time, float):
            total_time /= self.time_step
        time = range(0, int(total_time), 1)
        if progress_bar:
            time = tqdm(time)
        for _ in time:
            self.step()

    def step(self):
        """do a single FDTD step by first updating the electric field and then
        updating the magnetic field
        """
        #self.update_E()    # [E]       += c*dt*inv(ε)*curl(H) 
        #self.update_H()
        
        """
        self.update_P()     # [P]       = div [C]
        self.update_A()     # [A]       = curl [C]

        self.update_L()     # [L]       = grad [P]
        self.update_R()     # [R]       = curl [Omega]

        self.update_F()     # [F]       = [L] + [R]
        self.update_I()     # [I]       = k [F]
        
        self.update_T()     # [T]       = div  [I]
        self.update_Chi()   # [Chi]     = curl [I]
        
        self.update_Lambda()# [Lambda]  = grad [T]
        self.update_Psi()   # [Psi]     = curl [Chi]
        
        self.update_Y()     # [Y]       = [Lambda] + [Psi]
        """
        
        self.update_C()     # [C]      += [F]*dt + [Y]*dt^2
        
        
        self.time_steps_passed += 1

    def update_E(self):
        """ update the electric field by using the curl of the magnetic field """

        # update boundaries: step 1
        for boundary in self.boundaries:
            boundary.update_phi_E()

        curl = curl_H(self.H)
        self.E += self.courant_number * self.inverse_permittivity * curl

        # update objects
        for obj in self.objects:
            obj.update_E(curl)

        # update boundaries: step 2
        for boundary in self.boundaries:
            boundary.update_E()

        # add sources to grid:
        for src in self.sources:
            src.update_E()

        # detect electric field
        for det in self.detectors:
            det.detect_E()

    def update_H(self):
        """ update the magnetic field by using the curl of the electric field """

        # update boundaries: step 1
        for boundary in self.boundaries:
            boundary.update_phi_H()

        curl = curl_E(self.E)
        self.H -= self.courant_number * self.inverse_permeability * curl

        # update objects
        for obj in self.objects:
            obj.update_H(curl)

        # update boundaries: step 2
        for boundary in self.boundaries:
            boundary.update_H()

        # add sources to grid:
        for src in self.sources:
            src.update_H()

        # detect electric field
        for det in self.detectors:
            det.detect_H()


    def update_C(self):
        """ update the C field by using the vector Laplace operator """
        
        # update boundaries: step 1
        #for boundary in self.boundaries:
        #    boundary.update_phi_H()
        
        #curl = curl_E(self.E)
        #self.H -= self.courant_number * self.inverse_permeability * curl
        
        # Since C represents a force density, it would be located on
        # the faces of the grid and thus be a H-type field.
        
        self.P      = div_H (self.C)
        self.A      = curl_H(self.C)
        
        self.E      = grad_H(self.P)
        self.H      = curl_E(self.A)
        
        self.F      = self.E + self.H
        
        self.I      = self.k * self.F
        
        self.T      = div_H(self.I)
        self.Chi    = curl_H(self.I)
        
        self.Eambda = grad_H(self.T)
        self.Psi    = curl_H(self.Chi)
        
        self.Y      = self.Eambda + self.Psi
    
        
        self.C      -= self.courant_number * self.k * self.F
        self.C      -= self.courant_number * self.courant_number * self.k * self.Y
    
    
        # update objects
        #for obj in self.objects:
        #    obj.update_H(curl)
           
        # update boundaries: step 2
        #for boundary in self.boundaries:
        #    boundary.update_H()
           
        # add sources to grid:
        for src in self.sources:
            src.update_H()
            src.update_E()
           
           
        # detect electric field
        #for det in self.detectors:
        #    det.detect_H()
        


    def reset(self):
        """ reset the grid by setting all fields to zero """
        #self.H *= 0.0
        #self.E *= 0.0
        self.C          *= 0.0
        self.P          *= 0.0
        self.A          *= 0.0
        self.E          *= 0.0
        self.H          *= 0.0
        self.F          *= 0.0
        self.I          *= 0.0
        self.T          *= 0.0
        self.Chi        *= 0.0
        self.Eambda     *= 0.0
        self.Psi        *= 0.0
        self.Y          *= 0.0
        
        self.time_steps_passed *= 0

    def add_source(self, name, source):
        """ add a source to the grid """
        source._register_grid(self)
        self.sources[name] = source

    def add_boundary(self, name, boundary):
        """ add a boundary to the grid """
        boundary._register_grid(self)
        self.boundaries[name] = boundary

    def add_detector(self, name, detector):
        """ add a detector to the grid """
        detector._register_grid(self)
        self.detectors[name] = detector

    def add_object(self, name, obj):
        """ add an object to the grid """
        obj._register_grid(self)
        self.objects[name] = obj

    def __setitem__(self, key, attr):
        if not isinstance(key, tuple):
            x, y, z = key, slice(None), slice(None)
        elif len(key) == 1:
            x, y, z = key[0], slice(None), slice(None)
        elif len(key) == 2:
            x, y, z = key[0], key[1], slice(None)
        elif len(key) == 3:
            x, y, z = key
        else:
            raise KeyError("maximum number of indices for the grid is 3")

        attr._register_grid(
            grid=self,
            x=self._handle_single_key(x),
            y=self._handle_single_key(y),
            z=self._handle_single_key(z),
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape=({self.Nx},{self.Ny},{self.Nz}), "
            f"grid_spacing={self.grid_spacing:.2e}, courant_number={self.courant_number:.2f})"
        )

    def __str__(self):
        """string representation of the grid

        lists all the components and their locations in the grid.
        """
        s = repr(self) + "\n"
        if self.sources:
            s = s + "\nsources:\n"
            for src in self.sources:
                s += str(src)
        if self.detectors:
            s = s + "\ndetectors:\n"
            for det in self.detectors:
                s += str(det)
        if self.boundaries:
            s = s + "\nboundaries:\n"
            for bnd in self.boundaries:
                s += str(bnd)
        if self.objects:
            s = s + "\nobjects:\n"
            for obj in self.objects:
                s += str(obj)
        return s

    def save_simulation(self, sim_name=None):
        """
        Creates a folder and initializes environment to store simulation or related details.
        saveSimulation() needs to be run before running any function that stores data (generate_video(), save_data()).

        Parameters:-
            (optional) sim_name (string): Preferred name for simulation
        """
        makedirs("fdtd_output", exist_ok=True)  # Output master folder declaration
        # making full_sim_name with timestamp
        full_sim_name = (
            str(datetime.now().year)
            + "-"
            + str(datetime.now().month)
            + "-"
            + str(datetime.now().day)
            + "-"
            + str(datetime.now().hour)
            + "-"
            + str(datetime.now().minute)
            + "-"
            + str(datetime.now().second)
        )
        # Simulation name (optional)
        if sim_name is not None:
            full_sim_name = full_sim_name + " (" + sim_name + ")"
        folder = "fdtd_output_" + full_sim_name
        # storing folder path for saving simulation
        self.folder = os.path.abspath(path.join("fdtd_output", folder))
        # storing timestamp title for self.generate_video
        self.full_sim_name = full_sim_name
        makedirs(self.folder, exist_ok=True)
        return self.folder

    def generate_video(self, delete_frames=False):
        """Compiles frames into a video

        These framed should be saved through ``fdtd.Grid.visualize(save=True)`` while having ``fdtd.Grid.save_simulation()`` enabled.

        Args:
            delete_frames (optional, bool): delete stored frames after conversion to video.

        Returns:
            the filename of the generated video.

        Note:
            this function requires ``ffmpeg`` to be available in your path.
        """
        if self.folder is None:
            raise Exception(
                "Save location not initialized. Please read about 'fdtd.Grid.saveSimulation()' or try running 'grid.saveSimulation()'."
            )
        cwd = path.abspath(os.getcwd())
        chdir(self.folder)
        try:
            check_call(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "8",
                    "-i",
                    "file%04d.png",
                    "-r",
                    "30",
                    "-pix_fmt",
                    "yuv420p",
                    "fdtd_sim_video_" + self.full_sim_name + ".mp4",
                ]
            )
        except (FileNotFoundError, CalledProcessError):
            raise CalledProcessError(
                "Error when calling ffmpeg. Is ffmpeg installed and available in your path?"
            )
        if delete_frames:  # delete frames
            for file_name in glob("*.png"):
                remove(file_name)
        video_path = path.abspath(
            path.join(self.folder, f"fdtd_sim_video_{self.full_sim_name}.mp4")
        )
        chdir(cwd)
        return video_path

    def save_data(self):
        """
        Saves readings from all detectors in the grid into a numpy zip file. Each detector is stored in separate arrays. Electric and magnetic field field readings of each detector are also stored separately with suffix " (E)" and " (H)" (Example: ['detector0 (E)', 'detector0 (H)']). Therefore, the numpy zip file contains arrays twice the number of detectors.
        REQUIRES 'fdtd.Grid.save_simulation()' to be run before this function.

        Parameters: None
        """
        if self.folder is None:
            raise Exception(
                "Save location not initialized. Please read about 'fdtd.Grid.saveSimulation()' or try running 'grid.saveSimulation()'."
            )
        dic = {}
        for detector in self.detectors:
            dic[detector.name + " (E)"] = [x for x in detector.detector_values()["E"]]
            dic[detector.name + " (H)"] = [x for x in detector.detector_values()["H"]]
        savez(path.join(self.folder, "detector_readings"), **dic)
