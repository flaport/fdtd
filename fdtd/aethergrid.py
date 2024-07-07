""" The FDTD Aether Grid

The grid is the core of the FDTD Library. It is where everything comes together
and where the biggest part of the calculations are done.

This grid is an experimental implementation of a new aether theory, based on the
discovery of the quantum circulation constant, kinematic viscosity or
diffusivity k, with a value equal to light speed c squared but a unit of
[m^2/s].
    
This theory is documented in this jupyter notebook:
    
https://github.com/l4m4re/notebooks/blob/main/aether_physics.ipynb
        
Key point is that the quantum circulation constant can be combined with the
vector LaPlace operator to define the time derivative of any given vector field
F within the aether by:

dF/dt = -k Delta F,

with Delta the LaPlace operator. 

From here, we can define the acceleration field a as the time derivative of the
velocity field v by:

a = d/dt v = -k Delta v,

and jerk j as the time derivative of the acceleration field a by:

j = d/dt a = -k Delta a = k^2 Delta^2 v.

This way, we obtain a second order model, in contrast to Maxwell's equations as
well as Navier-Stokes equations, so we can define second order LaPlace and
Poisson equations in full 3D, which was heretofore impossible.

An interesting detail is that the wave equation for the velocity field v can be
written as:

j/c^2 + a/k = 0,

which is a full 3D second order wave equation, illustrating the expressive power
of utilizing the vector LaPlace operator, the second spatial derivative, when
combined with the quantum circulation constant k.

By writing out the definition of the vector LaPlacian for the acceleration and
jerk fields, various fields can be defined, amongst others the electric and
magnetic fields as well as uniquely defined scalar and vector potentials,
leaving no room for "gauge fixing".

Another key point is that there are only three units of measurement in this
model: the meter, the second and the kilogram. Within this model, electric
charge has a unit in [kg/s], while the electric field has a unit of velocity in
[m/s]. And the magnetic field B has a unit in [/m-s], while the magnetizing field
H has a unit in [kg/m^2-s^2].

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

from .operators import curl_edge_to_face, curl_face_to_edge, grad, div


from math import pi

## Constants

# base constants

c       = 299792458.0       # speed of light                [m/s]

eta     = 1/(4*pi*1e-7)     # viscosity (1/mu_0)            [kg/m-s],   [Pa-s]

h       = 6.62607015e-34    # Planck's constant             [kg-m^2/s], [J-s]

e       = 1.602176634e-19   # elementary charge             [kg/s]

# derived constants

k       = c**2              # quantum circulation constant
                            # 8.987551787368176e+16         [m^2/s]

rho     = eta/k             # mass density (eps_0) 
                            # 8.85418781762039e-12          [kg/m^3]

m       = h/k               # elementary mass 
                            # 7.372497323812708e-51         [kg]
                      
rho_q0  = e/m * rho         # vacuum charge density 
                            # 1.9241747011042014e+20        [kg/m^3-s]

sigma  = e/k                # Surface mass density
                            # 1.7826619216278975e-36        [kg/m^2], [kg-m/m^3]

inv_rho_q0 = 1/rho_q0      
rho_sigma    = rho/sigma        # = eta/e
sigma_rho    = sigma/rho        # = e/eta

## FDTD Grid Class
class AetherGrid:
    """The FDTD Aether Grid

    The grid is the core of the FDTD Library. It is where everything comes
    together and where the biggest part of the calculations are done.
    
    
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
            
        """
        For now, we assume our fields to propagate with a maximum speed of pi/2
        * c, since we assume Tesla's longitudinal sound-like waves to also exist
        and to propagate at that speed, even though Dr. Steffen Kuehn has
        demonstrated the transmission of information in electrically short
        coaxial cables at speeds of up to 3c, which is a strong argument to be
        made that the speed of light is not the maximum speed of propagation of
        dielectric wave phenomena nor information. 
        
        It is these dielectric wave phenomena that are distincly different from
        the electromagnetic waves we are familiar with that appear to have been
        overlooked by science, even though Tesla demonstrated the transmission
        and reception of telluric currents with a speed of pi/2*c, even
        wirelessly powering light bulbs at a distance of over half a mile from
        his laboratory in Colorado Springs, according to Hugo Gernsback.
        """
        # timestep of the simulation
        self.time_step = self.courant_number * self.grid_spacing / ((pi/2) * c)
        
        # define fields
        # velocity
        self.v = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        # first order scalar and vector potentials
        self.p = bd.zeros((self.Nx, self.Ny, self.Nz))
        self.A = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        # first order force and torque density fields
        self.L = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.R = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        # first order electric and magnetic fields
        self.E = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.H = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        # acceleration
        self.a = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        # second order scalar and vector potentials
        self.dpdt = bd.zeros((self.Nx, self.Ny, self.Nz))
        self.dAdt = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        # second order force and torque density fields
        self.dLdt = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.dRdt = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        # second order electric and magnetic fields
        self.dEdt = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.dHdt = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        # jerk
        self.j = bd.zeros((self.Nx, self.Ny, self.Nz, 3))
        
        # save the inverse of the relative permittiviy and the relative permeability
        # these tensors can be anisotropic!

        if bd.is_array(permittivity) and len(permittivity.shape) == 3:
            permittivity = permittivity[:, :, :, None]
        self.inverse_permittivity = bd.ones((self.Nx, self.Ny, self.Nz, 3)) / bd.array(
            permittivity, dtype=bd.float
        )

        if bd.is_array(permeability) and len(permeability.shape) == 3:
            permeability = permeability[:, :, :, None]
        self.inverse_permeability = bd.ones((self.Nx, self.Ny, self.Nz, 3)) / bd.array(
            permeability, dtype=bd.float
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
        """transform a distance to an integer number of gridpoints"""
        if not isinstance(distance, int):
            return int(float(distance) / self.grid_spacing + 0.5)
        return distance

    def _handle_time(self, time: Number) -> int:
        """transform a time value to an integer number of timesteps"""
        if not isinstance(time, int):
            return int(float(time) / self.time_step + 0.5)
        return time

    def _handle_tuple(
        self, shape: Tuple[Number, Number, Number]
    ) -> Tuple[int, int, int]:
        """validate the grid shape and transform to a length-3 tuple of ints"""
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
        """validate the slice and transform possibly float values to ints"""
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
        """transform a single index key to a slice or list"""
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
        """get the number of grid cells in the x-direction"""
        return self.Nx * self.grid_spacing

    @property
    def y(self) -> int:
        """get the number of grid cells in the y-direction"""
        return self.Ny * self.grid_spacing

    @property
    def z(self) -> int:
        """get the number of grid cells in the y-direction"""
        return self.Nz * self.grid_spacing

    @property
    def shape(self) -> Tuple[int, int, int]:
        """get the shape of the FDTD grid"""
        return (self.Nx, self.Ny, self.Nz)

    @property
    def time_passed(self) -> float:
        """get the total time passed"""
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
        """do a single FDTD step by first computing acceleration, jerk
        and the intermediate fields, and then updating the velocity field.
        """

        self.update()
        
        self.time_steps_passed += 1
        
    def update(self):
        """ update the fields along the vector Laplace operator """
        
        # update boundaries: step 1
        #for boundary in self.boundaries:
        #    boundary.update_phi_H()
        
        #curl = curl_E(self.E)
        #self.H -= self.courant_number * self.inverse_permeability * curl
        
        # potential fields
        self.p      = eta * div (self.v)
        self.A      = e   * curl_edge_to_face(self.v)
        
        # force and torque density fields
        self.L      = - grad(self.p)
        self.R      = curl_face_to_edge(self.A)
        
        # electric and magnetic fields
        self.E      = inv_rho_q0 * self.L
        self.H      = rho_sigma    * self.R
        
        # acceleration field
        self.a      = rho_q0 * self.E + sigma_rho * self.H
        
        # second order potential fields        
        self.dpdt   = eta * div (self.a)
        self.dAdt   = e   * curl_edge_to_face(self.a)
        
        #second order yank and d/dt torque density fields
        self.dLdt   = - grad(self.dpdt)
        self.dRdt   = curl_face_to_edge(self.dAdt)
        
        # second order (time derivative) of electric and magnetic fields
        self.dEdt   = inv_rho_q0 * self.dLdt
        self.dHdt   = rho_sigma    * self.dRdt
        
        # jerk field
        self.j      = rho_q0 * self.dEdt + sigma_rho * self.dHdt
    
        # update velocity field
        self.v      += self.courant_number    * self.a
        self.v      += self.courant_number**2 * self.j
        
        
        # update objects
        #for obj in self.objects:
        #    obj.update_H(curl)
           
        # update boundaries: step 2
        #for boundary in self.boundaries:
        #    boundary.update_H()
           
        # add sources to grid:
        #for src in self.sources:
        #    src.update_H()
        #    src.update_E()
           
           
        # detect electric field
        #for det in self.detectors:
        #    det.detect_H()    



    def reset(self):
        """reset the grid by setting all fields to zero"""
        self.v *= 0.0
        self.p *= 0.0
        self.A *= 0.0
        self.L *= 0.0
        self.R *= 0.0
        self.H *= 0.0
        self.E *= 0.0
        
        self.a    *= 0.0
        self.dpdt *= 0.0
        self.dAdt *= 0.0
        self.dLdt *= 0.0
        self.dRdt *= 0.0
        self.dEdt *= 0.0
        self.dHdt *= 0.0
        
        self.j *= 0.0
        
        self.time_steps_passed *= 0

    def add_source(self, name, source):
        """add a source to the grid"""
        source._register_grid(self)
        self.sources[name] = source

    def add_boundary(self, name, boundary):
        """add a boundary to the grid"""
        boundary._register_grid(self)
        self.boundaries[name] = boundary

    def add_detector(self, name, detector):
        """add a detector to the grid"""
        detector._register_grid(self)
        self.detectors[name] = detector

    def add_object(self, name, obj):
        """add an object to the grid"""
        obj._register_grid(self)
        self.objects[name] = obj
    
    def promote_dtypes_to_complex(self):
        self.E = self.E.astype(bd.complex)
        self.H = self.H.astype(bd.complex)
        [boundary.promote_dtypes_to_complex() for boundary in self.boundaries]

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
        def _numpyfy(item):
            if isinstance(item, list):
                return [_numpyfy(el) for el in item]
            elif bd.is_array(item):
                return bd.numpy(item)
            else:
                return item
                
        if self.folder is None:
            raise Exception(
                "Save location not initialized. Please read about 'fdtd.Grid.saveSimulation()' or try running 'grid.saveSimulation()'."
            )
        dic = {}
        for detector in self.detectors:
            values = detector.detector_values()
            dic[detector.name + " (E)"] = _numpyfy(values['E'])
            dic[detector.name + " (H)"] = _numpyfy(values['H'])
        savez(path.join(self.folder, "detector_readings"), **dic)
