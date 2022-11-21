""" Python 3D FDTD Simulator """

__author__ = "Floris laporte"
__version__ = "0.2.6"

from .grid import Grid
from .sources import PointSource, LineSource, PlaneSource
from .detectors import LineDetector, BlockDetector, CurrentDetector
from .objects import Object, AbsorbingObject, AnisotropicObject
from .boundaries import PeriodicBoundary, PML
from .backend import backend
from .backend import set_backend
from .fourier import FrequencyRoutines
from .visualization import dB_map_2D, plot_detection
