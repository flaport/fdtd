""" Python 3D FDTD Simulator """

__author__ = "Floris laporte"
__version__ = "0.0.0"

from .grid import Grid
from .sources import LineSource
from .detectors import LineDetector
from .objects import Object, AnisotropicObject
from .boundaries import PeriodicBoundary, PML
from .backend import backend
from .backend import set_backend
