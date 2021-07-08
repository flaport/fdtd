from math import pi, sin, cos
from .typing import Tuple, Number, ListOrSlice, List
from .grid import Grid
from .backend import backend as bd

'''
Optional reduced unit conversion functions for user use.


Might perhaps be worth putting a note in the readme about default units / suggested unit systems?
Also, if /fdtd/ gets the ability to dump to VTK, one would hate to
have to scale everything for physical results in paraview.
Is it reasonable to have a global flag for scaling?

don't know how severe noise is with fp32, whether it's worth the extra hassle for most people
Might be better suited to a user-side

Unit system described by the thesis
"Novel architectures for brain-inspired photonic computers",
https://www.photonics.intec.ugent.be/download/phd_259.pdf

Chapters 4.1.2 and 4.1.6.

"In SI units, the relative magnitude difference between the fields is related
by the electromagnetic impedance of free space, which for the current choice
of simulation units equals 1 per definition".

'''

#
# def tildeE_to_E(input):
#     '''
#     Where barE is the reduced simulation unit and E is the world unit
#     '''
#
#     return
#
# def E_to_sim_E(input):
#     '''
#     Where barE is the reduced simulation unit and E is the world unit
#     '''
#
#     return
#
#
# def tildeH_to_H(input):
#     return
