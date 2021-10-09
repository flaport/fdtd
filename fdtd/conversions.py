from math import sqrt

from . import constants as const

"""
Optional reduced unit conversion functions for user use.
Mainly to make it explicit where conversions happen.

Unit system described by the thesis
"Novel architectures for brain-inspired photonic computers",
https://www.photonics.intec.ugent.be/download/phd_259.pdf

Chapters 4.1.2 and 4.1.6.

> "In SI units, the relative magnitude difference between the fields is related
by the electromagnetic impedance of free space, which for the current choice
of simulation units equals 1 per definition".

FIXME: DC: find and add md notes on scaling

"""

# Might perhaps be worth putting a note in the readme about default units / suggested unit systems?
# (done, see next PR)
# Also, if /fdtd/ gets the ability to dump to VTK, one would hate to
# have to scale everything for physical results in paraview. Will put a scaling option there.
# Is it reasonable to have a global flag for scaling? probably not
#
# don't know how severe noise is with fp32, whether it's worth the extra hassle for most people


def simE_to_worldE(input):
    return input / sqrt(const.eps0)


def worldE_to_simE(input):
    return sqrt(const.eps0) * input


def simH_to_worldH(input):
    return input / sqrt(const.mu0)


def worldH_to_simH(input):
    return sqrt(const.mu0) * input
