# reverse literate programming?

from math import sin, cos, sqrt, log, exp

import numpy as np

# For Hanning window pulses
def hanning(f, t, n):
    return (1 / 2) * (1 - cos(f * t / n)) * (sin(f * t))


"""

Recipes for Z measurements usually specify gaussian or gaussian derivative input pulses.

Presumably because unlike a unit step or simply gated sine,
they're smooth, introducing minimal numerical noise, while retaining broadband frequency components.

I don't know what advantages the gaussian derivative provides over the gaussian.

These gaussian pulses are normalized to have a peak value of 1.0.

For derivation, see
https://github.com/0xDBFB7/fdtd_PCB_extensions/blob/master/normalizedgaussian.nb.pdf

http://www.cse.yorku.ca/~kosta/CompVis_Notes/fourier_transform_Gaussian.pdf
http://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/04%20Gaussian%20derivatives.pdf

"""

fwhm_constant = 2.0 * sqrt(2.0 * log(2))


def normalized_gaussian_pulse(x, fwhm, center=0.0):
    # '''
    # x - input value, usually time
    # fwhm - pulse Full Width Half Maximum
    # Center - just for convenience, used to move pulse time around
    # '''
    # apparently FWHM of t is properly called "full duration half maximum",
    # but I've never heard that used

    # TODO: flag on input type and use backend?

    sigma = fwhm / fwhm_constant
    return np.exp(-(((x - center) ** 2.0) / (2.0 * (sigma ** 2.0))))


def normalized_gaussian_derivative_pulse(x, fwhm, center=0.0):
    sigma = fwhm / fwhm_constant
    return (
        exp((1.0 / 2.0) - ((x - center) ** 2.0) / (2.0 * sigma ** 2.0)) * (x - center)
    ) / sigma
