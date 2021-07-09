#reverse literate programming?

from math import pi, sin, cos, sqrt, log, exp

# For Hanning window pulses
def hanning(f, t, n):
    return (1 / 2) * (1 - cos(f * t / n)) * (sin(f * t))

'''

Presumably because unlike a unit step or gated sine,
they're smooth, introducing minimal numerical noise, while retaining broadband frequency components.

I don't know what advantages the gaussian derivative provides over the gaussian.

These gaussian pulses are normalized to have a peak value of 1.0.

For derivation, see
https://github.com/0xDBFB7/fdtd_PCB_extensions/blob/master/normalizedgaussian.nb.pdf

http://www.cse.yorku.ca/~kosta/CompVis_Notes/fourier_transform_Gaussian.pdf
http://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/04%20Gaussian%20derivatives.pdf

'''

fwhm_constant = 2.0*sqrt(2.0 * log(2))

def normalized_gaussian_pulse(x,fwhm):
    #apparently FWHM of t is properly called "full duration half maximum",
    #but I've never heard that used
    sigma = fwhm/fwhm_constant
    return exp(-((x**2.0)/(2.0*(sigma**2.0))))

def normalized_gaussian_derivative_pulse(x,fwhm):
    sigma = fwhm/fwhm_constant
    return (exp((1.0/2.0) - (x**2.0)/(2.0*sigma**2.0))*x)/sigma