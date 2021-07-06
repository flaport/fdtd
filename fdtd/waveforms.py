

# For Hanning window pulses
def hanning(f, t, n):
    return (1 / 2) * (1 - cos(f * t / n)) * (sin(f * t))


'''

Presumably because there are smooth, containing minimal numerical noise, but also

I don't know what advantages the gaussian derivative provides.

Gaussian pulses

For derivation, see
https://github.com/0xDBFB7/fdtd_PCB_extensions/blob/master/normalizedgaussian.nb.pdf

http://www.cse.yorku.ca/~kosta/CompVis_Notes/fourier_transform_Gaussian.pdf
http://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/04%20Gaussian%20derivatives.pdf

'''

def normalized_gaussian_pulse(x,fwhm):
    sigma = fwhm/2.355
    return exp(-((x**2.0)/(2.0*(sigma**2.0))))

def normalized_gaussian_derivative_pulse(x,fwhm):
    sigma = fwhm/2.355
    return (exp((1.0/2.0) - (x**2.0)/(2.0*sigma**2.0))*x)/sigma
