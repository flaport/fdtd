""" Test the FFT system """

## Imports
import fdtd
import pytest
from fixtures import grid, pml, periodic_boundary




## Tests


def test_padding(grid):
    fr = FrequencyRoutines(grid)
    input_data = np.zeros((1000))
    dt = 1e-9
    #sample
    freq_window = ()

    padding_and_timing(self, input_data, freq_window_tuple=None, fft_num_bins_in_window=None,
                                        fft_bin_freq_resolution=None, dt=dt)









                                        # use QUCS to generate something 3 port with known S-params
