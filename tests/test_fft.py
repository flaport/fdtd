""" Test the FFT system """

## Imports
import fdtd
import pytest
from fixtures import grid, pml, periodic_boundary

from fdtd.fourier import FrequencyRoutines
from fdtd.backend import backend as bd



## Tests


def test_padding(grid):
    fr = FrequencyRoutines(grid)
    input_data = bd.zeros((1000))
    dt = 1e-9
    #sample rate 1 GHz
    #1 microsecond capture length
    #total length
    # freq_window = ()


    times, required_padding, spectrum_freqs, (begin_freq_idx, end_freq_idx), end_time = \
            fr.compute_padding_and_timing(input_data, dt=dt, freq_window_tuple=None, fft_num_bins_in_window=None,
                                        fft_bin_freq_resolution=None, )

    assert end_time == pytest.approx(2.3, rel=0.1)

                        #test that end_time is time_passed
    # assert end_time == pytest.approx(2.3, rel=0.1)


# use QUCS to generate something 3 port with known S-params
