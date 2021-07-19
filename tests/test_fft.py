""" Test the FFT system """

## Imports
import fdtd
import pytest
from fixtures import grid, pml, periodic_boundary

from fdtd.fourier import FrequencyRoutines
from fdtd.backend import backend as bd
from fdtd.detectors import LineDetector, BlockDetector, CurrentDetector



## Tests


def test_padding(grid):
    fr = FrequencyRoutines(grid, None)
    input_data = bd.zeros((1000))
    dt = 1e-9
    #sample rate 1 GHz
    #1 microsecond capture length
    #total length
    # freq_window = ()


    required_padding, end_time = \
            fr.compute_padding(input_data, dt=dt, freq_window_tuple=None, fft_num_bins_in_window=None,
                                        fft_bin_freq_resolution=None, )

    # assert end_time == pytest.approx(2.3, rel=0.1)

                        #test that end_time is time_passed
    # assert end_time == pytest.approx(2.3, rel=0.1)


def test_FFT_single_detector_empty_grid(grid):
    detector = BlockDetector()
    grid[4,4,4] = detector
    fr = FrequencyRoutines(grid, detector)
    fr.FFT()

# use QUCS to generate something 3 port with known S-params
