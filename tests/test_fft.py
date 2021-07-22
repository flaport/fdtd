""" Test the FFT system """

## Imports
import fdtd
import pytest
from fixtures import grid, pml, periodic_boundary

from fdtd.fourier import FrequencyRoutines
from fdtd.backend import backend as bd
from fdtd.detectors import LineDetector, BlockDetector, CurrentDetector
from fdtd.sources import SoftArbitraryPointSource
import numpy as np

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


def test_FFT_single_detector_empty_time(grid):
    detector = BlockDetector()
    grid[4,4,4] = detector
    fr = FrequencyRoutines(grid, detector)
    fr.FFT()

def test_FFT_single_detector(grid):
    detector = BlockDetector()
    grid[4,4,4] = detector
    grid.run(100)
    # detector.E[:][0][0][0] = np.sin().tolist()
    # tolist required for accurate test of existing multiple-dimension
    fr = FrequencyRoutines(grid, detector)
    spectrum_freqs, spectrum = fr.FFT()


def test_SAPS_impedance(grid):
    detector = SoftArbitraryPointSource(np.zeros(1),impedance = 50.0)
    grid[4,4,4] = detector
    grid.run(100)
    fr = FrequencyRoutines(grid, detector)
    fr.impedance()



# use QUCS to generate something 3 port with known S-params
