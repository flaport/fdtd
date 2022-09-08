
import pytest

import fdtd
from fdtd.backend import backend as bd
import numpy as np
from fdtd.waveforms import normalized_gaussian_pulse
# Thought for sure these wouldn't need testing, but nooo
def test_waveforms_all_bends(backends):
    fdtd.set_backend(backends)
    times = np.arange(0,1000) / 1000.0
    waveform_array = normalized_gaussian_pulse(times, 0.01,center =0.5)
    assert np.max(waveform_array) == pytest.approx(1.0, rel=0.1)
    assert waveform_array[0] == pytest.approx(0.0, rel=1e-6)
    assert waveform_array[-1] == pytest.approx(0.0, rel=1e-6)
