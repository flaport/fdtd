""" Test the FDTD detectors """

#To run tests with conda, python -m pytest
# to specify a test,  -k "test_current_detector".
# To view output, -rA


## Imports
import fdtd
import pytest
from fixtures import grid, pml, periodic_boundary, backend_parametrizer

## Tests

pytest_generate_tests = backend_parametrizer # perform tests over all backends

# @pytest.mark.skipif(skip_test_backend, reason="PyTorch is not installed.")
def test_current_detector_all_bends(grid, periodic_boundary, backends):
    fdtd.set_backend(backends)


    # curl_H(H) == pytest.approx(curl)

def test_current_detector(grid, periodic_boundary):
    pass
    # for backend in backends:
    #     fdtd.set_backend(backends)

    # curl_H(H) == pytest.approx(curl)
