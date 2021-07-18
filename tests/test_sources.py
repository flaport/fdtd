""" Test the FDTD sources """


## Imports
import fdtd
import pytest

from fixtures import grid, pml, periodic_boundary, backend_parametrizer
from sources import SoftArbitraryPointSource
## Tests


def test_SoftArbitraryPointSource(grid):
    j
