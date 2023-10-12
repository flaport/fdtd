import sys
sys.path.append('/home/arend/work/vscode/fdtd')
import sys
print(sys.path)


import numpy as np
import pytest
from fdtd.lgrid import LGrid

def test_update_C():
    # create a 3x3x3 Yee grid
    grid = LGrid((3, 3, 3), (1, 1, 1), 0.5)
    
    # set the courant number
    grid.courant_number = 0.5
    
    # set the C field to some initial values
    grid.C = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
    ])
    
    # call the update_C function
    grid.update_C()
    
    # check that the C field has been updated correctly
    assert np.allclose(grid.C, np.array([
        [[-0.125, -0.25, -0.375], [-0.5, -0.625, -0.75], [-0.875, -1.0, -1.125]],
        [[-1.25, -1.375, -1.5], [-1.625, -1.75, -1.875], [-2.0, -2.125, -2.25]],
        [[-2.375, -2.5, -2.625], [-2.75, -2.875, -3.0], [-3.125, -3.25, -3.375]]
    ]))
    
    
def main():
    test_update_C()

if __name__ == '__main__':
    main()
