## Imports

import time
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

import fdtd


## Simulation

# create FDTD Grid
grid = fdtd.Grid((400, 400))

# create and enable profiler
profiler = LineProfiler()
profiler.add_function(grid.update_E)
profiler.enable()

# run simulation
grid.run(20, progress_bar=False)

# print profiler summary
profiler.print_stats()

# show result
plt.imshow(grid.E[..., 2])
plt.show()
