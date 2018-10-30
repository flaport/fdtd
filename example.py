## Imports

import time
import numpy as np
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

import fdtd
from backend import backend as bd


## Set Backend

fdtd.set_backend("torch")


## Simulation

# create FDTD Grid
N = 7
n = N // 2
grid = fdtd.Grid((N, N, N))
print(f"courant number: {grid.courant_number}")

# create and enable profiler
profiler = LineProfiler()
profiler.add_function(grid.update_E)
profiler.enable()

# run simulation
grid.run(N, progress_bar=False)

# print profiler summary
profiler.print_stats()


## Plot Result

fig, axes = plt.subplots(3, 6, squeeze=False)
titles = [
    "Ex: yz", "Ey: zx", "Ez: xy", "Hx: yz", "Hy: zx", "Hz: xy",
    "Ex: zx", "Ey: xy", "Ez: yz", "Hx: zx", "Hy: xy", "Hz: yz",
    "Ex: xy", "Ey: yz", "Ez: zx", "Hx: xy", "Hy: yz", "Hz: zx",
]
fields = bd.stack([
    grid.E[n, :, :, 0], bd.transpose(grid.E[:, n, :, 1]), grid.E[:, :, n, 2],
    grid.H[n, :, :, 0], bd.transpose(grid.H[:, n, :, 1]), grid.H[:, :, n, 2],
    bd.transpose(grid.E[:, n, :, 0]), grid.E[:, :, n, 1], grid.E[n, :, :, 2],
    bd.transpose(grid.H[:, n, :, 0]), grid.H[:, :, n, 1], grid.H[n, :, :, 2],
    grid.E[:, :, n, 0], grid.E[n, :, :, 1], bd.transpose(grid.E[:, n, :, 2]),
    grid.H[:, :, n, 0], grid.H[n, :, :, 1], bd.transpose(grid.H[:, n, :, 2]),
])

m = max(abs(fields.min().item()), abs(fields.max().item()))

for ax, field, title in zip(axes.ravel(), fields, titles):
    ax.set_axis_off()
    ax.set_title(title)
    ax.imshow(field, vmin=-m, vmax=m, cmap='RdBu')

plt.show()
