## Imports

import time
import numpy as np
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

import fdtd
import fdtd.backend as bd


## Set Backend

fdtd.set_backend("torch.cuda")


## Simulation

# create FDTD Grid
M = N = 100
P = 1

grid = fdtd.Grid((M, N, P))
grid.source = fdtd.Source(p0=(48, 80, 0), p1=(52, 80, 0), period=20)
grid.detector = fdtd.Detector(x=[50], y=slice(None), z=[0])
grid.xbounds = fdtd.PeriodicBoundaryX()
grid.ybounds = fdtd.PeriodicBoundaryY()
grid.zbounds = fdtd.PeriodicBoundaryZ()
grid.pml = fdtd.PMLYhigh(thickness=10)
grid.pml2 = fdtd.PMLYlow(thickness=10)


print(f"courant number: {grid.courant_number}")

# create and enable profiler
profiler = LineProfiler()
profiler.add_function(grid.update_E)
profiler.enable()

# run simulation
grid.run(50, progress_bar=False)


# print profiler summary
profiler.print_stats()


## Plots

# Fields
fig, axes = plt.subplots(3, 2, squeeze=False)
titles = ["Ex: xy", "Ey: xy", "Ez: xy", "Hx: xy", "Hy: xy", "Hz: xy"]

fields = bd.stack(
    [
        grid.E[:, :, 0, 1],
        grid.E[:, :, 0, 0],
        grid.E[:, :, 0, 2],
        grid.H[:, :, 0, 0],
        grid.H[:, :, 0, 1],
        grid.H[:, :, 0, 2],
    ]
)

m = max(abs(fields.min().item()), abs(fields.max().item()))

for ax, field, title in zip(axes.ravel(), fields, titles):
    ax.set_axis_off()
    ax.set_title(title)
    ax.imshow(bd.numpy(field), vmin=-m, vmax=m, cmap="RdBu")

plt.show()

# Detected
Ez = bd.squeeze(bd.stack(grid.detector.E, 0)[...,2])
plt.imshow(bd.numpy(Ez), cmap="RdBu")
plt.show()

