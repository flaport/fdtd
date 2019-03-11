## Imports

import time
import numpy as np
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

import fdtd
import fdtd.backend as bd


## Set Backend

fdtd.set_backend("numpy")


## Constants
WAVELENGTH = 1550e-9
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


## Simulation

# create FDTD Grid
grid = fdtd.Grid(
    (2.5e-5, 1.5e-5, 1),
    grid_spacing=0.1 * WAVELENGTH,
    permittivity=1.0,
    permeability=1.0,
)

# sources
grid[7.5e-6:8.0e-6, 11.8e-6:13.0e-6, 0] = fdtd.LineSource(
    period=WAVELENGTH / SPEED_LIGHT, name="source"
)

# detectors
grid[12e-6, :, 0] = fdtd.Detector(name="detector")

# x boundaries
# grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# y boundaries
# grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

# z boundaries
grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")

# objects
grid[11:32, 30:84, 0:1] = fdtd.AnisotropicObject(permittivity=2.5, name="object")

print(grid)
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
if True:
    fig, axes = plt.subplots(2, 3, squeeze=False)
    titles = ["Ex: xy", "Ey: xy", "Ez: xy", "Hx: xy", "Hy: xy", "Hz: xy"]

    fields = bd.stack(
        [
            grid.E[:, :, 0, 0],
            grid.E[:, :, 0, 1],
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


# Visualize Grid
if True:
    grid.visualize()
