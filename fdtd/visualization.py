""" FDTD visualization

This module supplies visualization methods for the FDTD Grid. They are
imported by the Grid class and hence are available as Grid methods.

"""

## Imports

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

# relative
from .backend import backend as bd


# 2D visualization function


def visualize(
    grid,
    cmap="Blues",
    pbcolor="C3",
    pmlcolor=(0, 0, 0, 0.1),
    objcolor=(1, 0, 0, 0.1),
    srccolor="C0",
    detcolor="C2",
):
    """ visualize the grid and the optical energy inside the grid

    Args:
        grid: Grid: the grid instance to visualize

    Kwargs:
        cmap='Blues': the colormap to visualize the energy in the grid
        pbcolor='C3': the color to visualize the periodic boundaries
        pmlcolor=(0,0,0,0.1): the color to visualize the PML
        objcolor=(1,0,0,0.1): the color to visualize the objects in the grid
        objcolor='C0': the color to visualize the sources in the grid
    """
    # imports (placed here to circumvent circular imports)
    from .boundaries import _PeriodicBoundaryX, _PeriodicBoundaryY, _PeriodicBoundaryZ
    from .boundaries import (
        _PMLXlow,
        _PMLXhigh,
        _PMLYlow,
        _PMLYhigh,
        _PMLZlow,
        _PMLZhigh,
    )

    # just to create the right legend entries:
    plt.plot([], lw=7, color=objcolor, label="Objects")
    plt.plot([], lw=7, color=pmlcolor, label="PML")
    plt.plot([], lw=3, color=pbcolor, label="Periodic Boundaries")
    plt.plot([], lw=3, color=srccolor, label="Sources")
    plt.plot([], lw=3, color=detcolor, label="Detectors")

    # Grid energy
    grid_energy = bd.sum(grid.E ** 2 + grid.H ** 2, -1)
    if grid.Nx == 1:
        assert grid.Ny > 1 and grid.Nz > 1
        xlabel, ylabel = "y", "z"
        Nx, Ny = grid.Ny, grid.Nz
        pbx, pby = _PeriodicBoundaryY, _PeriodicBoundaryZ
        pmlxl, pmlxh, pmlyl, pmlyh = _PMLYlow, _PMLYhigh, _PMLZlow, _PMLZhigh
        grid_energy = grid_energy[0, :, :]
    elif grid.Ny == 1:
        assert grid.Nx > 1 and grid.Nz > 1
        xlabel, ylabel = "z", "x"
        Nx, Ny = grid.Nz, grid.Nx
        pbx, pby = _PeriodicBoundaryZ, _PeriodicBoundaryX
        pmlxl, pmlxh, pmlyl, pmlyh = _PMLZlow, _PMLZhigh, _PMLXlow, _PMLXhigh
        grid_energy = grid_energy[:, 0, :].T
    elif grid.Nz == 1:
        assert grid.Nx > 1 and grid.Ny > 1
        xlabel, ylabel = "x", "y"
        Nx, Ny = grid.Nx, grid.Ny
        pbx, pby = _PeriodicBoundaryX, _PeriodicBoundaryY
        pmlxl, pmlxh, pmlyl, pmlyh = _PMLXlow, _PMLXhigh, _PMLYlow, _PMLYhigh
        grid_energy = grid_energy[:, :, 0]
    else:
        raise ValueError("Visualization only works for 2D grids")
    plt.imshow(bd.numpy(grid_energy), cmap=cmap)

    # Sources
    for source in grid._sources:
        if grid.Nx == 1:
            x = [source.y[0], source.y[-1]]
            y = [source.z[0], source.z[-1]]
        elif grid.Ny == 1:
            x = [source.z[0], source.z[-1]]
            y = [source.x[0], source.x[-1]]
        elif grid.Nz == 1:
            x = [source.x[0], source.x[-1]]
            y = [source.y[0], source.y[-1]]

        plt.plot(y, x, lw=3, color=srccolor)

    # Detectors
    for detector in grid._detectors:
        if grid.Nx == 1:
            x = (
                [detector.y.start, detector.y.stop]
                if isinstance(detector.y, slice)
                else [detector.y[0], detector.y[-1]]
            )
            x[0] = x[0] if x[0] is not None else 0
            x[1] = x[1] if x[1] is not None else grid.Ny
            y = (
                [detector.z.start, detector.z.stop]
                if isinstance(detector.z, slice)
                else [detector.z[0], detector.z[-1]]
            )
            y[0] = y[0] if y[0] is not None else 0
            y[1] = y[1] if y[1] is not None else grid.Nz
        elif grid.Ny == 1:
            x = (
                [detector.z.start, detector.z.stop]
                if isinstance(detector.z, slice)
                else [detector.z[0], detector.z[-1]]
            )
            x[0] = x[0] if x[0] is not None else 0
            x[1] = x[1] if x[1] is not None else grid.Nz
            y = (
                [detector.x.start, detector.x.stop]
                if isinstance(detector.x, slice)
                else [detector.x[0], detector.x[-1]]
            )
            y[0] = y[0] if y[0] is not None else 0
            y[1] = y[1] if y[1] is not None else grid.Nx
        elif grid.Nz == 1:
            x = (
                [detector.x.start, detector.x.stop]
                if isinstance(detector.x, slice)
                else [detector.x[0], detector.x[-1]]
            )
            x[0] = x[0] if x[0] is not None else 0
            x[1] = x[1] if x[1] is not None else grid.Nx
            y = (
                [detector.y.start, detector.y.stop]
                if isinstance(detector.y, slice)
                else [detector.y[0], detector.y[-1]]
            )
            y[0] = y[0] if y[0] is not None else 0
            y[1] = y[1] if y[1] is not None else grid.Ny

        plt.plot(y, x, lw=3, color=detcolor)

    # Boundaries
    for boundary in grid._boundaries:
        if isinstance(boundary, pbx):
            x = [-0.5, -0.5, float("nan"), Nx - 0.5, Nx - 0.5]
            y = [-0.5, Ny - 0.5, float("nan"), -0.5, Ny - 0.5]
            plt.plot(y, x, color=pbcolor, linewidth=3)
        elif isinstance(boundary, pby):
            x = [-0.5, Nx - 0.5, float("nan"), -0.5, Nx - 0.5]
            y = [-0.5, -0.5, float("nan"), Ny - 0.5, Ny - 0.5]
            plt.plot(y, x, color=pbcolor, linewidth=3)
        elif isinstance(boundary, pmlyl):
            patch = ptc.Rectangle(
                xy=(-0.5, -0.5),
                width=boundary.thickness,
                height=Nx,
                linewidth=0,
                edgecolor="none",
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)
        elif isinstance(boundary, pmlxl):
            patch = ptc.Rectangle(
                xy=(-0.5, -0.5),
                width=Ny,
                height=boundary.thickness,
                linewidth=0,
                edgecolor="none",
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)
        elif isinstance(boundary, pmlyh):
            patch = ptc.Rectangle(
                xy=(Ny - 0.5 - boundary.thickness, -0.5),
                width=boundary.thickness,
                height=Nx,
                linewidth=0,
                edgecolor="none",
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)
        elif isinstance(boundary, pmlxh):
            patch = ptc.Rectangle(
                xy=(-0.5, Nx - boundary.thickness - 0.5),
                width=Ny,
                height=boundary.thickness,
                linewidth=0,
                edgecolor="none",
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)

    for obj in grid._objects:
        if (xlabel, ylabel) == ("y", "z"):
            x = (obj.y.start, obj.y.stop)
            y = (obj.z.start, obj.z.stop)
        elif (xlabel, ylabel) == ("z", "x"):
            x = (obj.z.start, obj.z.stop)
            y = (obj.x.start, obj.x.stop)
        else:
            x = (obj.x.start, obj.x.stop)
            y = (obj.y.start, obj.y.stop)
        patch = ptc.Rectangle(
            xy=(min(y) - 0.5, min(x) - 0.5),
            width=max(y) - min(y),
            height=max(x) - min(x),
            linewidth=0,
            edgecolor="none",
            facecolor=objcolor,
        )
        plt.gca().add_patch(patch)

    # finalize the plot
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.ylim(Nx, -1)
    plt.xlim(-1, Ny)
    plt.figlegend()
    plt.tight_layout()
    plt.show()
