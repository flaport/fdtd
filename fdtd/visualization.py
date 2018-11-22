## Imports

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

# relative
from .backend import backend as bd


# 2D visualization function

def visualize(grid, cmap='Blues', pbcolor='red', pmlcolor=(0,0,0,0.1)):
    """ visualize the grid and the optical energy inside the grid

    Args:
        grid: Grid: the grid instance to visualize

    Kwargs:
        cmap='Blues': the colormap to visualize the energy in the grid
        pbcolor='red': the color to show the periodic boundaries
        pmlcolor=(0,0,0,0.1): the color to show the PML
    """
    # imports (placed here to circumvent circular imports)
    from .boundaries import PeriodicBoundaryX, PeriodicBoundaryY, PeriodicBoundaryZ
    from .boundaries import PMLXlow, PMLXhigh, PMLYlow, PMLYhigh, PMLZlow, PMLZhigh

    # just to create the right legend entries:
    plt.plot([], lw=5, color=pmlcolor, label='PML')
    plt.plot([], lw=3, color=pbcolor, label='Periodic Boundary')

    # Grid energy
    grid_energy = bd.sum(grid.E**2 + grid.H**2, -1)
    if grid.Nx == 1:
        assert grid.Ny > 1 and grid.Nz > 1
        xlabel, ylabel = 'y', 'z'
        Nx, Ny = grid.Ny, grid.Nz
        pbx, pby = PeriodicBoundaryY, PeriodicBoundaryZ
        pmlxl, pmlxh, pmlyl, pmlyh = PMLYlow, PMLYhigh, PMLZlow, PMLZhigh
        grid_energy = grid_energy[0,:,:]
    elif grid.Ny == 1:
        assert grid.Nx > 1 and grid.Nz > 1
        xlabel, ylabel = 'z', 'x'
        Nx, Ny = grid.Nz, grid.Nx
        pbx, pby = PeriodicBoundaryZ, PeriodicBoundaryX
        pmlxl, pmlxh, pmlyl, pmlyh = PMLZlow, PMLZhigh, PMLXlow, PMLXhigh
        grid_energy = grid_energy[:,0,:].T
    elif grid.Nz == 1:
        assert grid.Nx > 1 and grid.Ny > 1
        xlabel, ylabel = 'x', 'y'
        Nx, Ny = grid.Nx, grid.Ny
        pbx, pby = PeriodicBoundaryX, PeriodicBoundaryY
        pmlxl, pmlxh, pmlyl, pmlyh = PMLXlow, PMLXhigh, PMLYlow, PMLYhigh
        grid_energy = grid_energy[:,:,0]
    else:
        raise ValueError("Visualization only works for 2D grids")
    plt.imshow(bd.numpy(grid_energy), cmap=cmap)

    # Sources
    plt.plot([]) # cycle to C1
    for name, source in grid._sources.items():
        if grid.Nx == 1:
            x = [source.y[0], source.y[-1]]
            y = [source.z[0], source.z[-1]]
        elif grid.Ny == 1:
            x = [source.z[0], source.z[-1]]
            y = [source.x[0], source.x[-1]]
        elif grid.Nz == 1:
            x = [source.x[0], source.x[-1]]
            y = [source.y[0], source.y[-1]]

        plt.plot(y, x, lw=4, label=name)

    # Boundaries
    for name, boundary in grid._boundaries.items():
        if isinstance(boundary, pbx):
            x = [-.5, -.5, float("nan"), Nx-.5, Nx-.5]
            y = [-.5, Ny-.5, float("nan"), -.5, Ny-.5]
            plt.plot(y, x, color=pbcolor, linewidth=3)
        elif isinstance(boundary, pby):
            x = [-.5, Nx-.5, float("nan"), -.5, Nx-.5]
            y = [-.5, -.5, float("nan"), Ny-.5, Ny-.5]
            plt.plot(y, x, color=pbcolor, linewidth=3)
        elif isinstance(boundary, pmlyl):
            patch = ptc.Rectangle(
                xy=(-0.5,-0.5),
                width=boundary.thickness,
                height=Nx,
                linewidth=0,
                edgecolor='none',
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)
        elif isinstance(boundary, pmlxl):
            patch = ptc.Rectangle(
                xy=(-0.5,-0.5),
                width=Ny,
                height=boundary.thickness,
                linewidth=0,
                edgecolor='none',
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)
        elif isinstance(boundary, pmlyh):
            patch = ptc.Rectangle(
                xy=(Ny-0.5-boundary.thickness,-0.5),
                width=boundary.thickness,
                height=Nx,
                linewidth=0,
                edgecolor='none',
                facecolor=pmlcolor,
            )
            plt.gca().add_patch(patch)
        elif isinstance(boundary, pmlxh):
            patch = ptc.Rectangle(
                xy=(-.5,Nx-boundary.thickness-0.5),
                width=Ny,
                height=boundary.thickness,
                linewidth=0,
                edgecolor='none',
                facecolor=pmlcolor,
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

