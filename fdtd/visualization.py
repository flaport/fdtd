""" visualization methods for the fdtd Grid.

This module supplies visualization methods for the FDTD Grid. They are
imported by the Grid class and hence are available as Grid methods.

"""

## Imports
import os

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from matplotlib.colors import LogNorm

# 3rd party
from tqdm import tqdm
from numpy import log10, where
from scipy.signal import hilbert  # TODO: Write hilbert function to replace using scipy

# relative
from .backend import backend as bd


# 2D visualization function


def visualize(
    grid,
    x=None,
    y=None,
    z=None,
    cmap="Blues",
    pbcolor="C3",
    pmlcolor=(0, 0, 0, 0.1),
    objcolor=(1, 0, 0, 0.1),
    srccolor="C0",
    detcolor="C2",
    norm="linear",
    show=False,  # default False to allow animate to be true
    animate=False,  # True to see frame by frame states of grid while running simulation
    index=None,  # index for each frame of animation (visualize fn runs in a loop, loop variable is passed as index)
    save=False,  # True to save frames (requires parameters index, folder)
    folder=None,  # folder path to save frames
):
    """visualize a projection of the grid and the optical energy inside the grid

    Args:
        x: the x-value to make the yz-projection (leave None if using different projection)
        y: the y-value to make the zx-projection (leave None if using different projection)
        z: the z-value to make the xy-projection (leave None if using different projection)
        cmap: the colormap to visualize the energy in the grid
        pbcolor: the color to visualize the periodic boundaries
        pmlcolor: the color to visualize the PML
        objcolor: the color to visualize the objects in the grid
        srccolor: the color to visualize the sources in the grid
        detcolor: the color to visualize the detectors in the grid
        norm: how to normalize the grid_energy color map ('linear' or 'log').
        show: call pyplot.show() at the end of the function
        animate: see frame by frame state of grid during simulation
        index: index for each frame of animation (typically a loop variable is passed)
        save: save frames in a folder
        folder: path to folder to save frames
    """
    if norm not in ("linear", "lin", "log"):
        raise ValueError("Color map normalization should be 'linear' or 'log'.")
    # imports (placed here to circumvent circular imports)
    from .sources import PointSource, LineSource, PlaneSource
    from .boundaries import _PeriodicBoundaryX, _PeriodicBoundaryY, _PeriodicBoundaryZ
    from .boundaries import (
        _PMLXlow,
        _PMLXhigh,
        _PMLYlow,
        _PMLYhigh,
        _PMLZlow,
        _PMLZhigh,
    )

    if animate:  # pause for 0.1s, clear plot
        plt.pause(0.02)
        plt.clf()
        plt.ion()  # ionteration on for animation effect

    # validate x, y and z
    if x is not None:
        if not isinstance(x, int):
            raise ValueError("the `x`-location supplied should be a single integer")
        if y is not None or z is not None:
            raise ValueError(
                "if an `x`-location is supplied, one should not supply a `y` or a `z`-location!"
            )
    elif y is not None:
        if not isinstance(y, int):
            raise ValueError("the `y`-location supplied should be a single integer")
        if z is not None or x is not None:
            raise ValueError(
                "if a `y`-location is supplied, one should not supply a `z` or a `x`-location!"
            )
    elif z is not None:
        if not isinstance(z, int):
            raise ValueError("the `z`-location supplied should be a single integer")
        if x is not None or y is not None:
            raise ValueError(
                "if a `z`-location is supplied, one should not supply a `x` or a `y`-location!"
            )
    else:
        raise ValueError(
            "at least one projection plane (x, y or z) should be supplied to visualize the grid!"
        )

    # just to create the right legend entries:
    plt.plot([], lw=7, color=objcolor, label="Objects")
    plt.plot([], lw=7, color=pmlcolor, label="PML")
    plt.plot([], lw=3, color=pbcolor, label="Periodic Boundaries")
    plt.plot([], lw=3, color=srccolor, label="Sources")
    plt.plot([], lw=3, color=detcolor, label="Detectors")

    # Grid energy
    grid_energy = bd.sum(grid.E ** 2 + grid.H ** 2, -1)
    if x is not None:
        assert grid.Ny > 1 and grid.Nz > 1
        xlabel, ylabel = "y", "z"
        Nx, Ny = grid.Ny, grid.Nz
        pbx, pby = _PeriodicBoundaryY, _PeriodicBoundaryZ
        pmlxl, pmlxh, pmlyl, pmlyh = _PMLYlow, _PMLYhigh, _PMLZlow, _PMLZhigh
        grid_energy = grid_energy[x, :, :]
    elif y is not None:
        assert grid.Nx > 1 and grid.Nz > 1
        xlabel, ylabel = "z", "x"
        Nx, Ny = grid.Nz, grid.Nx
        pbx, pby = _PeriodicBoundaryZ, _PeriodicBoundaryX
        pmlxl, pmlxh, pmlyl, pmlyh = _PMLZlow, _PMLZhigh, _PMLXlow, _PMLXhigh
        grid_energy = grid_energy[:, y, :].T
    elif z is not None:
        assert grid.Nx > 1 and grid.Ny > 1
        xlabel, ylabel = "x", "y"
        Nx, Ny = grid.Nx, grid.Ny
        pbx, pby = _PeriodicBoundaryX, _PeriodicBoundaryY
        pmlxl, pmlxh, pmlyl, pmlyh = _PMLXlow, _PMLXhigh, _PMLYlow, _PMLYhigh
        grid_energy = grid_energy[:, :, z]
    else:
        raise ValueError("Visualization only works for 2D grids")

    for source in grid.sources:
        if isinstance(source, LineSource):
            if x is not None:
                _x = [source.y[0], source.y[-1]]
                _y = [source.z[0], source.z[-1]]
            elif y is not None:
                _x = [source.z[0], source.z[-1]]
                _y = [source.x[0], source.x[-1]]
            elif z is not None:
                _x = [source.x[0], source.x[-1]]
                _y = [source.y[0], source.y[-1]]
            plt.plot(_y, _x, lw=3, color=srccolor)
        elif isinstance(source, PointSource):
            if x is not None:
                _x = source.y
                _y = source.z
            elif y is not None:
                _x = source.z
                _y = source.y
            elif z is not None:
                _x = source.x
                _y = source.y
            plt.plot(_y - 0.5, _x - 0.5, lw=3, marker="o", color=srccolor)
            grid_energy[_x, _y] = 0  # do not visualize energy at location of source
        elif isinstance(source, PlaneSource):
            if x is not None:
                _x = (
                    source.y
                    if source.y.stop > source.y.start + 1
                    else slice(source.y.start, source.y.start)
                )
                _y = (
                    source.z
                    if source.z.stop > source.z.start + 1
                    else slice(source.z.start, source.z.start)
                )
            elif y is not None:
                _x = (
                    source.z
                    if source.z.stop > source.z.start + 1
                    else slice(source.z.start, source.z.start)
                )
                _y = (
                    source.x
                    if source.x.stop > source.x.start + 1
                    else slice(source.x.start, source.x.start)
                )
            elif z is not None:
                _x = (
                    source.x
                    if source.x.stop > source.x.start + 1
                    else slice(source.x.start, source.x.start)
                )
                _y = (
                    source.y
                    if source.y.stop > source.y.start + 1
                    else slice(source.y.start, source.y.start)
                )
            patch = ptc.Rectangle(
                xy=(_y.start - 0.5, _x.start - 0.5),
                width=_y.stop - _y.start,
                height=_x.stop - _x.start,
                linewidth=0,
                edgecolor="none",
                facecolor=srccolor,
            )
            plt.gca().add_patch(patch)

    # Detector
    for detector in grid.detectors:
        if x is not None:
            _x = [detector.y[0], detector.y[-1]]
            _y = [detector.z[0], detector.z[-1]]
        elif y is not None:
            _x = [detector.z[0], detector.z[-1]]
            _y = [detector.x[0], detector.x[-1]]
        elif z is not None:
            _x = [detector.x[0], detector.x[-1]]
            _y = [detector.y[0], detector.y[-1]]

        if detector.__class__.__name__ == "BlockDetector":
            # BlockDetector
            plt.plot(
                [_y[0], _y[1], _y[1], _y[0], _y[0]],
                [_x[0], _x[0], _x[1], _x[1], _x[0]],
                lw=3,
                color=detcolor,
            )
        else:
            # LineDetector
            plt.plot(_y, _x, lw=3, color=detcolor)

    # Boundaries
    for boundary in grid.boundaries:
        if isinstance(boundary, pbx):
            _x = [-0.5, -0.5, float("nan"), Nx - 0.5, Nx - 0.5]
            _y = [-0.5, Ny - 0.5, float("nan"), -0.5, Ny - 0.5]
            plt.plot(_y, _x, color=pbcolor, linewidth=3)
        elif isinstance(boundary, pby):
            _x = [-0.5, Nx - 0.5, float("nan"), -0.5, Nx - 0.5]
            _y = [-0.5, -0.5, float("nan"), Ny - 0.5, Ny - 0.5]
            plt.plot(_y, _x, color=pbcolor, linewidth=3)
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

    for obj in grid.objects:
        if x is not None:
            _x = (obj.y.start, obj.y.stop)
            _y = (obj.z.start, obj.z.stop)
        elif y is not None:
            _x = (obj.z.start, obj.z.stop)
            _y = (obj.x.start, obj.x.stop)
        elif z is not None:
            _x = (obj.x.start, obj.x.stop)
            _y = (obj.y.start, obj.y.stop)

        patch = ptc.Rectangle(
            xy=(min(_y) - 0.5, min(_x) - 0.5),
            width=max(_y) - min(_y),
            height=max(_x) - min(_x),
            linewidth=0,
            edgecolor="none",
            facecolor=objcolor,
        )
        plt.gca().add_patch(patch)

    # visualize the energy in the grid
    cmap_norm = None
    if norm == "log":
        cmap_norm = LogNorm(vmin=1e-4, vmax=grid_energy.max() + 1e-4)
    plt.imshow(bd.numpy(grid_energy), cmap=cmap, interpolation="sinc", norm=cmap_norm)

    # finalize the plot
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.ylim(Nx, -1)
    plt.xlim(-1, Ny)
    plt.figlegend()
    plt.tight_layout()

    # save frame (require folder path and index)
    if save:
        plt.savefig(os.path.join(folder, f"file{str(index).zfill(4)}.png"))

    # show if not animating
    if show:
        plt.show()


def dB_map_2D(block_det=None, choose_axis=2, interpolation="spline16"):
    """
    Displays detector readings from an 'fdtd.BlockDetector' in a decibel map spanning a 2D slice region inside the BlockDetector.
    Compatible with continuous sources (not pulse).
    Currently, only x-y 2D plot slices are accepted.

    Parameter:-
        block_det (numpy array): 5 axes numpy array (timestep, row, column, height, {x, y, z} parameter) created by 'fdtd.BlockDetector'.
        (optional) choose_axis (int): Choose between {0, 1, 2} to display {x, y, z} data. Default 2 (-> z).
        (optional) interpolation (string): Preferred 'matplotlib.pyplot.imshow' interpolation. Default "spline16".
    """
    if block_det is None:
        raise ValueError(
            "Function 'dBmap' requires a detector_readings object as parameter."
        )
    if len(block_det.shape) != 5:  # BlockDetector readings object have 5 axes
        raise ValueError(
            "Function 'dBmap' requires object of readings recorded by 'fdtd.BlockDetector'."
        )

    # TODO: convert all 2D slices (y-z, x-z plots) into x-y plot data structure

    plt.ioff()
    plt.close()
    a = []  # array to store wave intensities
    for i in tqdm(range(len(block_det[0]))):
        a.append([])
        for j in range(len(block_det[0][0])):
            temp = [x[i][j][0][choose_axis] for x in block_det]
            a[i].append(max(temp) - min(temp))

    peakVal, minVal = max(map(max, a)), min(map(min, a))
    print(
        "Peak at:",
        [
            [[i, j] for j, y in enumerate(x) if y == peakVal]
            for i, x in enumerate(a)
            if peakVal in x
        ],
    )
    a = 10 * log10([[y / minVal for y in x] for x in a])

    plt.title("dB map of Electrical waves in detector region")
    plt.imshow(a, cmap="inferno", interpolation=interpolation)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("dB scale", rotation=270)
    plt.show()


def plot_detection(detector_dict=None, specific_plot=None):
    """
    1. Plots intensity readings on array of 'fdtd.LineDetector' as a function of timestep.
    2. Plots time of arrival of pulse at different LineDetector in array.
    Compatible with pulse sources.

    Args:
        detector_dict (dictionary): Dictionary of detector readings, as created by 'fdtd.Grid.save_data()'.
        (optional) specific_plot (string): Plot for a specific axis data. Choose from {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}.
    """
    if detector_dict is None:
        raise Exception(
            "Function plotDetection() requires a dictionary of detector readings as 'detector_dict' parameter."
        )
    detectorElement = 0  # cell to consider in each detectors
    maxArray = {}
    plt.ioff()
    plt.close()

    for detector in detector_dict:
        if len(detector_dict[detector].shape) != 3:
            print("Detector '{}' not LineDetector; dumped.".format(detector))
            continue
        if specific_plot is not None:
            if detector[-2] != specific_plot[0]:
                continue
        if detector[-2] == "E":
            plt.figure(0, figsize=(15, 15))
        elif detector[-2] == "H":
            plt.figure(1, figsize=(15, 15))
        for dimension in range(len(detector_dict[detector][0][0])):
            if specific_plot is not None:
                if ["x", "y", "z"].index(specific_plot[1]) != dimension:
                    continue
            # if specific_plot, subplot on 1x1, else subplot on 2x2
            plt.subplot(
                2 - int(specific_plot is not None),
                2 - int(specific_plot is not None),
                dimension + 1 if specific_plot is None else 1,
            )
            hilbertPlot = abs(
                hilbert([x[0][dimension] for x in detector_dict[detector]])
            )
            plt.plot(hilbertPlot, label=detector)
            plt.title(detector[-2] + "(" + ["x", "y", "z"][dimension] + ")")
            if detector[-2] not in maxArray:
                maxArray[detector[-2]] = {}
            if str(dimension) not in maxArray[detector[-2]]:
                maxArray[detector[-2]][str(dimension)] = []
            maxArray[detector[-2]][str(dimension)].append(
                [detector, where(hilbertPlot == max(hilbertPlot))[0][0]]
            )

    # Loop same as above, only to add axes labels
    for i in range(2):
        if specific_plot is not None:
            if ["E", "H"][i] != specific_plot[0]:
                continue
        plt.figure(i)
        for dimension in range(len(detector_dict[detector][0][0])):
            if specific_plot is not None:
                if ["x", "y", "z"].index(specific_plot[1]) != dimension:
                    continue
            plt.subplot(
                2 - int(specific_plot is not None),
                2 - int(specific_plot is not None),
                dimension + 1 if specific_plot is None else 1,
            )
            plt.xlabel("Time steps")
            plt.ylabel("Magnitude")
        plt.suptitle("Intensity profile")
    plt.legend()
    plt.show()

    for item in maxArray:
        plt.figure(figsize=(15, 15))
        for dimension in maxArray[item]:
            arrival = bd.numpy(maxArray[item][dimension])
            plt.plot(
                [int(x) for x in arrival.T[1]],
                arrival.T[0],
                label=["x", "y", "z"][int(dimension)],
            )
        plt.title(item)
        plt.xlabel("Time of arrival (time steps)")
        plt.legend()
        plt.suptitle("Time-of-arrival plot")
    plt.show()


#
# def dump_to_vtk(pcb, filename, iteration, Ex_dump=False, Ey_dump=False, Ez_dump=False, Emag_dump=True, objects_dump=True, ports_dump=True):
#     '''
#     Extension is automatically chosen, you don't need to supply it
#
#     thanks
#     https://pyscience.wordpress.com/2014/09/06/numpy-to-vtk-converting-your-numpy-arrays-to-vtk-arrays-and-files/
#     https://bitbucket.org/pauloh/pyevtk/src/default/src/hl.py
#
#     Paraview needs a threshold operation to view the objects correctly.
#
#
#     argument scaling_factor=None,True(epsilon and hbar), or Float, to accomodate reduced units
#     or maybe
#     '''
