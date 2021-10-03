""" Test the FFT system """

## Imports
import fdtd
import pytest
from fixtures import grid, pml, periodic_boundary
from fdtd.boundaries import DomainBorderPML
from fdtd.objects import Object, AbsorbingObject
from fdtd.waveforms import normalized_gaussian_pulse
from fdtd.sources import SoftArbitraryPointSource
from fdtd.fourier import FrequencyRoutines
from fdtd.backend import backend as bd
import numpy as np
import matplotlib.pyplot as plt


def create_patch_antenna(normalized_probe_position):
    # [Samaras 2004]
    # Probe fed microstrip antenna
    # probe in middle and translated along X
    # designed f = 2.3 GHz,
    # excited with gaussian pulse
    # centered at 2.3 GHz, harmonics up to 4 GHz
    # 6-layer thick PML
    # A non-uniform grid was used in the Samaras' model.
    # The maximum cell size was 2 mm.
    # other tests should be physically smaller if possible;
    # performance limitation here seems to be
    # the discretization size, not the timestep
    border_cells = 6
    patch_width =  59.4e-3 # m
    patch_length = 40.4e-3 # m
    substrate_thickness = 1.27e-3 # m
    dielectric_constant = 2.42 # m
    loss_tangent = 0.0019 #



    N_substrate_cells = 3
    spacing = substrate_thickness / N_substrate_cells

    # 1 / (((1.27e-3 / 10)  meters) / c) ≈ 100x f, A-ok

    Nx = int(patch_width/spacing)+(2*border_cells)
    Ny = int(patch_length/spacing)+(2*border_cells)
    Nz = int(substrate_thickness/spacing)+(2*border_cells) + 2 # finite thickness of copper planes

    grid = fdtd.Grid(
        shape=(Nx, Ny, Nz),
        grid_spacing=spacing,
        permittivity=1.0,
        permeability=1.0,
        courant_number=None,  # calculate the courant number
    )


    pulse_fwhm = 2.3e-9

    simulation_steps = int(5 * (pulse_fwhm / grid.time_step))
    pulse_center_time = (simulation_steps * grid.time_step) / 5.0

    DomainBorderPML(grid, border_cells=border_cells)

    sl = slice(border_cells, -border_cells)

    copper_object = AbsorbingObject(permittivity=1.0, conductivity=1e8)


    grid[sl, sl, border_cells:border_cells+1] = copper_object # ground plane

    bottom_copper_plane = border_cells
    top_copper_plane = (border_cells+1+N_substrate_cells)

    grid[sl, sl, top_copper_plane:top_copper_plane+1] = copper_object #top plane

    probe_Nx = int(patch_width/spacing)//2 + border_cells
    probe_Ny = int(normalized_probe_position*(int(patch_length/spacing)//2)) + border_cells

    times = np.arange(0,simulation_steps) * grid.time_step

    waveform_array = normalized_gaussian_pulse(times, pulse_fwhm, center=pulse_center_time)

    source = SoftArbitraryPointSource(waveform_array=waveform_array, impedance=50.0)

    grid[probe_Nx,probe_Ny,border_cells+1] = source

    grid[probe_Nx:probe_Nx+1, probe_Ny:probe_Ny+1,
                        border_cells+2:top_copper_plane+1] = copper_object # probe feed via

    substrate = Object(permittivity=dielectric_constant)
    grid[border_cells:probe_Nx, sl, bottom_copper_plane+1:top_copper_plane] = substrate
    grid[probe_Nx+1:-border_cells, sl, bottom_copper_plane+1:top_copper_plane] = substrate

    grid[sl, border_cells:probe_Ny, bottom_copper_plane+1:top_copper_plane] = substrate
    grid[sl, probe_Ny+1:-border_cells, bottom_copper_plane+1:top_copper_plane] = substrate
    # don't overlap with the source port!

    return grid, source, simulation_steps
    #


# use pytest -s here to see live output
@pytest.mark.slow
def test_antenna_impedance():
    fdtd.set_backend("torch.cuda.float32")
    # a very slow test
    distance_from_edge = bd.array([0.0,0.25,0.35,0.50,0.67,1.00])
    samaras_FDTD_R = bd.array([169,136,106,72,32,0])

    grid, source, simulation_steps = create_patch_antenna(0)
    for idx, d in enumerate(distance_from_edge):
        grid, source, _ = create_patch_antenna(d)

        grid.run(simulation_steps)

        fr = FrequencyRoutines(grid, source)
        fr.impedance()

# use QUCS to generate something 3 port with known S-params

#ask floris if there're any test cases for optical
