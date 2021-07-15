""" Test the FFT system """

## Imports
import fdtd
import pytest
from fixtures import grid, pml, periodic_boundary




## Tests

# [Samaras 2004]
# patch_width =  59.4e-3
# patch_length = 40.4e-3
# feed_length = 0e-3
# substrate_thickness = 1.2e-3
# dielectric_constant = 2.42
#
#
# def create_patch_antenna(pcb, patch_width, patch_length):
#     MICROSTRIP_FEED_WIDTH = 3e-3
#     MICROSTRIP_FEED_LENGTH = 5e-3
#
#     z_slice = slice(pcb.component_plane_z,(pcb.component_plane_z+1))
#
#     #wipe copper
#     pcb.copper_mask[:, :, z_slice] = 0
#     pcb.copper_mask[:,:,pcb.ground_plane_z_top:pcb.component_plane_z] = 0 # vias
#
#     #rectangle
#     p_N_x = int(patch_width / pcb.cell_size)
#     p_N_y = int(patch_length / pcb.cell_size)
#     pcb.copper_mask[pcb.xy_margin:pcb.xy_margin+p_N_x, pcb.xy_margin:pcb.xy_margin+p_N_y, z_slice] = 1
#
#     # #feedport
#     # fp_N_x = int(MICROSTRIP_FEED_WIDTH/pcb.cell_size)
#     # fp_N_y = int(MICROSTRIP_FEED_LENGTH/pcb.cell_size)
#     # pcb.copper_mask[pcb.xy_margin+(p_N_x//2 - (fp_N_x//2)):pcb.xy_margin+(p_N_x//2 + (fp_N_x//2)),  \
#     #                                     pcb.xy_margin+p_N_y:pcb.xy_margin+p_N_y+fp_N_y, z_slice] = 1
#
#
#     probe_position = (p_N_y-1)
#
#     pcb.component_ports = [] # wipe ports
#     pcb.component_ports.append(fd.Port(pcb, 0, ((p_N_x//2)-1)*pcb.cell_size, probe_position*pcb.cell_size))
#


                                        # use QUCS to generate something 3 port with known S-params
