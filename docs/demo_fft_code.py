


def get_current(self, pcb):
    #really needs to be fixed!

    #[Luebbers 1996 1992]

    z_slice = slice(pcb.component_plane_z-1,pcb.component_plane_z)

    current = (((pcb.grid.H[self.N_x,self.N_y-1,z_slice,X]/sqrt(mu_0))-
                (pcb.grid.H[self.N_x,self.N_y,z_slice,X]/sqrt(mu_0)))*pcb.cell_size)
    current += (((pcb.grid.H[self.N_x,self.N_y,z_slice,Y]/sqrt(mu_0))-
                (pcb.grid.H[self.N_x-1,self.N_y,z_slice,Y]/sqrt(mu_0)))*pcb.cell_size)

    current = float(current.cpu())
    # current /= (pcb.cell_size)

    #field normalized according to Flaport's thesis, chapter 4.1.6

    # account for Yee cell inaccuracies [Fang 1994].
    z_slice_2 = slice(pcb.component_plane_z-2,pcb.component_plane_z-1)

    current_2 = (((pcb.grid.H[self.N_x,self.N_y-1,z_slice_2,X]/sqrt(mu_0))-
                (pcb.grid.H[self.N_x,self.N_y,z_slice_2,X]/sqrt(mu_0)))*pcb.cell_size)
    current_2 += (((pcb.grid.H[self.N_x,self.N_y,z_slice_2,Y]/sqrt(mu_0))-
                (pcb.grid.H[self.N_x-1,self.N_y,z_slice_2,Y]/sqrt(mu_0)))*pcb.cell_size)
    # current
    current_2 = float(current_2.cpu())
    # current_2 /= (pcb.cell_size)


    current = ((current+current_2) / 2.0)

    return current

def set_voltage(self, pcb, voltage):
    z_slice = slice(pcb.component_plane_z-1,pcb.component_plane_z)

    pcb.grid.E[self.N_x,self.N_y,z_slice,Z] = sqrt(epsilon_0) * (voltage / (pcb.cell_size))


def get_voltage(self, pcb):
    z_slice = slice(pcb.component_plane_z-1,pcb.component_plane_z)

    return (pcb.grid.E[self.N_x,self.N_y,z_slice,Z]/sqrt(epsilon_0))*(pcb.cell_size)



    pcb.grid.update_E()

    pcb.grid.E[pcb.copper_mask] = 0


    source_voltage = normalized_gaussian_derivative_pulse(pcb,0.1e-9)
    # source_voltage = gaussian_derivative_pulse(pcb, 4e-12, 32)/(26.804e9)

    # source_voltage = (pcb.time*f)/((pcb.time*f)+1) # smooth ramp

    # source_voltage = sin(pcb.time * 2.0 * pi * f)
    # print(source_voltage)

    z_slice = slice(pcb.component_plane_z-1,pcb.component_plane_z)

    current = pcb.component_ports[0].get_current(pcb)
    #[Luebbers 1996]

    source_resistive_voltage = (50.0 * current)

    pcb.component_ports[0].set_voltage(pcb, source_voltage + source_resistive_voltage)

    port_2_voltage = (pcb.component_ports[1].get_current(pcb)*50.0)
    pcb.component_ports[1].set_voltage(pcb, port_2_voltage)


    print(pcb.component_ports[0].get_current(pcb))
    print(pcb.component_ports[1].get_current(pcb))

    # print(pcb.component_ports[1].get_current(pcb))
    # # print(port_2_voltage)
    print(pcb.component_ports[0].get_voltage(pcb))
    print(pcb.component_ports[1].get_voltage(pcb))

    voltages = np.append(voltages, source_voltage)
    currents = np.append(currents, current)

    print(pcb.time)
    #
    if((dump_step and abs(pcb.time-prev_dump_time) > dump_step) or pcb.grid.time_steps_passed == 0):
        #paraview gets confused if the first number isn't zero.
        fd.dump_to_vtk(pcb,'dumps/test',pcb.grid.time_steps_passed)
        prev_dump_time = pcb.time



    pcb.grid.update_H()

    pcb.grid.time_steps_passed += 1
    pcb.time += pcb.grid.time_step # the adaptive
    pcb.times.append(pcb.time)
