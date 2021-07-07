


def set_voltage(self, pcb, voltage):
    z_slice = slice(pcb.component_plane_z-1,pcb.component_plane_z)

    pcb.grid.E[self.N_x,self.N_y,z_slice,Z] = sqrt(epsilon_0) * (voltage / (pcb.cell_size))


def get_voltage(self, pcb):
    z_slice = slice(pcb.component_plane_z-1,pcb.component_plane_z)

    return (pcb.grid.E[self.N_x,self.N_y,z_slice,Z]/sqrt(epsilon_0))*(pcb.cell_size)



    pcb.grid.update_E()

    pcb.grid.E[pcb.copper_mask] = 0


    source_voltage = normalized_gaussian_derivative_pulse(pcb,0.1e-9)



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
