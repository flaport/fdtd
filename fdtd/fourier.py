


class FrequencyDetector:
    # give it an id or name of a detector,
    # or an arbitrary numpy list,
    # converts it into a
    # checks the class type
    # pad to get frequency bins

    desired_binning_Npoints = 300 #100 points below F_max
    fft_F_max = 15e9
    required_length = int(desired_binning_Npoints / (fft_F_max * pcb.grid.time_step))
    print(required_length)

    voltages = np.pad(voltages, (0, required_length), 'edge')
    currents = np.pad(currents, (0, required_length), 'edge')

    times_padded = np.pad(pcb.times, (1, required_length), 'edge')

    voltage_spectrum = np.fft.fft(voltages)

    current_spectrum = np.fft.fft(currents)

    spectrum_freqs = np.fft.fftfreq(len(voltages), d=pcb.grid.time_step)

    begin_freq = np.abs(spectrum_freqs - 1e9).argmin()
    end_freq = np.abs(spectrum_freqs - 15e9).argmin()

    plt.plot(times_padded, voltages)

    plt.plot(times_padded, currents)
    plt.plot(spectrum_freqs[begin_freq:end_freq], abs(voltage_spectrum[begin_freq:end_freq]), label="volt")
    plt.plot(spectrum_freqs[begin_freq:end_freq], abs(current_spectrum[begin_freq:end_freq]), label="curr")

    # power_spectrum = -1.0*((voltage_spectrum[begin_freq:end_freq]*np.conj(current_spectrum[begin_freq:end_freq])).real)
    # power_spectrum /= np.linalg.norm(power_spectrum)
    # plt.plot(spectrum_freqs[begin_freq:end_freq],power_spectrum)

    plt.figure()
    #
    # Z0 = scipy.constants.physical_constants['characteristic impedance of vacuum'][0]

    impedance_spectrum = abs(voltage_spectrum/current_spectrum)

    plt.plot(spectrum_freqs[begin_freq:end_freq],impedance_spectrum[begin_freq:end_freq])
    # plt.plot(spectrum_freqs[begin_freq:end_freq],impedance_spectrum[begin_freq:end_freq])
    plt.savefig('/tmp/impedance_spectrum.svg')
    # # plt.plot(spectrum_freqs,(voltage_spectrum/current_spectrum))
    # plt.plot(spectrum_freqs[begin_freq:end_freq],(voltage_spectrum[begin_freq:end_freq]/current_spectrum[begin_freq:end_freq]).real)
    # plt.plot(spectrum_freqs[begin_freq:end_freq],(voltage_spectrum[begin_freq:end_freq]/current_spectrum[begin_freq:end_freq]).imag)

    plt.draw()
    plt.pause(0.001)
    input()
    #
    # print("Z: ", impedance_spectrum[np.abs(spectrum_freqs - 2.3e9).argmin()])
    #
    # files = ['/tmp/voltages.svg', '/tmp/currents.svg', '/tmp/spectrum.svg', '/tmp/impedance_spectrum.svg', "U_patch_antenna_designer.py", "stdout.log"]
    # store.ask(files)
