
# The following cases should be supported:
# - A single arbitrary numpy array from anywhere
# - A single detector
# - The ratio between two detectors?
#

class FrequencyDomain:
    # give it an id or name of a detector or SoftArbitraryPointSource
    # or an arbitrary numpy list,
    # converts it into the frequency domain.
    # frequency in Hz, not angular frequency
    # checks the class type
    # pad to get frequency bins



    def __init__(grid, input_name=None, input_id=None):
        self.grid = grid


    # def S_parameters():
    #     # https://inst.eecs.berkeley.edu/~ee232/sp17/discussions/Discussion%207%20-%20Photonic%20circuit%20simulation.pptx

    def pad_appropriately(fft_num_bins=None,fft_bin_resolution=None,end_time=grid.time_passed):
        '''

        - fft_num_bins
        - FFT bin resolution (optional) Hz.
        '''
        # Paraphrasing from the great explanation from
        # https://www.bitweenie.com/listings/fft-zero-padding/
        # there are two different frequency resolutions at play.
        # The first one, a physical resolution,
        # is


        if(fft_num_bins == None):
            fft_bin_resolution

        fft_num_bins
        waveform_frequency_resolution = 1.0 / end_time
        fft_bin_resolution = (1.0 / grid.time_step) / ()
        print("Waveform data has an intrinsic resolution of {:.2f}"\
                    .format(waveform_frequency_resolution))
        print("Waveform data has an intrinsic resolution of {:.2f}"\
                    .format(waveform_frequency_resolution))
        # There are some
        # the key is that indeed no extra information is being added;
        # sinc interplolation
        # https://dsp.stackexchange.com/questions/31783/mathematical-justification-for-zero-padding


        # https://dsp.stackexchange.com/questions/741/why-should-i-zero-pad-a-signal-before-taking-the-fourier-transform
        # https://dsp.stackexchange.com/questions/24410/zero-padding-of-fft/24426#24426

        # This seems like magic - how could this be?
        # there are other ways to get a higher bin resolution

    def complex_impedance():
        pass

    def plot_impedance(begin_end_freq_tuple=None):
        '''
        Frequencies are in Hz.
        '''

        begin_freq, end_freq = begin_end_freq_tuple
        desired_binning_Npoints = 300 #100 points below F_max
        required_length = int(desired_binning_Npoints / ((begin_freq-end_freq) * grid.time_step))
        print(required_length)

        voltages = bd.pad(voltages, (0, required_length), 'edge')
        currents = bd.pad(currents, (0, required_length), 'edge')

        # assumes a uniform timestep. It might be useful to add a .times vector to the grid
        # if the timestep is made variable at some point.
        # on the other hand, doing a non-uniform FFT is probably non-trivial at this point anyway
        times_padded = np.linspace(grid.time_passed)

        voltage_spectrum = bd.fft(voltages)
        current_spectrum = bd.fft(currents)

        spectrum_freqs = bd.fftfreq(len(voltages), d=pcb.grid.time_step)

        begin_freq = np.abs(spectrum_freqs - begin_freq).argmin()
        end_freq = np.abs(spectrum_freqs - end_freq).argmin()

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
