
# def pretty_freqs():
#     ["KHz", "MHz", "GHz", "THz"]
#     {:.2f}"\
#                 .format(fft_bin_resolution)

# In the electrical domain,

# https://inst.eecs.berkeley.edu/~ee232/sp17/discussions/Discussion%207%20-%20Photonic%20circuit%20simulation.pptx

# in the optical domain, the impedance would just be that of free space?

from math import ceil
from .backend import backend as bd
from .detectors import LineDetector, BlockDetector, CurrentDetector
from .sources import SoftArbitraryPointSource

class FrequencyRoutines:
    # originally called FrequencyDomain, renamed because 'fd' is already a namespace

    #inputs:
    # One of, or a list of:
    # - a single plain array or numpy array of values (like the waveform_array input to SoftArb)
    # only works with FFT(), not impedance.
    # The length does not need to correspond to the simulation length (e.g. it can be shorter);
    # The timestep is assumed to be the same, however.
    #
    # - a single plain detector, of any type,
    #
    #
    # - a single SoftArbitraryPointSource object,
    # works with impedance or 1-port s-param,
    # - an array of SoftArbitraryPointSource
    # for multi-port S-param,
    # - the tuple (detector, current_detector) or [(detector, current_detector),...]
    # (the impedance is then assumed to be that of free-space).
    # FIXME:
    # free-space impedance should also eventually
    # depend on the permittivity and permeability of the detector cell
    #
    # converts it into the frequency domain.
    # frequency in Hz, not angular frequency



    def __init__(self, grid, objs):
        self.grid = grid
        self.objs = objs



    def compute_padding(self, input_data, dt, freq_window_tuple=None, fft_num_bins_in_window=None,
                                            fft_bin_freq_resolution=None):
        '''
        input_data must be a 1d array with the time history of one detector. the length of input_data
        is expected to be the same as time_steps_passed.

        Does not apply the padding to input_data,
        since multiple vectors will probably need the same padding and it
        seems to make more sense to do that in the caller.

        - fft_num_bins
        - FFT bin resolution (optional) Hz.
        '''

        # This seems like magic - how could this be?

        # the key is that indeed no extra information is being added;
        # sinc interplolation

        # https://dsp.stackexchange.com/questions/31783/mathematical-justification-for-zero-padding
        # https://dsp.stackexchange.com/questions/741/why-should-i-zero-pad-a-signal-before-taking-the-fourier-transform
        # https://dsp.stackexchange.com/questions/24410/zero-padding-of-fft/24426#24426

        # Paraphrasing from the great explanation from
        # https://www.bitweenie.com/listings/fft-zero-padding/
        # there are two different frequency resolutions at play.
        # The first one, a physical resolution,

        # there are other ways to get a higher bin resolution without explicitly padding, too


        input_length = input_data.shape[0]

        if(freq_window_tuple == None):
            begin_freq = 0
            # from numpy fftfreq docs
            end_freq = (input_length/2.0) / (dt*input_length) # off by one?
        else:
            begin_freq, end_freq = freq_window_tuple

        end_time = input_length * dt

        if(fft_num_bins_in_window == None and not fft_bin_freq_resolution == None):
            fft_num_bins_in_window = ((begin_freq-end_freq)/fft_bin_freq_resolution)

        elif(not (fft_num_bins_in_window or fft_bin_freq_resolution)):
            # the window is the default, whole array (even if it's trimmed later)
            fft_num_bins_in_window = input_data.shape[0]

        waveform_frequency_resolution = 1.0 / end_time
        fft_bin_resolution = (1.0 / dt) / (fft_num_bins_in_window)
        print("Waveform data has an intrinsic resolution of: {:.2E} Hz"\
                    .format(waveform_frequency_resolution))
        print("FFT bin: {:.2E} Hz"\
                    .format(fft_bin_resolution))

        required_padding = ceil(fft_num_bins_in_window / ((end_freq-begin_freq) * dt)) - input_length



        return required_padding, end_time


        # assumes a uniform timestep. It might be useful to add a .times vector to the grid
        # if the timestep is made variable at some point.
        # on the other hand, doing a non-uniform FFT is probably non-trivial at this point anyway
        # times = bd.linspace(0,end_time + (required_padding*dt),
        #                                         (input_length+required_padding))


    def compute_frequencies(self,length_with_padding, dt, freq_window_tuple=None):
        '''
        Outputs the
        The indexes are for the real-frequency part of the span.
        '''

        spectrum_freqs = bd.fftfreq(length_with_padding, d=dt)

        if(freq_window_tuple == None):
            begin_freq_idx = 0
            # end_freq_idx = ((length_with_padding/2)-1)
            end_freq_idx = -1
        else:
            #closest frequencies
            begin_freq_idx = bd.abs(spectrum_freqs - begin_freq).argmin()
            end_freq_idx = bd.abs(spectrum_freqs - end_freq).argmin()

        return spectrum_freqs, begin_freq_idx, end_freq_idx

    # UNTESTED
    def S_parameters(waveform, node):
        raise NotImplementedError
        # outputs the S params for each frequency

        # [1]Kurokawa K.
        # Power Waves and the Scattering Matrix.
        # IEEE Transactions on Microwave Theory and Techniques 1965
        # ;13:194–202. https://doi.org/10.1109/TMTT.1965.1125964.

        # Need to construct the matrix from Kurokawa "power wave" coefficients:
        # a_1 =  #incident
        # b_1 =  #reflected
        # Wikipedia has a great section:
        # https://en.wikipedia.org/wiki/Scattering_parameters

        # inverse matrix?
        #
        # if(isinstance(self.objs, list)):
        #
        # elif(isinstance(self.objs,SoftArbitraryPointSource)):
        # elif(isinstance(self.objs,tuple) and isinstance(self.objs[0],(BlockDetector,LineDetector))
        #                                 and isinstance(self.objs[1],CurrentDetector)):
        # else:
        #     raise ValueError("Sorry, FFT can't yet interpret the argument given.")

    def export_touchstone_s2p():
        raise NotImplementedError



    def FFT(self, freq_window_tuple=None, fft_num_bins_in_window=None,
                                            fft_bin_freq_resolution=None):
        '''
        FIXME: If is type BlockDetector or other similar output,
        for now the fft of only one index is returned.
        '''
        #scaling factor here?

        if(self.grid.time_steps_passed == 0):
            return [], []
        #I now see how Multiple Dispatch might be effective at speeding things up
        if(bd.is_array(self.objs)):
            input_data = self.objs
        elif(isinstance(self.objs,(LineDetector, BlockDetector))):
            #FIXME:
            input_data = self.objs.E[:][0][0][0]
        elif(isinstance(self.objs,SoftArbitraryPointSource)):
            #FIXME:
            input_data = self.objs.source_voltage[:][0][0][0]
        elif(isinstance(self.objs,CurrentDetector)):
            #FIXME:
            input_data = self.objs.I[:][0][0][0]
        else:
            raise ValueError("Sorry, FFT can't yet interpret the argument given.")

        input_data = bd.array(input_data)
        required_padding, _ = self.compute_padding(input_data,
                                                self.grid.time_step,
                                                freq_window_tuple=freq_window_tuple,
                                                fft_num_bins_in_window=fft_num_bins_in_window,
                                                fft_bin_freq_resolution=fft_bin_freq_resolution)

        input_data = bd.pad(input_data, (0, required_padding), 'edge')

        spectrum = bd.fft(input_data)

        spectrum_freqs, begin_freq_idx, end_freq_idx = self.compute_frequencies(
                                        input_data.shape[0], self.grid.time_step,
                                         freq_window_tuple=freq_window_tuple)

        return spectrum_freqs[begin_freq_idx:end_freq_idx], spectrum[begin_freq_idx:end_freq_idx]



    def impedance(self, freq_window_tuple=None, fft_num_bins_in_window=None,
                                            fft_bin_freq_resolution=None):
        '''
        Takes a single "node", a current and voltage record. The form of:
        SoftArbitraryPointSource or tuple.
        '''
        if(self.grid.time_steps_passed == 0):
            return [], []

        if(isinstance(self.objs, SoftArbitraryPointSource)):
            voltage_array = bd.array(self.objs.source_voltage)
            current_array = bd.array(self.objs.current_detector.I) #FIXME: this definitely isn't right

        elif(isinstance(self.objs,tuple) and isinstance(self.objs[0],(BlockDetector,LineDetector))
                                        and isinstance(self.objs[1],CurrentDetector)):
            #FIXME:
            voltage_array = bd.array(self.objs[0].E)[:][0,0,0]
            current_array = bd.array(self.objs[1].I)
        else:
            raise ValueError("Sorry, FFT can't yet interpret the argument given.")

        required_padding, _ = self.compute_padding(voltage_array,
                                                self.grid.time_step,
                                                freq_window_tuple=freq_window_tuple,
                                                fft_num_bins_in_window=fft_num_bins_in_window,
                                                fft_bin_freq_resolution=fft_bin_freq_resolution)

        voltage_array = bd.reshape(voltage_array, self.grid.time_steps_passed) # flatten
        current_array = bd.reshape(current_array, self.grid.time_steps_passed)

        voltage_array = bd.pad(voltage_array, (0, required_padding), 'edge')
        current_array = bd.pad(current_array, (0, required_padding), 'edge')

        voltage_spectrum = bd.fft(voltage_array)
        current_spectrum = bd.fft(current_array)

        impedance_spectrum = bd.divide(voltage_spectrum, current_spectrum, out=
                                    bd.zeros_like(voltage_spectrum), where=current_spectrum!=0)


        spectrum_freqs, begin_freq_idx, end_freq_idx = self.compute_frequencies(
                                        voltage_array.shape[0], self.grid.time_step,
                                         freq_window_tuple=freq_window_tuple)

        return spectrum_freqs[begin_freq_idx:end_freq_idx], impedance_spectrum[begin_freq_idx:end_freq_idx]


        #
        # voltages = bd.pad(voltages, (0, required_length), 'edge')
        # currents = bd.pad(currents, (0, required_length), 'edge')
        #
        # voltage_spectrum = bd.fft(voltages)
        # current_spectrum = bd.fft(currents)
        #
        # spectrum_freqs, begin_freq_idx, end_freq_idx = \
        #                         compute_frequencies(length_with_padding, dt, freq_window_tuple=None)
        #
        # spectrum_freqs[begin_freq:end_freq]


    def plot_impedance(self, freq_window_tuple=None, fft_num_bins_in_window=None,
                                            fft_bin_freq_resolution=None):
        '''
        Frequencies are in Hz.
        '''

        plt.plot(times_padded, voltages)

        plt.plot(times_padded, currents)
        # plt.plot(, abs(voltage_spectrum[begin_freq:end_freq]), label="volt")
        plt.plot(spectrum_freqs[begin_freq:end_freq], abs(current_spectrum[begin_freq:end_freq]), label="curr")


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

        plt.show()
        # power_spectrum = -1.0*((voltage_spectrum[begin_freq:end_freq]*np.conj(current_spectrum[begin_freq:end_freq])).real)
        # power_spectrum /= np.linalg.norm(power_spectrum)
        # plt.plot(spectrum_freqs[begin_freq:end_freq],power_spectrum)
