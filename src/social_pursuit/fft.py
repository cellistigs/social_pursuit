### Implements fft based transformations of individual pursuit events.
import os
import matplotlib.pyplot as plt
import numpy as np

class PursuitFFT():    
    """Calculates FFT-related quantities (transform, reconstruction, clustering, ordering, segmentation) on a collection of pursuit events.

    """
    def __init__(self,events,fps):
        """

        :param events: A list of filenames which contain numpy object arrays containing all relevant pursuit information.  
        :param fps: The sampling rate of the traces. Used to normalize frequencies given.

        """
        assert type(events) == list
        for event in events:
            assert os.path.exists(event),"must pass a list of existing files pointing to pursuit events."
        assert type(fps) == int
        assert fps > 0
        self.samp_rate = 1./fps

    def load_data(self,path):
        """Load in the data stored at the provided path.  

        :param path: string path to the data we need. 
        """
        assert os.path.splitext(path)[1] == ".npy"
        data = np.load(path,allow_pickle = True)[()]
        return data

    def get_fft_freqs(self,trace):
        """Wrapper function for numpy get frequencies. 

        """
        freqs = np.fft.fftfreq(len(trace),self.samp_rate)
        return freqs

    def get_ordered_fft_trace(self,trace,ordering):
        """Orderings: original, signed, signed_symmetric, 

        :param trace: the numpy array of positions we are applying the fft to. 
        :param ordering: a string with value in [original, signed, or signed_symm]. original indicates the raw ordering given by the numpy.fft.fft function ([0] is the sum, then ordered with largest magnitude frequencies in the middle). signed indicates ordering along R, most negative to the left and most positive to the right. signed symmetric indicates that for traces of even length, we will take the nyquist frequency, and copy it to the other side of the array for representational parity.
        :returns: tuple containing the actual fourier coefficients and the frequencies at which they are registered according to the given ordering.
        """
        assert ordering in ["original","signed","signed_symmetric"]
        fft = self.run_fft_trace(trace)
        freqs = self.get_fft_freqs(trace)
        
        if ordering is "original": 
            idx = np.arange(len(freqs))
            ordered = freqs[idx]
        elif ordering is "signed":
            idx = np.argsort(freqs)
            ordered = freqs[idx]
        elif ordering is "signed_symmetric":
            if len(trace)%2 == 1: 
                idx = np.argsort(freqs)
                ordered = freqs[idx]
            else:
                #The nyquist frequency is registered as a negative frequency. We can also take the negative frequency and show it on the positive side as well for the same of symmetrizing the representation. It should be understood that a symmetrized representation is purely for visual depiction purposes.   
                idx = np.concatenate([np.argsort(freqs),np.array([int(len(trace)/2)])])
                ordered = freqs[idx]
                ordered[-1] = abs(ordered[-1])
        return (fft[idx],ordered)


    def get_top_k_frequencies(self,trace):
        """Return only

        """
        pass
    
    def run_fft_trace(self,trace):
        """Wrapper function for numpy dft applied to a single x-y trace. 

        First convert to complex sequence, then calculate.
        :param trace: a numpy array of shape (t,2) providing the input trace. 
        """
        ## As a convention, y will be the imaginary part. 
        assert len(trace.shape) == 2, "must be a two dimensional array"
        assert trace.shape[1] == 2, "must be an x-y trace"
        ctrace = trace[:,0]+1j*trace[:,1]
        fft = np.fft.fft(ctrace)
        return fft

    def run_ifft_trace(self,ctrace):
        """Wrapper function for numpy idft applied to a single x-y trace.


        First convert apply ifft, then convert back to xy space. 
        :param ctrace: a numpy array of shape (t,) that provides the complex trace.  
        """
        assert len(ctrace.shape) == 1
        ifft = np.fft.ifft(ctrace)
        x = np.concatenate([np.real(ifft)[:,None],np.imag(ifft)[:,None]],axis=1) 
        return x

    def plot_spectrum(self,trace,path,order = "signed_symmetric"):
        """ Plot the amplitudes of each frequency component. 

        :param trace: numpy array representing the trace we care about. 
        :param path: path where the resulting image will be saved.  
        :param order: ordering for the frequencies when plotting. 
        """
        coef,freqs = self.get_ordered_fft_trace(trace,order)
        plt.plot(freqs,np.abs(coef))
        plt.xlabel("frequency")
        plt.ylabel("coefficient magnitude")
        plt.title("Spectrum")
        plt.savefig(path)
        plt.close()

    def plot_phase(self,trace,path,order = "signed_symmetric"):
        """Plot the phases of each frequency component.

        :param trace: numpy array representing the trace we care about. 
        :param path: path where the resulting image will be saved.  
        :param order: ordering for the frequencies when plotting. 
        """
        coef,freqs = self.get_ordered_fft_trace(trace,order)
        plt.plot(freqs,np.angle(coef))
        plt.xlabel("frequency")
        plt.ylabel("coefficient phase (rad)")
        plt.title("Phase")
        plt.savefig(path)
        plt.close()

    def __plot_phase_amp_internal(self,mag,phase,freqs,order):
        """Internal plotting function to plot both phase and amplitude when given these signals

        :param mag: Magnitude of fourier coefficients. 
        :param phase: Phase of fourier coefficients. 
        :param freqs: Corresponding fourier frequencies. 
        :param order: String, representing the ordering of frequencies. 
        """
        fig,ax = plt.subplots(1,2,figsize = (20,10))
        if order == "original":
            spacing = int(len(freqs)/8)
            ax[0].plot(phase)
            ax[0].set_xticks(np.arange(len(freqs))[::spacing])
            ax[0].set_xticklabels(["{:.2e}".format(f) for f in freqs[::spacing]])
            ax[1].plot(mag)
            ax[1].set_xticks(np.arange(len(freqs))[::spacing])
            ax[1].set_xticklabels(["{:.2e}".format(f) for f in freqs[::spacing]])
        else:
            ax[0].plot(freqs,phase)
            ax[1].plot(freqs,mag)
        ax[1].set_xlabel("frequency")
        ax[1].set_ylabel("coefficient amplitude")
        ax[1].set_title("Amplitude")
        ax[0].set_xlabel("frequency")
        ax[0].set_ylabel("coefficient phase (rad)")
        ax[0].set_title("Phase")
        return fig,ax

    def plot_phase_amplitude(self,trace,path,order = "signed_symmetric"):
        """Plot the phase amplitude representation of the fourier transform for a given trajectory. 
        :param trace: numpy array representing the trace we care about. 
        :param path: path where the resulting image will be saved.  
        :param order: ordering for the frequencies when plotting. 

        """
        assert order in ["original","signed","signed_symmetric"]
        coef,freqs = self.get_ordered_fft_trace(trace,order)
        phase = np.angle(coef)
        mag = np.abs(coef)
        fig,ax = self.__plot_phase_amp_internal(mag,phase,freqs,order)
        fig.savefig(path)
        plt.close(fig)

    def plot_psd(self,trace,path):
        """ Plot the power spectral density. Note this is square of the spectrum.   

        :param trace: numpy array representing the trace we care about. 
        """
        freqs = self.run_fft_trace(trace)
        ps = np.abs(freqs)**2
        freqs = np.fft.fftfreq(len(trace),1./30.)
        idx = np.argsort(freqs)
        plt.plot(freqs[idx],ps[idx])
        plt.title("Power Spectral Density")
        plt.xlabel("frequency")
        plt.ylabel("power")
        plt.savefig(path)
        plt.close()

    def reconstruct_trace_bandpass(self,trace,band):
        """Computes an fft, cuts off coefficients outside a given frequency band, and reconstructs the resulting trajectory. 

        :param trace: traces to use for analysis. 
        :param band: a list with two entries (one can be none) giving the lower and upper frequencies of a frequency band. 
        """
        assert len(band) == 2; "must give valid band limits."
        assert all([type(b) in [float,int,type(None)] for b in band]), "must be float, int or Nonetype. "

        ## convert to array
        bandarray = np.array([-np.inf,np.inf])
        for bi,b in enumerate(band):
            if b is None:
                pass
            else:
                bandarray[bi] = b

        freqs = self.get_fft_freqs(trace)
        conditionarrays = [freqs>b for b in bandarray]
        ## Find indices to remove:
        to_remove = np.logical_or(~conditionarrays[0],conditionarrays[1])
        coefs = self.run_fft_trace(trace)
        coefs[to_remove] = 0
        return self.run_ifft_trace(coefs)

    def measure_spectral_asymmetry_trace(self,trace):
        """Measures the difference between coefficients at frequencies of identical magnitude but opposite sign in the fft output. Leverages the fact that signals purely real signals have fourier transforms that are hermitian symmetric, and measured the degree of deviation from that hermitian symmetry.    

        :param trace: the behavioral trace that we will analyze.  
        :return: difference spectrum: a tuple containing the complex valued differences in the spectrum, measured by subtracting the coefficient at the negative frequency value from that at the positive corresponding frequency value. Separated into phase and amplitude information. 
        """
        coefs,freqs = self.get_ordered_fft_trace(trace,"signed_symmetric")
        ## With the signed symmetric representation, we will always have an odd number of frequencies, corresponding to the zero frequency and matching pairs. 
        zero_freq_ind = int(np.floor(len(freqs)/2.))
        negative_coefs = coefs[:zero_freq_ind]
        positive_coefs = coefs[zero_freq_ind+1:]
        ## Flip the negative coefficients so we follow the positive ordering
        neg_coefs_ordered = np.flip(negative_coefs,axis = 0)
        ## the symmetry is hermitian
        neg_coefs_conj = np.conjugate(neg_coefs_ordered)
        magdiffs = np.abs(positive_coefs) - np.abs(neg_coefs_conj)
        phasediffs = np.mod(np.angle(positive_coefs)-np.angle(neg_coefs_conj),2*np.pi) 
        return magdiffs,phasediffs,freqs[zero_freq_ind+1:]

    def plot_spectral_asymmetry_trace(self,trace,path):
        """Plots the spectral asymmetry of a given trace. 

        :param trace: the behavioral trace that we will analyze.  
        """
        m,p,f = self.measure_spectral_asymmetry_trace(trace)
        fig,ax = self.__plot_phase_amp_internal(m,p,f,"signed_symmetric")
        ax[0].set_title("Phase Asymmetry (Hermitian)")
        ax[1].set_title("Magnitude Asymmetry")
        fig.savefig(path)
        plt.close(fig)

    def plot_compare_trace_reconstruct(self,trace,path,erase=None):
        """Plots both the original trace and a reconstruction from ffts. Optionally, you can erase certain frequencies to do a partial reconstruction.

        :param trace: numpy array representing the trace we care about. 
        :param erase: (optional) a list representing the possible values that we can erase in the spectrum.
        """
        freqs = self.run_fft_trace(trace)
        if erase is None:
            pass
        else:
            freqs[erase] = 0
        reconstruct = self.run_ifft_trace(freqs)
        all_t = np.concatenate([trace,reconstruct], axis = 0)
        plt.plot(trace[:,0],trace[:,1],"r",label = "true")
        plt.plot(reconstruct[:,0],reconstruct[:,1],"b",label = "reconstruct")
        ax = plt.gca()
        ax.set_ylim(np.max(all_t)+50,np.min(all_t)-50)
        plt.legend()
        plt.title("Erased: {}".format(erase))
        plt.savefig(path)
        plt.close()

    def run_fft(self):
        """Computes the discrete fourier transform for all events in the given list. 

        First convert each x-y trace to a complex set.  
        """
        pass
        
