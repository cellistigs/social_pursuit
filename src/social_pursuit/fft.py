### Implements fft based transformations of individual pursuit events.
import os
import matplotlib.pyplot as plt
import numpy as np

class PursuitFFT():    
    """Calculates FFT-related quantities (transform, reconstruction, clustering, ordering, segmentation) on a collection of pursuit events.

    """
    def __init__(self,events):
        """

        :param events: A list of filenames which contain numpy object arrays containing all relevant pursuit information.  

        """
        assert type(events) == list
        for event in events:
            print(event)
            assert os.path.exists(event),"must pass a list of existing files pointing to pursuit events."

    def load_data(self,path):
        """Load in the data stored at the provided path.  

        :param path: string path to the data we need. 
        """
        assert os.path.splitext(path)[1] == ".npy"
        data = np.load(path,allow_pickle = True)[()]
        return data
    
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

    def plot_compare_trace_reconstruct(self,trace,erase=None):
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
        plt.plot(trace[:,0],trace[:,1],"r",label = "true")
        plt.plot(reconstruct[:,0],reconstruct[:,1],"b",label = "reconstruct")
        plt.legend()
        plt.title("Erased: {}".format(erase))
        plt.show()

    def plot_spectrum(self,trace):
        """ Plot the amplitudes of each frequency component. 

        :param trace: numpy array representing the trace we care about. 
        """
        freqs = self.run_fft_trace(trace)
        plt.plot(np.abs(freqs))
        plt.title("Absolute Magnitudes of each component. ")
        plt.show()


    def run_fft(self):
        """Computes the discrete fourier transform for all events in the given list. 

        First convert each x-y trace to a complex set.  
        """
        pass
        
