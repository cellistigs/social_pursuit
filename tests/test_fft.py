from social_pursuit.fft import PursuitFFT
import os
import numpy as np

drivepath = "/Volumes/TOSHIBA EXT STO/RTTemp_Traces/test_dir"

def test_PursuitFFT_init():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542, 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)

def test_PursuitFFT_load_data():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542, 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    assert list(z.keys()) == ['ExperimentName', 'ROI', 'VideoPart', 'Interval', 'mtraj', 'vtraj', 'mtip', 'vtip', 'mlear', 'vlear', 'mrear', 'vrear', 'mtail', 'vtail', 'pursuit_direction', 'pursuit_direction_agg']

def test_PursuitFFT_get_fft_freqs():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542, 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)["mtraj"]
    freqs = fft.get_fft_freqs(z)
    assert np.all(freqs == np.fft.fftfreq(len(z),1/30.))

def test_PursuitFFT_get_ordered_fft_trace_original():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542, 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)["mtraj"]
    a = fft.get_ordered_fft_trace(z,"original")
    assert np.all(a[1] == np.fft.fftfreq(len(z),1/30.))

def test_PursuitFFT_get_ordered_fft_trace_signed():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542, 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)["mtraj"]
    a = fft.get_ordered_fft_trace(z,"signed")
    assert np.all(a[1] == np.sort(np.fft.fftfreq(len(z),1/30.)))

def test_PursuitFFT_get_ordered_fft_trace_signed_symm_even():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542, 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)["mtraj"][:-1,:]
    a = fft.get_ordered_fft_trace(z,"signed_symmetric")
    assert np.all(a[1] == np.concatenate([np.sort(np.fft.fftfreq(len(z),1/30.)),-np.array([np.fft.fftfreq(len(z),1/30.)[int(len(z)/2)]])]))

def test_PursuitFFT_get_ordered_fft_trace_signed_symm_odd():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542, 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)["mtraj"]
    a = fft.get_ordered_fft_trace(z,"signed_symmetric")
    assert np.all(a[1] == np.sort(np.fft.fftfreq(len(z),1/30.)))

def test_PursuitFFT_run_fft_trace():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    transformed = fft.run_fft_trace(z["mtraj"])

def test_PursuitFFT_run_ifft_trace():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    trace = z["mtraj"]
    transformed = fft.run_fft_trace(trace)
    returned = fft.run_ifft_trace(transformed)
    ## check up to numerical precision.
    assert not np.any(abs(trace-returned)> 1e-6)

def test_PursuitFFT_reconstruct_trace_bandpass_full():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    trace = z["mtraj"]
    band = [None,None]
    reconstructed = fft.reconstruct_trace_bandpass(trace,band)
    assert not np.any(abs(trace-reconstructed)> 1e-6)
    
def test_PursuitFFT_reconstruct_trace_bandpass():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    trace = z["mtraj"]
    band = [-10,10]
    reconstructed = fft.reconstruct_trace_bandpass(trace,band)
    assert np.any(abs(trace-reconstructed)> 1e-6)

def test_PursuitFFT_plot_compare_trace_reconstruct():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    trace = z["mtraj"]
    trace[20:50,:] = trace[20:50,:] + 50
    savepath = os.path.join(drivepath,"test_plot_compare_trace_reconstruct.png")
    fft.plot_compare_trace_reconstruct(trace,savepath,erase = [slice(35-25,35+25)])

def test_PursuitFFT_measure_spectral_asymmetry_trace():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    trace = np.concatenate([np.sin(np.linspace(0,1,10))[:,None],np.linspace(0,0,10)[:,None]],axis = 1)
    m,p,f = fft.measure_spectral_asymmetry_trace(trace)
    assert np.all(m == 0), "should be 0 uniformly for a purely real signal" 
    assert np.all(p == 0), "should be 0 uniformly for a purely real signal" 

def test_PursuitFFT_plot_spectral_asymmetry_trace():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    trace = np.concatenate([np.linspace(0,1,10)[:,None],np.linspace(0,-1,10)[:,None]],axis = 1)
    savepath = os.path.join(drivepath,"test_spectral_asymmetry.png")    
    fft.plot_spectral_asymmetry_trace(trace,savepath)

def test_PursuitFFT_plot_spectrum():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    trace = z["mtraj"]
    savepath = os.path.join(drivepath,"test_plot_spectrum.png")
    fft.plot_spectrum(trace,savepath)

def test_PursuitFFT_plot_phase():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    trace = z["mtraj"]
    savepath = os.path.join(drivepath,"test_plot_phase.png")
    fft.plot_phase(trace,savepath)
    
def test_PursuitFFT_plot_phase_amplitude():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    trace = z["mtraj"]
    savepath = os.path.join(drivepath,"test_plot_phase_amplitude.png")
    fft.plot_phase_amplitude(trace,savepath,order = "signed_symmetric")

def test_PursuitFFT_plot_psd():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path],30)
    z = fft.load_data(path)
    trace = z["mtraj"]
    savepath = os.path.join(drivepath,"test_plot_psd.png")
    ## Point discontinuities lead to wiggles in the frequency spectrum.
    ## Point discontinuities lead to wiggles in the frequency spectrum (not as intense)
    fft.plot_psd(trace,savepath)
