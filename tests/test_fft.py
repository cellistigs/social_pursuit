from social_pursuit.fft import PursuitFFT
import os
import numpy as np

drivepath = "/Volumes/TOSHIBA EXT STO/RTTemp_Traces/test_dir"

def test_PursuitFFT_init():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path])

def test_PursuitFFT_load_data():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path])
    z = fft.load_data(path)
    assert list(z.keys()) == ['ExperimentName', 'ROI', 'VideoPart', 'Interval', 'mtraj', 'vtraj', 'mtip', 'vtip', 'mlear', 'vlear', 'mrear', 'vrear', 'mtail', 'vtail', 'pursuit_direction', 'pursuit_direction_agg']

def test_PursuitFFT_run_fft_trace():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path])
    z = fft.load_data(path)
    transformed = fft.run_fft_trace(z["mtraj"])

def test_PursuitFFT_run_ifft_trace():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path])
    z = fft.load_data(path)
    trace = z["mtraj"]
    transformed = fft.run_fft_trace(trace)
    returned = fft.run_ifft_trace(transformed)
    ## check up to numerical precision.
    assert not np.any((trace-returned)> 1e-6)

def test_PursuitFFT_plot_compare_trace_reconstruct():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path])
    z = fft.load_data(path)
    trace = z["mtraj"]
    trace[20:50,:] = trace[20:50,:] + 50
    fft.plot_compare_trace_reconstruct(trace,erase = [slice(35-25,35+25)])

def test_PursuitFFT_plot_spectrum():
    filepath = "TempTrial2_Pursuit_Events/FILTER_groundtruth/ROI_1/PART_0/Pursuit[27542 27617]_Direction1.0.npy"
    path = os.path.join(drivepath,filepath)
    fft = PursuitFFT([path])
    z = fft.load_data(path)
    trace = z["mtraj"]
    ## Point discontinuities lead to wiggles in the frequency spectrum.
    #trace[50,:] = np.array([0,0])
    ## Point discontinuities lead to wiggles in the frequency spectrum (not as intense)
    fft.plot_spectrum(trace)
    
