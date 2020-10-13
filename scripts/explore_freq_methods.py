### Script to explore frequency and time-frequency methods for the analysis of this data. 
import os
import numpy as np
import matplotlib.pyplot as plt
from script_doc_utils import initialize_doc,insert_image,get_relative_image_path
from social_pursuit import data, fft
from mdutils.mdutils import MdUtils
from analysis_params import metadata_path
from joblib import Memory
cachedir = "/Volumes/TOSHIBA EXT STO/cache"
memory = Memory(cachedir, verbose=0)

imagedir = "../docs/script_docs/images/"
scriptdir = "../docs/script_docs/"
fps = 30

def initialize_analysis_objects(experimentname,roi,part,interval,filtername):
    pt = data.PursuitTraces(metadata_path)
    tracedirectory = pt.get_trace_filter_dir(experimentname,roi,part,filtername)
    filename = pt.get_pursuit_eventname(tracedirectory,interval)
    ## Initialize analysis object first.
    analysisobj = fft.PursuitFFT([filename],fps)   

    return pt,analysisobj,filename

def plot_trace_bandpass(corrupted,band,image3_path):
    reconstructed = analysisobj.reconstruct_trace_bandpass(corrupted,band)
    plt.plot(reconstructed[:,0],reconstructed[:,1])
    plt.axis("off")
    plt.title("Reconstruction from band {}".format(band))
    plt.savefig(image3_path)
    plt.close()

if __name__ == "__main__":
    experimentname = "TempTrial2"
    filtername = "groundtruth"
    roi = 1
    part = 0
    interval = [27542,27617]
    pt,analysisobj,filename = initialize_analysis_objects(experimentname,roi,part,interval,filtername)
    md = initialize_doc({"parent":"summary_week_9_25_20"})
    md.new_header(level = 1, title = "Summary")
    md.new_paragraph("This script is meant to explore the effect of frequency based methods on the kinds of traces that we care about. It follows the approach of papers in air traffic control (Annoni et al. 2012) and ecology (Polansky et al. 2010) to apply frequency based methods to the study of real, two dimensional trajectories. This approach could complement and provide foundations for our current data analysis techniques, and gives us nice ways to characterize trajectories of different length.  We will show how to implement the fourier transform, obtain the power spectrum, and reconstruct these trajectories with a low rank approximation. All of these methods will be handled by the `social_pursuit.fft.PursuitFFT` object.")
    
    ## Lay out some basic characteristics. 
    md.new_header(level = 2, title = "Basic Properties of the trajectory FFT")
    md.new_header(level = 3, title = "Amplitude and Phase representation")
    md.new_paragraph("First, we try applying fourier transforms to a straight line segment that aligns with y = -x.")
    
    straighttrace = np.concatenate([np.linspace(-0.5,0.5,30)[:,None],np.linspace(0.5,-0.5,30)[:,None]],axis =1)
    plt.plot(straighttrace[:,0],straighttrace[:,1])
    plt.axis("off")
    plt.title("Toy trajectory")
    toyimpath = os.path.join(imagedir,"toytrajectory.png")
    plt.savefig(toyimpath)
    plt.close()
    insert_image(md,get_relative_image_path(toyimpath),align = 'center')

    md.new_paragraph("We generate a fourier transform for this trace by projecting the y axis to the imaginary plane, and treating the trace as a 1-d complex signal.")
    imagestraightpath = "../docs/script_docs/images/straighttracephaseamp.png"
    analysisobj.plot_phase_amplitude(straighttrace,imagestraightpath)
    relimagestraightpath = get_relative_image_path(imagestraightpath)
    insert_image(md,relimagestraightpath,align = 'center')
    md.new_paragraph("On the left, we can see the phase content of the fourier transform, ordered by frequency. The phase component determines the 'initial conditions' for each frequency value. Moreover, for we can see that (despite the discontinuities introduced by the range (-pi, pi)), the coefficient phases at the negative and positive frequencies have the same slope. By applying a rotation to all of these phases, we can similarly rotate the trajectory in space (because the Fourier Transform is a linear operation). We can separately examine the amplitude content to the right. The existence of positive and negative frequencies indicate components that rotate in opposing directions. Symmetry between the positive and negative frequencies indicates straight trajectories, or trajectories with an equal and opposing amount of rotation to the left vs. the right. Note that when we have an even number of input points, the resulting fft representation gives the positive and negative nyquist frequency as a single term, leading to apparent asymmetry. We have symmetrized the representation here for purposes of visual depiction. Note that because 1) the FFT is linear and 2) spectra throw away all rotational information, the spectrum of the FFT will be identical regardless of the rotational orientation of a trajectory. ")

    ## Dim reduction: what happens if we throw away some components? 
    md.new_header(level = 3,title = "reconstruction")
    md.new_paragraph("One very useful property of the transform is the ability to analyze and decompose the trajectory into movement on different scales. Consider what happens when we work with the same trace, but corrupted by random gaussian noise of magnitude 0.05")
    corrupted = straighttrace + np.random.randn(*straighttrace.shape)*0.05
    plt.plot(corrupted[:,0],corrupted[:,1])
    plt.axis("off")
    plt.title("Toy trajectory (noised)")
    toyimnoisepath = os.path.join(imagedir,"toytrajectorynoised.png")
    plt.savefig(toyimnoisepath)
    plt.close()
    insert_image(md,get_relative_image_path(toyimnoisepath),align = 'center')
 
    md.new_paragraph("We can then examine the fft of this noised data:")
    imagestraightpath = "../docs/script_docs/images/noisedtracephaseamp.png"
    analysisobj.plot_phase_amplitude(corrupted,imagestraightpath)
    relimagestraightpath = get_relative_image_path(imagestraightpath)
    insert_image(md,relimagestraightpath,align = 'center')
    md.new_paragraph("We can see that the resulting phase and amplitude diagram are significantly altered by the noise (and are notably no longer symmetric). Since we know that much of the fluctuations are at a small scale, what happens if we just discard the small scale information by zeroing trajectories below a certain threshold?")
    band = [-5,5]
    image3_path = "../docs/script_docs/images/noisylinereconstruct.png"
    plot_trace_bandpass(corrupted,band,image3_path)
    image3_rel_path = get_relative_image_path(image3_path)
    insert_image(md,image3_rel_path,align = 'center')
    md.new_paragraph("This reconstruction was performed by zeroing the fourier coefficients outside of the range {}. It is not a great reconstruction, but one can see that it preserves the general shape features of the trajectory well. This is a good basis for featurization, if not for outright reconstruction.".format(band))
    ## What happens if we symmetrize a spectrum?  
    md.new_header(level = 3, title = "Asymmetry analysis and symmetrization")
    md.new_paragraph("Finally, we want to take a closer look at the asymmetry between the positive and negative frequencies of the fourier spectrum, which indicates overall rotational trends.")
    image_symm_path = "../docs/script_docs/images/straightlinesymm.png"
    analysisobj.plot_spectral_asymmetry_trace(straighttrace,image_symm_path)
    insert_image(md,get_relative_image_path(image_symm_path),align = "center")
    md.new_paragraph("We motivate this analysis from the idea that the FFT for a purely real trajectory is Hermitian: F(-x) - F*(x). This means that for our trajectories, where we project the y coordinate to the imaginary plane, any trajectories that are purely horizontal will have a amplitude and phase difference of zero (when measuring relative to hermitian symmetry.) Any straight lines will therefore have an amplitude difference of 0, and a constant phase difference across all trajectories. We can confirm this visually in the plot above (note the scale on the y axis.)")
    md.new_paragraph("Let's see what happens when we measure this same asymmetry for our corrupted trajectory. ")
    image_noisesymm_path = "../docs/script_docs/images/noisedlinesymm.png"
    analysisobj.plot_spectral_asymmetry_trace(corrupted,image_noisesymm_path)
    insert_image(md,get_relative_image_path(image_noisesymm_path),align = "center")
    md.new_paragraph("We see that the difference measures fluctuate around the true values given above.")

    


    
    

    ## Plot a test trace that looks realistic.  
    md.new_header(level = 2, title = "Candidate dataset")
    md.new_paragraph("We will pilot these analyses on a candidate pursuit instance whose filename can be found in the accompanying script.")
    image1_path = "../docs/script_docs/images/candidate_pursuit.png"
    pt.plot_trajectories_from_filenames(experimentname,roi,part,[filename],image1_path)
    image1_rel_path = get_relative_image_path(image1_path) 
    insert_image(md,image1_rel_path,align = 'center')
    md.new_paragraph("We will focus on the dam's trajectory in subsequent analyses (shown here in purple)")

    ### Look at the fourier transform, spectrum, power spectral density, and reconstruction. 
    md.new_paragraph("First, we can generate a fourier transform of the data. We do so by first projecting the y axis of the data to the imaginary axis, treating the two dimensional dam's centroid trajectory as a one dimensional complex trajectory. Here we show the spectrum of the dam's centroid trajectory, discarding the phase information for ease of presentation.")
    analysisobj = fft.PursuitFFT([filename],30)   
    z = analysisobj.load_data(filename)
    trace = z["mtraj"]
    image2_path = "../docs/script_docs/images/candidate_pursuit_dam_spectrum.png" 
    analysisobj.plot_spectrum(trace,image2_path)
    image2_rel_path = get_relative_image_path(image2_path)
    insert_image(md,image2_rel_path,align = 'center')
    md.new_paragraph("This representation shows that there is a sharp peak in the frequency content of this trajectory in frequencies close to zero. This means that relatively slow frequencies dominate the activity that we see, setting us up to reduce dimensionality by considering only the low frequency activity in further analyses.")
    md.new_paragraph("We can also apply an inverse fourier transform to reconstruct the data back from this frequency representation:")
    image3_path = "../docs/script_docs/images/candidate_pursuit_dam_full_reconstruct.png"
    analysisobj.plot_compare_trace_reconstruct(trace,image3_path)
    image3_rel_path = get_relative_image_path(image3_path)
    insert_image(md,image3_rel_path,align = 'center')

    ## Plot a rotated and translated version of that test trace and see how that fares under same analytics.  
    md.new_paragraph("One important thing we want to do with these datasets now is to cluster them. In order to do so, we should figure out which invariances we care about. For example, by disregarding the 0th component of the fourier transform, we can translate all trajectories back to be centered at the origin, allowing for comparison only of the morphologies of the resulting trajectories. ")

    ## Plot a corrupted version with a point discontinuity. Can we distinguish this in the fourier spectrum?   
    ## What about over different corruptions (create a bootstrap estimate)

    ## Likewise what if it's not a single discontinuity? Consider the switch case.  

    ## Can we cluster the real social pursuit events? Generate a low dim representation of trajectories.  
    md.create_md_file()
