### Script to explore frequency and time-frequency methods for the analysis of this data. 
import os
import numpy as np
from script_doc_utils import initialize_doc,insert_image,get_relative_image_path
from social_pursuit import data, fft
from mdutils.mdutils import MdUtils
from analysis_params import metadata_path

if __name__ == "__main__":
    experimentname = "TempTrial2"
    filtername = "groundtruth"
    roi = 1
    part = 0
    interval = [27542,27617]
    pt = data.PursuitTraces(metadata_path)
    tracedirectory = pt.get_trace_filter_dir(experimentname,roi,part,filtername)
    filename = pt.get_pursuit_eventname(tracedirectory,interval)
    md = initialize_doc()
    md.new_header(level = 1, title = "Summary")
    md.new_paragraph("This script is meant to explore the effect of frequency based methods on the kinds of traces that we care about. We will show how to implement the fourier transform, obtain the power spectrum, and reconstruct these trajectories with a low rank approximation. All of these methods will be handled by the `social_pursuit.fft.PursuitFFT` object.")
    
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
    analysisobj = fft.PursuitFFT([filename])   
    z = analysisobj.load_data(filename)
    trace = z["mtraj"]
    image2_path = "../docs/script_docs/images/candidate_pursuit_dam_spectrum.png" 
    analysisobj.plot_spectrum(trace,image2_path)
    image2_rel_path = get_relative_image_path(image2_path)
    insert_image(md,image2_rel_path,align = 'center')
    md.new_paragraph("This representation shows that there is a sharp peak in the frequency content of this trajectory in frequencies close to zero. This means that relatively slow frequencies dominate the activity that we see, setting us up to reduce dimensionality by considering only the low frequency activity in further analyses. Furthermore, the existence of positive and negative frequencies indicate components that rotate in opposing directions. Symmetry between the positive and negative frequencies should (I think) indicate straight trajectories, or trajectories with an equal and opposing amount of rotation to the left vs. the right.")
    ## straight diagonal
    #straighttrace = np.concatenate([np.linspace(0,1,10)[:,None],np.linspace(-1,0,10)[:,None]],axis =1)
    ## straight y only
    straighttrace = np.concatenate([np.linspace(0,1,10)[:,None],np.linspace(0,0,10)[:,None]],axis =1)
    print(straighttrace)
    analysisobj.plot_spectrum(straighttrace,"../docs/script_docs/images/straighttracespectrum.png")
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
