### Script to explore frequency and time-frequency methods for the analysis of this data. 
import os
from script_doc_utils import initialize_doc,insert_image
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
    image1_path = "../docs/script_docs/images/candidate_pursuit.png"
    pt.plot_trajectories_from_filenames(experimentname,roi,part,[filename],image1_path)

    insert_image(md,image1_path)

    ### Look at the fourier transform, spectrum, power spectral density, and reconstruction. 

    ## Plot a rotated and translated version of that test trace and see how that fares under same analytics.  

    ## Plot a corrupted version with a point discontinuity. Can we distinguish this in the fourier spectrum?   
    ## What about over different corruptions (create a bootstrap estimate)

    ## Likewise what if it's not a single discontinuity? Consider the switch case.  

    ## Can we cluster the real social pursuit events? Generate a low dim representation of trajectories.  
    md.create_md_file()
