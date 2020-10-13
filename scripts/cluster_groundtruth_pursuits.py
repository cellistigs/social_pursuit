## Run clustering on pursuit algorithms, in the sense of Annoni et al. 2012. 
import os 
import numpy as np
import matplotlib.pyplot as plt
from script_doc_utils import initialize_doc,save_and_insert_image,insert_image,get_relative_image_path

if __name__ == "__main__":
    md = initialize_doc({"parent":"summary_week_9_25_20"})
    md.new_header(title="Clustering Analysis",level =1)
    md.new_paragraph("First, we're going to characterize and study individual trajectories. We will characterize the groundtruth pursuit  dataset, in the following way: 1. Characterize the e")

    ## First cluster based on single trajectories. 
    ## Then cluster based on paired dam/virgin trajectories. 
    ## Then consider clustering to detect erroneous jumps to the other animal. Is there value to having a "fourier representation of pose?" Consider, for example, subtracting off the centroid position from other body parts and decomposing that...
    ## What features should we consider? 
    ## What's the right representation dimensionality?


