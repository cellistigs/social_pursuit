## Notebook for validation of an error classification algorithm. 
import os 
import numpy as np
from social_pursuit.data import Polar,PursuitVideo,mkdir_notexists
from scipy.io import loadmat
from script_doc_utils import initialize_doc,insert_image,save_and_insert_image,get_relative_image_path,insert_vectors_as_table
from joblib import Memory
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
})
from script_doc_utils import initialize_doc,insert_image,get_relative_image_path
cachedir = "/Volumes/TOSHIBA EXT STO/cache"
memory = Memory(cachedir, verbose=1)

if __name__ == "__main__":
    md = initialize_doc({"prev":"polar_classification","parent":"summary_week_10_9_20"})
    md.new_header(title = "Summary", level = 1)
    md.new_paragraph("We want to validate our ability to detect errors in animal body part positions. In order to do this, our ground truth dataset will be a set of human labeled frames. We will look at the distribution of body part distances from this ground truth dataset, and examine how well data extracted from our sample lines up with this ground truth dataset.")   
    ### The algorithm is as follows: 
    md.new_paragraph("The algorithm is as follows:")
    md.new_list(["Identity Resolution",["Get Mean Position","Cluster Body Parts Based on Mean Position","Detect Points where only one animal is being detected"],"Part Resolution",["Calculate per-identity centroids","Calculate per-identity distance matrices","Filter parts with distance matrix based hypothesis testing","Reconstruct body centroid from accepted parts"]])
    md.new_paragraph("In order to validate this algorithm we have created a dataset class, `social_pursuit.labeled.LabeledData`, that can retrieve the statistics of bodies labeled in the training data, as well as sample from the training data/generate surrogate data with simulated errors.")
    md.new_paragraph("The next step will be to implement the variance ratio criterion with 2 and 5 clusters to the spectral representations of our data. We expect that this will give us a good indicator for when we have achieved good clustering, and when we have degenerated to detection of a single animal. We should compare this to variance ration criteria on the raw angles directly, as well as on the relevant analogue measures for the actual animal coordinates, not just the angles.")
    md.create_md_file()
