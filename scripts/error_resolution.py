## Error Detection and Resolution in social behavior tracking. 
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

if __name__ == "__main__":
    md = initialize_doc({"next":"polar_classification"})
    md.new_header(title = "Summary", level = 1)
    md.new_paragraph("We will give an overview of our error detection and resolution pipeline for social behavior data. Note that our goal is to focus on the issue of identity resolution.")
    md.new_header(title = "Background and other approaches",level = 2)
    md.new_paragraph("The issue of error detection and resolution for social behavior data is being tackled from several different appraches by various groups in the pose tracking field. The authors of DLC approach this by applying state space models to detected poses- either to a bounding box around the animal, or to the animal's estimated skeleton directly. The methods (as far as I understand them from the code here (https://github.com/DeepLabCut/DeepLabCut/blob/be788615dcfe6add93c09033bf84545ca78a0130/deeplabcut/pose_estimation_tensorflow/lib/trackingutils.py)) are entirely linear, and will suffer from the issues of linear tracking when applied to problems with highly nonlinear noise. As a detection step, they also introduce part affinity fields that bias detections to respect skeleton boundaries. It is known however that this does not resolve issues in tracking. Separately we have groups that aim to improve detections not at the level of tracked points, but probability densities of these points with structured variational inference (DGP), introducing graphical models that impose temporal smoothness and skeleton constraints via potentials in the detection cost. This approach is promising, but it is also potentially overkill. Let's see if we can do anything simpler with the methods we know. Both of these methods. Beyond this, I only know of heuristic approaches to the problem.")
