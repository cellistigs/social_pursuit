### Make a statistical shape model for your training data. 
import os 
import numpy as np
from social_pursuit.data import Polar,PursuitVideo,mkdir_notexists
from social_pursuit.labeled import LabeledData
from scipy.io import loadmat
from script_doc_utils import initialize_doc,insert_image,save_and_insert_image,get_relative_image_path,insert_vectors_as_table
from joblib import Memory
import joblib
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

datapath = os.path.join("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/TempTrial2roi_2cropped_part2DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat")
labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"

if __name__ == "__main__":
    md = initialize_doc()
    md.new_header(title = "Summary",level = 1)
    md.new_paragraph("We can use the contours extracted by "+md.new_inline_link(link = "./get_statistical_shape_model.py",text = "the previous file")+" ")
    all_contours = joblib.load("./script_data/all_contours")
    ## Assume for now that the first contour is the most representative: 
    contourdict = {i:all_contours[i] for i in range(len(all_contours))}
    ## Remove wonky looking contours for now
    defectlist = [14,15,16,55,80,81,97,99]
    for i in defectlist:
        contourdict.pop(i)
    



    md.create_md_file()

