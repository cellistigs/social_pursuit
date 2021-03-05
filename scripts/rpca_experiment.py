import os
import numpy as np
from social_pursuit.labeled import LabeledData
from social_pursuit.onlineRPCA.rpca.pcp import pcp
import joblib

from script_doc_utils import initialize_doc,insert_image,save_and_insert_image,get_relative_image_path,insert_vectors_as_table
datapath = os.path.join("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/TempTrial2roi_2cropped_part2DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat")
labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"

if __name__ == "__main__":
    md = initialize_doc()
    md.new_header(title = "Testing RPCA for foreground segmentation",level = 1)
    md.new_paragraph("I recently came across an interesting paper (Candes et al. 2009) that uses Robust PCA for foreground segmentation This is an interesting application, and one that I'd like to see put into practice if it's feasible. We will try it out here on our labeled training frames." )
    data = LabeledData(labeled_data,additionalpath)
    frameinds = [1,2,3,4] ## ignore 0 as size is different. 
    frames = data.get_images(frameinds)
    flatframes = [f.flatten() for f in frames]
    dataarray = np.stack(flatframes,axis = -1) ## data should be in columns. 
    L,S,niter,rank = pcp(dataarray)
    joblib.dump(L,"testL")
    joblib.dump(S,"testS")
    print(niter,rank)


