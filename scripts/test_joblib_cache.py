import os 
import datetime
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

@memory.cache(ignore = ['md'])
def test_redos(md,labeleddata,ident):
    print("ran this: {}!".format(ident))
    return labeleddata.marker 


if __name__ == "__main__":
    md = initialize_doc({"parent":"summary_week_10_9_20"})
    md.new_header(title = "Summary",level = 1)
    md.new_paragraph("This is a training script to see how best to use the joblib cache. We're going to define a function test_redos that is cached, and takes the markdown file itself and a data object as input. This will not activate the cache, but rerun every time in its raw form:")

    data = LabeledData(labeled_data,additionalpath)
    data.marker = 0
    ident = "mdfile run"
    md.new_line("`test_redos(md,data,ident)`")
    assert test_redos(md,data,ident) == 0## will rerun every time
    md.new_line("However, replacing the markdown document will not trigger a rerun.")
    ident = "integer run"
    md.new_line("`test_redos(0,data,ident)`")
    test_redos(0,data,ident) ## will not rerun after one successful run
    md.new_paragraph("What happens if we change a field of the data object?")
    md.new_line("`test_redos(0,data1,ident)`")
    ident = "changed object run"
    data.marker = 1 
    assert test_redos(0,data,ident) == 1 ## will not rerun after one successful run
    
    md.new_paragraph("What happens if we set the ignore flag on the markdown document? It turns out that this works.")
    md.new_paragraph("The only thing we have to be careful of is to make sure that all cached functions do not themselves add material to the md document, whether that material consists of text or images. ")
    md.create_md_file()


