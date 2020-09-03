## Examine the comparisons between raw and groundtruth. 
import os
from social_pursuit import data
import pathlib
import datetime
from analysis_params import distance_threshold,velocity_threshold,metadata_path
from mdutils.mdutils import MdUtils

def convert_stats_to_list_of_strings(stats):
    """Converts statistics dictionary to a list of strings to match table format. 

    """
    keys = stats.keys()
    rois = [k for k in keys if type(k) == int]
    defaultstats = [] 
    for i in range(9):
        defaultstats.append([str(i),"None","None","None","None","None","None"])
    for roi in rois:
        roistats = stats[roi]
        for p,pdata in roistats.items():
            td = pdata["true_detect"]
            fd = pdata["false_detect"]
            defaultstats[p][1+roi*2] = str(td)
            defaultstats[p][roi*2+2] = str(fd)
    return defaultstats
        

if __name__ == "__main__":
    filternames = ["groundtruth"]
    pt = data.PursuitTraces(metadata_path)
    experiment_stats = {}
    for exp,edict in pt.experiments.items():
        if edict.get("GroundTruth",None):
            output = pt.retrieve_groundtruth_statistics(exp,filternames)
            experiment_stats[exp] = output


    pathname = "../docs/script_docs/"
    filename = pathlib.Path(__file__).stem 
    mdFile = MdUtils(file_name = os.path.join(pathname,filename),title = "Script documentation for file: "+ filename+", Updated on:" +str(datetime.datetime.now()))
    mdFile.new_header(level=1, title = "Summary")
    mdFile.new_paragraph("This script assumes that one has already generated comparisons between raw auto-detected pursuits and groundtruth labeled by Zahra using the `data.calculate_statistics()` method. It then takes the generated comparisons, and calculates the percentage of groundtruth pursuits that are correctly captured by raw auto-detection and the percentage of auto-detections that do not correspond to a groundtruth labeled pursuit events.  These represent the true detection rate and false detection rate, respectively. It is important to note that as a hyperparameter we include the a 'buffer': when evaluating if pursuit events from one set of criteria (auto-detection or groundtruth) are found by another, we allow for a frame buffer to capture slight offsets. The default value of this buffer is 5 frames on each side. ".format(filternames[0]))
    mdFile.new_header(level=1, title = "Data Overview")
    mdFile.new_paragraph("Here we present the raw/groundtruth comparison data, divided into experiments, rois, and 40 minute segments.")
    for exp,stats in experiment_stats.items():
        mdFile.new_line()
        mdFile.new_header(level = 3,title = exp)
        rois = pt.get_boxes(exp)
        header = ["Part"]
        types = ["true detection proporition","false detection proportion"]
        header.extend(["ROI{i}_{t}".format(i=i,t=t) for i in rois for t in types])
        full_l_l_s = [header]
        full_l_l_s.extend(convert_stats_to_list_of_strings(stats))
        flatten = [i for l in full_l_l_s for i in l]
        print(len(flatten))
        mdFile.new_table(columns = 7,rows = 10,text = flatten,text_align = "center")
    
    mdFile.new_line()
    mdFile.create_md_file()
