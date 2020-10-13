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
    for i in range(10):
        defaultstats.append(["None"]*7)
    for roi in rois:
        roistats = stats[roi]
        for p,pdata in roistats.items():
            if type(p) is int:
                index = p
                defaultstats[index][0] = str(p)
                td = pdata["true_detect"]
                fd = pdata["false_detect"]
                defaultstats[index][1+roi*2] = str(td)[:4]
                defaultstats[index][roi*2+2] = str(fd)[:4]
            elif p == "total":
                index = -1
                defaultstats[index][0] = p
                td = pdata["true_detect"]
                fd = pdata["false_detect"]
                tt = pdata["true_total"]
                tf = pdata["false_total"]
                defaultstats[index][1+roi*2] = "{p} (Total:{t})".format(p = str(td)[:4],t = str(tt))
                defaultstats[index][roi*2+2] = "{p} (Total:{t})".format(p = str(fd)[:4],t = str(tf))
            else:
                raise Exception("index type invalid.")
    return defaultstats
        
def make_table(exp,stats):
    rois = pt.get_boxes(exp)
    header = ["Part"]
    types = ["true detection proportion","false detection proportion"]
    header.extend(["ROI{i}_{t}".format(i=i,t=t) for i in rois for t in types])
    full_l_l_s = [header]
    full_l_l_s.extend(convert_stats_to_list_of_strings(stats))
    flatten = [i for l in full_l_l_s for i in l]
    return flatten

if __name__ == "__main__":
    filternames = ["groundtruth"]
    pt = data.PursuitTraces(metadata_path)
    experiment_stats = {}
    for exp,edict in pt.experiments.items():
        if edict.get("GroundTruth",None):
            output = pt.retrieve_groundtruth_statistics_eventwise(exp,filternames)
            experiment_stats[exp] = output


    pathname = "../docs/script_docs/"
    filename = pathlib.Path(__file__).stem 
    mdFile = MdUtils(file_name = os.path.join(pathname,filename),title = "Script documentation for file: "+ filename+", Updated on:" +str(datetime.datetime.now()))

    mdFile.new_header(level=1, title = "Summary")
    mdFile.new_paragraph("This script assumes that one has already generated comparisons between raw auto-detected pursuits and groundtruth labeled by Zahra using the `data.calculate_statistics()` method. It then takes the generated comparisons, and calculates the percentage of groundtruth pursuits that are correctly captured by raw auto-detection and the percentage of auto-detections that do not correspond to groundtruth labeled pursuit events, when measured by event, by duration, and taking directionality into account.  It is important to note that as a hyperparameter we include a 'buffer': when evaluating if pursuit events from one set of criteria (auto-detection or groundtruth) are found by another, we allow for a frame buffer to capture slight offsets. The default value of this buffer is 5 frames on each side. ".format(filternames[0]))

    mdFile.new_header(level=1, title = "Data Overview (Eventwise)")
    mdFile.new_paragraph("Here we present the raw/groundtruth comparison data, divided into experiments, rois, and 40 minute segments. We present an eventwise comparison: what are the proportion of groudntruth sheperding events that are correctly detected at any frame by automatic tracking (true detection), and what proportion of these automatically tracked events have no overlap with manual labels? (false detection)")
    
    for exp,stats in experiment_stats.items():
        mdFile.new_line()
        mdFile.new_header(level = 3,title = exp+" (Eventwise)")
        flatten = make_table(exp,stats)
        mdFile.new_table(columns = 7,rows = 11,text = flatten,text_align = "center")
    
    mdFile.new_header(level = 1, title = "Next Steps (Eventwise)")
    mdFile.new_paragraph("These tables show that we have a pretty high baseline rate of capturing pursuit events (around 70-90 percent for most cases). However at the same time we also have a pretty high false detection rate (around 50-80 percent for most cases). Are there any features of falsely detected pursuits that distinguish them from true detected pursuits? Likewise, are there any features of false negative pursuits that we should be selecting for? ")

    mdFile.create_md_file()
