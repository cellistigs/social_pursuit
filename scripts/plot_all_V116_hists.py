import script_doc_utils 
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from social_pursuit.labeled import PoseDistribution,FourierDescriptor
from scipy.spatial import distance as scidist
from scipy.stats import wasserstein_distance
import numpy as np
import joblib
from joblib import Memory
cachedir = "/Volumes/TOSHIBA EXT STO/cache"
memory = Memory(cachedir, verbose=1)
experimentdictpath = "../src/social_pursuit/trace_template_V116.json"
labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out',rotation = 30)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Pursuit Start (Seconds)')

if __name__ == "__main__":
    md = script_doc_utils.initialize_doc({"prev":"compare_V116_image_hists"})
    md.new_header(title = "Summary",level = 1)
    md.new_paragraph("In the previous file, `compare_V116_image_hists`, we generated emd distances between the contours of each animal and 1) the average histogram from the training set, and 2) the previous histogram in a sequence. We want to show the aggregate values of these distances across an entire experiment, chunking over video segments and individual events.")

    with open(experimentdictpath,"r") as f:
        experimentdata = json.load(f)

    localpath = experimentdata["trace_directory"]

    experiments = {f["ExperimentName"]:f for f in experimentdata["traces_expected"]}

    for experiment in experiments:
        base_virgin_dists = []
        sd_virgin_dists = []
        base_dam_dists = []
        sd_dam_dists = []
        tmp_virgin_dists = []
        tmp_sd_virgin_dists = []
        tmp_dam_dists = []
        tmp_sd_dam_dists = []

        intervallist = []

        experimentinfo = experiments[experiment]
        pursuitpath = os.path.join(localpath,experiment,"pursuits")
        partpaths = [os.path.join(pursuitpath,d) for d in os.listdir(pursuitpath) if os.path.isdir(os.path.join(pursuitpath,d))]
        for partpath in partpaths:
            part = os.path.basename(partpath).split("part")[-1]

            segmentpaths = [os.path.join(partpath,d) for d in os.listdir(partpath) if os.path.isdir(os.path.join(partpath,d))]
            for segmentpath in segmentpaths:
                segment = os.path.basename(segmentpath).split("interval")[-1].split("_")
                dictdata = joblib.load(os.path.join(segmentpath,"emd_dists"))
                base_virgin_dists.append(dictdata["base"]["virgin"])
                sd_virgin_dists.append(dictdata["sd"]["virgin"])
                tmp_virgin_dists.append(dictdata["tmp"]["virgin"])
                tmp_sd_virgin_dists.append(dictdata["tmp_sd"]["virgin"])
                base_dam_dists.append(dictdata["base"]["dam"])
                sd_dam_dists.append(dictdata["sd"]["dam"])
                tmp_dam_dists.append(dictdata["tmp"]["dam"])
                tmp_sd_dam_dists.append(dictdata["tmp_sd"]["dam"])

                #intervallist.append("{}-{}".format(int(part)*36000+int(segment[0]),int(part)*36000+int(segment[1])))
                framecount = int(part)*36000+int(segment[0])
                seccount = framecount/30

                intervallist.append("{}".format(str(seccount)[:5]))
                
        raw_color = "red"
        sd_color = "blue"

        raw_patch = mpatches.Patch(color = raw_color)
        sd_patch = mpatches.Patch(color = sd_color)
        label = ["Raw Contour","SD Filtered Contour"]
        fig,ax = plt.subplots(4,1,figsize = (10,7),sharey = True)
        parts = ax[0].violinplot(base_virgin_dists)
        for pc in parts["bodies"]:
            pc.set_facecolor(raw_color)
            pc.set_edgecolor(raw_color)
        ax[0].legend([raw_patch,sd_patch],label)
        parts = ax[0].violinplot(sd_virgin_dists)
        for pc in parts["bodies"]:
            pc.set_facecolor(sd_color)
            pc.set_edgecolor(sd_color)
        parts = ax[1].violinplot(tmp_virgin_dists)
        for pc in parts["bodies"]:
            pc.set_facecolor(raw_color)
            pc.set_edgecolor(raw_color)
        parts = ax[1].violinplot(tmp_sd_virgin_dists)
        for pc in parts["bodies"]:
            pc.set_facecolor(sd_color)
            pc.set_edgecolor(sd_color)
        parts = ax[2].violinplot(base_dam_dists)
        for pc in parts["bodies"]:
            pc.set_facecolor(raw_color)
            pc.set_edgecolor(raw_color)
        parts = ax[2].violinplot(sd_dam_dists)
        for pc in parts["bodies"]:
            pc.set_facecolor(sd_color)
            pc.set_edgecolor(sd_color)
        parts = ax[3].violinplot(tmp_dam_dists)
        for pc in parts["bodies"]:
            pc.set_facecolor(raw_color)
            pc.set_edgecolor(raw_color)
        parts = ax[3].violinplot(tmp_sd_dam_dists)
        for pc in parts["bodies"]:
            pc.set_facecolor(sd_color)
            pc.set_edgecolor(sd_color)
        ax[0].set_title("EMD Distance for virgin contours (template)")
        ax[1].set_title("EMD Distance for virgin contours (temporal)")
        ax[2].set_title("EMD Distance for dam contours (template)")
        ax[3].set_title("EMD Distance for dam contours (temporal)")
        for axis in ax:
            set_axis_style(axis,intervallist)
        


        plt.legend()
        plt.suptitle("EMD Distances for {}".format(experiment),y = 1.0)
        plt.tight_layout()
        script_doc_utils.save_and_insert_image(md,fig,"../docs/script_docs/images/EMD_DISTANCES_{}.png".format(experiment))

    md.new_paragraph("These graphs communciate to us an approximate metric of pursuit quality. In the example pursuit, it was shown that a distance of around 0.001 is the threshold for a valid-looking pursuit. The expectation would be that pursuits that are below this threshold do not need too much further processing. ")

    md.create_md_file()



