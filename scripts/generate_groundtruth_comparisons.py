### Generate groundtruth comparisons for all datasets that have annotated groundtruth data.  
import os
from social_pursuit import data
from analysis_params import distance_threshold,velocity_threshold,metadata_path
from mdutils.mdutils import MdUtils

docs = "../"
if __name__ == "__main__":
    filternames = ["groundtruth"]
    pt = data.PursuitTraces(metadata_path)

    for exp in pt.experiments:
        edict = pt.experiments[exp]
        if edict.get("GroundTruth",None):
            pt.process_groundtruth(edict["ExperimentName"])
            pt.compare_pursuits(edict["ExperimentName"],filternames)


