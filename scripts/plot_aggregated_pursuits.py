### Generate plots for filtered pursuits. 
import os
from social_pursuit import data
from analysis_params import distance_threshold,velocity_threshold,metadata_path

if __name__ == "__main__":
    ## Name the filter
    filtername = "velocity_{v}_distance_{d}_filter".format(v = velocity_threshold,d = distance_threshold)
    location = "../docs/meeting_reports/images"
   
    ## Load Metadata
    experiment_data = data.PursuitTraces(metadata_path)
    for exp,edict in experiment_data.experiments.items():
        boxes= experiment_data.get_boxes(exp) 
        parts = edict["nb_trace_sets"]
        for r in boxes:
            path = "{}_Aggregate_Pursuit_Plots_ROI_{}".format(exp,r)
            savepath = os.path.join(location,path) 
            data.mkdir_notexists(savepath)
            for p in range(parts):
                filepath = os.path.join(savepath,"AGGREGATE_PART_{}".format(p))
                experiment_data.plot_all_trajectories_in_traceset(exp,r,p,filtername = filtername,plotpath = filepath)
                vledfilepath = os.path.join(savepath,"AGGREGATE_PART_{}vled".format(p))
                experiment_data.plot_vled_trajectories_in_traceset(exp,r,p,filtername = filtername,plotpath = vledfilepath)
                mledfilepath = os.path.join(savepath,"AGGREGATE_PART_{}mled".format(p))
                experiment_data.plot_mled_trajectories_in_traceset(exp,r,p,filtername = filtername,plotpath = mledfilepath)

        

