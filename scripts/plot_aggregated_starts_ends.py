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
            mspath = "MLED_STARTS"
            mepath = "MLED_ENDS"
            vspath = "VLED_STARTS"
            vepath = "VLED_ENDS"

            experiment_data.plot_all_virgin_starts_in_experiment(exp,r,os.path.join(savepath,vspath),filtername=filtername)
            experiment_data.plot_all_mother_starts_in_experiment(exp,r,os.path.join(savepath,mspath),filtername=filtername)
            experiment_data.plot_all_virgin_ends_in_experiment(exp,r,os.path.join(savepath,vepath),filtername=filtername)
            experiment_data.plot_all_mother_ends_in_experiment(exp,r,os.path.join(savepath,mepath),filtername=filtername)

        

