### Generate plots for filtered pursuits. 
from social_pursuit import data
from analysis_params import distance_threshold,velocity_threshold,metadata_path

if __name__ == "__main__":
    ## Name the filter
    filtername = "velocity_{v}_distance_{d}_filter".format(v = velocity_threshold,d = distance_threshold)
    ## Load Metadata
    experiment_data = data.PursuitTraces(metadata_path)
    for exp in experiment_data.experiments:
        print(exp)
        experiment_data.plot_trajectories(exp,filtername = filtername)

