### Script to filter pursuits by distance and velocity spikes. 
## Assumes that all pursuits have already been extracted into experiment specific directories, using the extract_pursuits script. 
from social_pursuit import data
from analysis_params import distance_threshold,velocity_threshold,metadata_path

if __name__ == "__main__":
    ## Determine parameters for filtering here: 

    ## Name the filter
    filtername = "velocity_{v}_distance_{d}_filter".format(v = velocity_threshold,d = distance_threshold)
    ## Load Metadata
    experiment_data = data.PursuitTraces(metadata_path)
    ## Prepare filter functions:
    distance_func = lambda data: experiment_data.filter_distance(distance_threshold,data)
    velocity_func = lambda data: experiment_data.filter_velocity(velocity_threshold,data)
    full_filter = lambda data: distance_func(data) and velocity_func(data)

    ## Apply this filter to all experiments: 
    for exp in experiment_data.experiments:
        experiment_data.filter_pursuits(exp,full_filter,filtername)




