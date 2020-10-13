### Not run, putting down this function for reproducibility. 
#8/13/2020
## Script to extract all pursuits, and put them in experiment indexed directories.  
from social_pursuit import data

if __name__ == "__main__":
    metadata = data.PursuitTraces("../tests/test_fixtures/trace_template.json")
    for experiment in metadata.experiments:
        metadata.get_pursuits(experiment["ExperimentName"])


