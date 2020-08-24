from social_pursuit import data

if __name__ == "__main__":
    tracedata = data.PursuitTraces("../../src/social_pursuit/trace_template.json")
    ## Plot an example trajectory:
    experimentname = "TempTrial2"
    roi = 1
    p = 0
    interval = [41435, 41466]
    filtername = "velocity_20_distance_30_filter"
    plotpath = "./images/example_trace1.png"
    tracedata.plot_trajectory(experimentname,roi,p,interval,filtername,plotpath)

    ## Check one thing
    experimentname = "TempTrial17"
    roi = 2
    p = 7
    interval = [46058, 46092]
    filtername = "velocity_20_distance_30_filter"
    tracedata = tracedata.load_trace_data(experimentname,roi,p,interval,filtername)
    print(tracedata["pursuit_direction"])
