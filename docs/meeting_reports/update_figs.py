from social_pursuit import data

if __name__ == "__main__":
    tracedata = data.PursuitTraces("../../src/social_pursuit/trace_template.json")
    ## Plot an example trajectory:
    experimentname = "TempTrial2"
    roi = 1
    p = 0
    interval = [70567, 70630]
    filtername = "velocity_20_distance_30_filter"
    plotpath = "./images/example_trace1.png"
    tracedata.plot_trajectory(experimentname,roi,p,interval,filtername,plotpath)
