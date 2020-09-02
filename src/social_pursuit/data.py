import json 
import re
import os
import yaml
from datetime import datetime
from scipy.io import loadmat  
from scipy.ndimage.morphology import binary_dilation
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from moviepy.editor import VideoFileClip
import pandas as pd
import pathlib
#import boto3
#from botocore.exceptions import ClientError

#s3_client = boto3.client("s3")

package_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(package_dir,"local_paths.json"),"r") as f:
    pathdict = json.load(f)

def mkdir_notexists(foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)

def transfer_if_not_found(s3_path):
    """
    function to transfer a video file from s3 to a local drive if it cannot be found in that drive already.
    """
    filename = os.path.basename(s3_path)
    localname = os.path.join(pathdict["tmp_storage_loc"],filename)
    exists = filename in os.listdir(pathdict["tmp_storage_loc"])
    if not exists:
        split_path = s3_path.split("s3://")[1].split("/")
        
        bucketname = split_path[0]
        objectname = "/".join(split_path[1:])
        print(bucketname,objectname)
        
        try:
            s3_client.download_file(bucketname,objectname,localname)
        except ClientError:
            raise Exception("Object could not be downloaded.")
    else:
        print("already exists in tmpdir.")
    return localname


def find_segments(indices):
    """Helper function to find segments in binarized data. 

    :param indices: binarized indices indicating occurence of an event or not. 

    """
    differences = np.diff(indices)
    all_intervals = []
    ## Initialize with the first element added:
    interval = []
    interval.append(indices[0])
    for i,diff in enumerate(differences):
        if diff == 1:
            pass # interval not yet over
        else:
            # last interval ended
            if interval[0] == indices[i]:
                interval.append(indices[i]+1)
            else:
                interval.append(indices[i])
            all_intervals.append(interval)
            # start new interval
            interval = [indices[i+1]]
        if i == len(differences)-1:
            interval.append(indices[-1]+1)
            all_intervals.append(interval)
    return all_intervals

## Class for keeping data organized. 
class PursuitTraces(object):
    """PursuitTraces."""

    """
    Data class for traces of behavioral pursuit. Initialized from json files providing metadata about the data location. Assumes that all data (traces, video, configs, human labels) are in one pile under the "trace_directory".  

    :param filename: the name of the file where the trace metadata is stored.  
    """
    def __init__(self,filename):
        """Initialzation for this class takes in

        """
        self.metadata = self.__load_spec(filename)
        self.trace_suffix = "DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat"
        self.trace_length_frames = 72000
        clean_experiments = self.__check_data_integrity()
        self.experiments = clean_experiments

    def __load_spec(self,filename):
        """
        Load specification file. Include linting to make sure this is the right file later. 
        """
        with open(filename,"r") as f:
            spec = json.load(f)
        return spec

    def __load_config(self,configname):
        """
        Load the configuration file corresponding to a particular experiment. 
        """
        with open(configname,"r") as f:
            try:
                yaml_content = yaml.safe_load(f)
            except yaml.scanner.ScannerError as ve:
                raise ValueError("Yaml is misformatted.")
        return yaml_content

    def __check_data_integrity(self):
        """Check that the videos, configuration files, and traces are included as expected. 

        """
        try:
            self.metadata
        except AttributeError:
            print("metadata does not exist.")
            raise
        all_traces = self.metadata["traces_expected"]

        path = self.metadata["trace_directory"]
        self.path = pathlib.Path(path)
        ## get all files in the trace directory. 
        all_contents = os.listdir(path)
        self.all_contents = all_contents

        cleared_experiments = {} 
        all_resources = ["config","video","traces","groundtruth"]
        for trace_dict in all_traces:
            config_exists = self.__check_config(trace_dict,all_contents)
            video_exists = self.__check_video(trace_dict,all_contents)
            traces_exist,trace_nb = self.__check_traces(trace_dict,all_contents)
            gt_exists = self.__check_gt(trace_dict,all_contents)
            all_conditions = [config_exists,video_exists,traces_exist,gt_exists]
            approved = all(all_conditions)
            if approved:
                trace_dict["nb_trace_sets"] = trace_nb
                trace_dict["config"] = self.__load_config(os.path.join(self.path,trace_dict["ConfigName"]))
                cleared_experiments[trace_dict["ExperimentName"]] = trace_dict
            else:
                "Name missing resources:"
                missing_resources = [a for i,a in enumerate(all_resources) if all_conditions[i] is False]
                print("ATTENTION: Experiment {e} has errors with the following resources: {l}. It will not be included in further analysis.".format(e = trace_dict["ExperimentName"],l = missing_resources))

        return cleared_experiments

    def __check_config(self,trace_dict,all_contents):
        """
        Simple condition to check if config is in the specified directory. 
        """
        cond = trace_dict["ConfigName"] in all_contents
        return cond
    
    def __check_video(self,trace_dict,all_contents):
        """
        Simple condition to check if video is in the specified directory. 
        """
        cond = trace_dict["VideoName"] in all_contents
        return cond

    def __check_traces(self,trace_dict,all_contents):
        """
        Condition to check if a contiguous series of traces for each roi are included. Depends upon the config existing.  
        Right now just checks that there are the same number of traces for each roi, does not account for dropped traces in all rois or 
        the case where there are an equivalent number of drops in each roi.
        """
        if self.__check_config(trace_dict,all_contents):
            ## Get config
            config = self.__load_config(os.path.join(self.path,trace_dict["ConfigName"]))
            ## Get number of boxes we expect from config. 
            box_indices = self.__get_box_indices(config)
            ## Get the video files for each of these boxes and ensure that there are an equal number of each. 
            roi_lambdas = [lambda p: self.__check_trace_function(trace_dict["ExperimentName"],r,p) for r in box_indices]
            per_lambda = []
            for l in roi_lambdas:
                out = filter(l,all_contents)
                per_lambda.append(list(out))
            lens = [len(p) for p in per_lambda]
            result = all(e == lens[0] for e in lens)
            return result,lens[0]
        else:
            return False,None

    def __check_gt(self,trace_dict,all_contents):
        """Condition to check if the ground truth xlsx file indicating manually scored shepherding events is included.  
        Only returns false if the path to a groundtruth file is indicated in the experiment metadata, but is not included. 
        """
        try:
            gt_name = trace_dict["GroundTruth"]
        except KeyError:
            print("groundtruth not provided for experiment {f}".format(f=trace_dict["ExperimentName"]))
            return True
        cond = gt_name in all_contents
        return cond


    def get_boxes(self,experimentname):
        """public function to get box numbers. 

        """
        config = self.experiments[experimentname]["config"]
        return self.__get_box_indices(config)

    def __get_box_indices(self,config):
        """
        helper function to get the roi indices from a config file. 
        """
        boxes = config["coordinates"].keys()
        box_indices = [int(list(filter(str.isdigit,b))[0]) for b in boxes]
        return box_indices
    
    def __check_trace_function(self,experiment_name,roi_index,pathname):
        """ Function to be curried when checking for existence of data files. 

        :param experiment_name: the name of the experiment that we are looking for. Will be used to prefix the string we are looking for. 
        :param roi_index: the integer giving the roi within the video that shows the mice we are observing.  
        :param pathname: the name of the path that we are evaluating against the condition given by first two functions. 
        """
        formatted_prefix = "{e}roi_{r}".format(e=experiment_name,r=roi_index)

        prefix = pathname.startswith(formatted_prefix)
        suffix = pathname.endswith(".mat")
        cond = prefix and suffix
        return cond
        
    def get_tracedicts(self,experiment_names=None):
        """ Function to load the data dictionaries from the matlab files. should be done on demand to preserve memory. 
       
        :param experiment_name: a list containing the experiment names you want to fetch data for. If not given, will be all. 
        """
        ## Check that all exist. 
        if experiment_names is None:
            query_experiments = self.experiments
        else:
            assert type(experiment_names) is list,"experiment names must be list"
            query_experiments = {}
            for e in experiment_names:
                query_experiments[e] = self.experiments[e]
        ## Now load. 
        tracedicts_raw = {}
        for e,edict in query_experiments.items():
            ## Get all variables that we need to iterate over:  
            parts = edict["nb_trace_sets"]
            rois = self.__get_box_indices(edict["config"])
            tracenames = {"{e}roi_{r}cropped_part{p}".format(e=edict["ExperimentName"],r=r,p=p)+self.trace_suffix:(r,p) for r in rois for p in range(parts)}
            tracedict = {}
            print(tracenames)
            for tracename,indices in tracenames.items():
                trace_path = os.path.join(self.path,tracename)
                data = loadmat(trace_path)
                tracedict[indices] = data
            tracedicts_raw[e] = tracedict
        return tracedicts_raw

    def get_pursuits(self,experiment_name):
        """Function to extract all of the pursuit events in a single experiment, and save them as individual files.  
        Within each file, we should specify the experiment, roi, video part, trajectory of virgin, trajectory of dam (centroids), nose tip of virgin, nost tip of dam, 
        and direction of pursuit. The name of the file should be the timestamp of event onset, and it should be saved in folders specifying the experiment and roi.  

        :param experiment_name: a list containing the experiment names you want to fetch data for. If not given, will be all. 
        """
        ## Vet input
        assert type(experiment_name) is str,"experiment name must be string."
        try:
            edict = self.experiments[experiment_name]
        except KeyError:
            print("specified experiment name does not exist or was not cleared for analysis.")
            raise
        ## Now get total number of rois and parts:
        parts = edict["nb_trace_sets"]
        rois = self.__get_box_indices(edict["config"])

        ## make folders if they don't exist: 
        pursuit_root_dir = os.path.join(self.path,"{}_Pursuit_Events".format(experiment_name))
        mkdir_notexists(pursuit_root_dir)

        for r in rois:
            roi_dir = "ROI_{}".format(r)
            roi_fullpath = os.path.join(pursuit_root_dir,roi_dir)
            mkdir_notexists(roi_fullpath)
            for p in range(parts):
                part_dir = "PART_{}".format(p)
                part_fullpath = os.path.join(roi_fullpath,part_dir)
                mkdir_notexists(part_fullpath)
                self.get_pursuit_traceset(edict,r,p,part_fullpath)
    
    def get_intervaldict(self,dataset,interval,r,p,experimentname):
        """Helper function to get the actual trajectory information given a dataset and relevant specification info. 

        :param dataset: dataset dictionary containing the whole 40 minute trace. 
        :param interval: interval to extract.
        :param r: roi
        :param p: part of the video. 
        :param experimentname: name of the experiment we are extracting. 

        """
        intervaldict = {}
        indexslice = slice(*interval)

        mtraj = dataset["dam_centroid_traj"][indexslice,:]
        vtraj = dataset["virgin_centroid_traj"][indexslice,:]
        mtip = dataset["dam_tip_traj"][indexslice,:]
        vtip = dataset["virgin_tip_traj"][indexslice,:]
        pursuit_direction = dataset["pursuit_direction"][:,indexslice]
        pursuit_direction_agg = np.sign(np.sum(pursuit_direction))

        intervaldict["ExperimentName"] = experimentname
        intervaldict["ROI"] = r
        intervaldict["VideoPart"] = p
        intervaldict["Interval"] = interval
        intervaldict["mtraj"] = mtraj
        intervaldict["vtraj"] = vtraj
        intervaldict["mtip"] = mtip
        intervaldict["vtip"] = vtip
        intervaldict["pursuit_direction"] = pursuit_direction
        intervaldict["pursuit_direction_agg"] = pursuit_direction_agg
        return intervaldict,pursuit_direction_agg

    def get_pursuit_traceset(self,edict,r,p,part_fullpath):
        """Function to extract all pursuit events in a single trace set. 
        
        :param edict: experiment dictionary, should exist as an internal variable of the instance. 
        :param r: integer providing the roi we are looking at. 
        :param p: integer providing the experimental chunk we care about.  
        """
        tracename = "{e}roi_{r}cropped_part{p}".format(e=edict["ExperimentName"],r=r,p=p)+self.trace_suffix
        dataset = loadmat(os.path.join(self.path,tracename))
        ## Now within the dataset we want to index by each event:  
        pursuit_times = dataset["pursuit_times"]
        if len(pursuit_times) == 0:
            pass
        else:
            ## Convert pursuit times to data indices: 
            pursuit_frames =  [self.convert_time(pt,p) for pt in pursuit_times]
            intervals = self.extract_segments(pursuit_frames,5)
            ## Now get relevant data in these indices: 
            
            for interval in intervals:
                intervaldict,pdir = self.get_intervaldict(dataset,interval,r,p,experimentname=edict["ExperimentName"]) 
                #intervaldict = {}
                #indexslice = slice(*interval)

                #mtraj = dataset["dam_centroid_traj"][indexslice,:]
                #vtraj = dataset["virgin_centroid_traj"][indexslice,:]
                #mtip = dataset["dam_tip_traj"][indexslice,:]
                #vtip = dataset["virgin_tip_traj"][indexslice,:]
                #pursuit_direction = dataset["pursuit_direction"][:,indexslice]
                #pursuit_direction_agg = np.sign(np.sum(pursuit_direction))

                #intervaldict["ExperimentName"] = edict["ExperimentName"]
                #intervaldict["ROI"] = r
                #intervaldict["VideoPart"] = p
                #intervaldict["Interval"] = interval
                #intervaldict["mtraj"] = mtraj
                #intervaldict["vtraj"] = vtraj
                #intervaldict["mtip"] = mtip
                #intervaldict["vtip"] = vtip
                #intervaldict["pursuit_direction"] = pursuit_direction
                #intervaldict["pursuit_direction_agg"] = pursuit_direction_agg
                dict_name = "Pursuit{I}_Direction{P}.npy".format(I = interval,P = pdir)

                np.save(os.path.join(part_fullpath,dict_name),intervaldict)


    def convert_time(self,pt,p):
        """helper function to convert time from a dt string to frames. 

        :param pt: string datetime. 
        :param p: video part. Assume each video except last is of length self.trace_length_frames
        """
        timeformat1 = "%H:%M:%S.%f"
        timeformat0 = "%H:%M:%S       "
        basetime = datetime.strptime("00:00:00.000000",timeformat1)
        pt = "0"+pt
        if pt[-1] == " ":
            dtime = datetime.strptime(pt,timeformat0)
        else:
            dtime = datetime.strptime(pt,timeformat1)
        time = (dtime-basetime).total_seconds()
        time_frames = time*30
        offset = p*self.trace_length_frames
        pursuit_time_baseline_subtracted = int(np.round(time_frames-(p*self.trace_length_frames)))
        return pursuit_time_baseline_subtracted

    def extract_segments(self,pursuit_frames,dilation_length):
        """helper function to perform binary dilation on a set of pursuit events, and then select out individual segments where pursuit happened. 

        """
        ## First binarize the pursuit frames:
        pursuits = np.zeros(self.trace_length_frames,)
        pursuits[np.array(pursuit_frames)] = 1

        ## Second elongate each event:
        struct = np.ones(dilation_length*2+1,)
        buffered = binary_dilation(pursuits,struct).astype(int)

        ## Third convert back to indices:
        output = np.where(buffered)[0]

        ## Now extract unique events:  
        all_intervals = find_segments(output)
        return all_intervals

    def plot_trajectory(self,experimentname,r,p,interval,filtername = None,plotpath = None):
        """Imports the data at the npy file indicated by arguments, and plots the trajectories it finds within

        :param experimentname: name of the experiment we will be analyzing.
        :param r: Integer giving the roi we are interested in.
        :param p: Integer giving the video part we are interested in.
        :param interval: Interval of frames that define this event. 
        :param filtername: If you applied a filter, looks in that filter directory to plot. 
        :param plotpath: Specify a location to plot to if you have one in mind. Defaults to the original data location, in an appropriate subfolder. 
        """
        if filtername is None:
            tracedirectory = self.get_trace_dir(experimentname,r,p)
            data = self.load_trace_data(experimentname,r,p,interval)
        else:
            tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)
            data = self.load_trace_data(experimentname,r,p,interval,filtername)
        plotsdir= os.path.join(tracedirectory,"plots")
        mkdir_notexists(plotsdir)
        ## We will assume that all traces are relative to the enclosure coordinates, so coordinates should be relative. 
        if plotpath is None:
            plotpath = os.path.join(plotsdir,"Pursuit{}_Direction{}".format(interval,data["pursuit_direction_agg"])+"plot.png")
        else:
            pass
        self.plot_trajectory_from_data(experimentname,r,p,data,plotpath)

    def plot_trajectories_traceset(self,experimentname,r,p,filtername = None):
        """Imports data for each pursuit detected in a traceset, and plots all the trajectories found there.  
        
        :param experimentname: name of the experiment we will be analyzing.
        :param r: Integer giving the roi we are interested in.
        :param p: Integer giving the video part we are interested in.
        :param interval: Interval of frames that define this event. 
        :param filtername: If you applied a filter, looks in that filter directory to plot. 
        """
        if filtername is None:
            tracedirectory = self.get_trace_dir(experimentname,r,p)
        else:
            tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

        plotsdir = os.path.join(tracedirectory,"plots")
        mkdir_notexists(plotsdir)

        all_contents = os.listdir(tracedirectory)
        all_pursuits = [a for a in all_contents if a.endswith(".npy")]
        for pursuit in all_pursuits:
            plotname = os.path.splitext(pursuit)[0]+".png"
            plotpath = os.path.join(plotsdir,plotname)
            datapath = os.path.join(tracedirectory,pursuit)
            data = np.load(datapath,allow_pickle = True)[()]
            self.plot_trajectory_from_data(experimentname,r,p,data,plotpath)

    def plot_trajectories(self,experimentname,filtername = None):
        """Plots trajectories for all pursuits in a given experiment. 
        """
        ## Vet input
        assert type(experimentname) is str,"experiment name must be string."
        try:
            edict = self.experiments[experimentname]
        except KeyError:
            print("specified experiment name does not exist or was not cleared for analysis.")
            raise
        ## Now get total number of rois and parts:
        parts = edict["nb_trace_sets"]
        rois = self.__get_box_indices(edict["config"])
        for r in rois:
            for p in range(parts):
                self.plot_trajectories_traceset(experimentname,r,p,filtername)

    def plot_trajectory_from_data(self,experimentname,r,p,data,plotpath):
        """Plots a trajectory set from given data.   

        """
        ## Get config: 
        try:
            config = self.experiments[experimentname]["config"]
        except KeyError:
            print("given experiment {} does not exist or is not clean.".format(experimentname))
            raise

        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]

        ## Get box and nest info: 
        dims = self.get_boxdims(config,r)
        left,bottom,width,height = self.get_nestcoords(config,r)

        fig,ax = plt.subplots()
        ax.set_xlim([0,dims[0]])
        ax.set_ylim([dims[1],0])

        ## Start Rendering
        nest = plt.Rectangle((left,bottom),width,height)

        ax.add_patch(nest)

        plt.plot(mtraj[:,0],mtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "red",label="dam")
        plt.plot(vtraj[:,0],vtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "blue",label="virgin")
        plt.plot(mtraj[0,0],mtraj[0,1],"ro")
        plt.plot(mtraj[-1,0],mtraj[-1,1],"rx")
        plt.plot(vtraj[0,0],vtraj[0,1],"bo")
        plt.plot(vtraj[-1,0],vtraj[-1,1],"bx")
        plt.legend()
        plt.savefig(plotpath)
        plt.close()

    def plot_trajectories_from_filenames(self,experimentname,r,p,filenames,plotpath,plotfunc = None):
        ## Get config: 
        try:
            config = self.experiments[experimentname]["config"]
        except KeyError:
            print("given experiment {} does not exist or is not clean.".format(experimentname))
            raise

        fig,ax = plt.subplots()
        ## Get box and nest info: 
        dims = self.get_boxdims(config,r)
        left,bottom,width,height = self.get_nestcoords(config,r)

        ax.set_xlim([0,dims[0]])
        ax.set_ylim([dims[1],0])

        ## Start Rendering
        nest = plt.Rectangle((left,bottom),width,height)

        ax.add_patch(nest)

        for f,filename in enumerate(filenames):
            data = np.load(os.path.join(self.path,filename),allow_pickle=True)[()]
            if plotfunc is None:
                self.__plot_func_internal(data,fig,ax,f)
            else: 
                plotfunc(data,fig,ax,f)
        plt.legend()
        plt.savefig(plotpath)
        plt.close()

    def __plot_func_internal(self,data,fig,ax,it):
        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]
        if it == 0:
            ax.plot(mtraj[:,0],mtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "red",label="dam")
            ax.plot(vtraj[:,0],vtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "blue",label="virgin")
        else:
            ax.plot(mtraj[:,0],mtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "red")
            ax.plot(vtraj[:,0],vtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "blue")
        ax.plot(mtraj[0,0],mtraj[0,1],"ro")
        ax.plot(mtraj[-1,0],mtraj[-1,1],"rx")
        ax.plot(vtraj[0,0],vtraj[0,1],"bo")
        ax.plot(vtraj[-1,0],vtraj[-1,1],"bx")

    def __plot_mled_internal(self,data,fig,ax,it):
        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]
        if data["pursuit_direction_agg"] == -1:
            if it == 0:
                ax.plot(mtraj[:,0],mtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "red",label="dam")
                ax.plot(vtraj[:,0],vtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "blue",label="virgin")
            else:
                ax.plot(mtraj[:,0],mtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "red")
                ax.plot(vtraj[:,0],vtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "blue")
            ax.plot(mtraj[0,0],mtraj[0,1],"ro")
            ax.plot(mtraj[-1,0],mtraj[-1,1],"rx")
            ax.plot(vtraj[0,0],vtraj[0,1],"bo")
            ax.plot(vtraj[-1,0],vtraj[-1,1],"bx")
        else:
            pass

    def __plot_vled_internal(self,data,fig,ax,it):
        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]
        if data["pursuit_direction_agg"] == 1:
            if it == 0:
                ax.plot(mtraj[:,0],mtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "red",label="dam")
                ax.plot(vtraj[:,0],vtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "blue",label="virgin")
            else:
                ax.plot(mtraj[:,0],mtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "red")
                ax.plot(vtraj[:,0],vtraj[:,1],marker = "o",markersize = 0.5,linestyle = "None",color = "blue")
            ax.plot(mtraj[0,0],mtraj[0,1],"ro")
            ax.plot(mtraj[-1,0],mtraj[-1,1],"rx")
            ax.plot(vtraj[0,0],vtraj[0,1],"bo")
            ax.plot(vtraj[-1,0],vtraj[-1,1],"bx")
        else:
            pass


    def __plot_starts_internal(self,data,fig,ax,it):
        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]
        if it == 0:
            ax.plot(mtraj[0,0],mtraj[0,1],"ro",label = "dam")
            ax.plot(vtraj[0,0],vtraj[0,1],"bo",label = "virgin")
        else:
            ax.plot(mtraj[0,0],mtraj[0,1],"ro")
            ax.plot(vtraj[0,0],vtraj[0,1],"bo")

    def __plot_ends_internal(self,data,fig,ax,it):
        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]
        if it == 0:
            ax.plot(mtraj[-1,0],mtraj[-1,1],"rx",label = "dam")
            ax.plot(vtraj[-1,0],vtraj[-1,1],"bx",label = "virgin")
        else:
            ax.plot(mtraj[-1,0],mtraj[-1,1],"rx")
            ax.plot(vtraj[-1,0],vtraj[-1,1],"bx")

    def __plot_starts_vled_internal(self,data,fig,ax,it):
        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]
        if data["pursuit_direction_agg"] == 1:
            if it == 0:
                ax.plot(mtraj[0,0],mtraj[0,1],"ro",label = "dam")
                ax.plot(vtraj[0,0],vtraj[0,1],"bo",label = "virgin")
            else:
                ax.plot(mtraj[0,0],mtraj[0,1],"ro")
                ax.plot(vtraj[0,0],vtraj[0,1],"bo")
        else:
            pass

    def __plot_ends_vled_internal(self,data,fig,ax,it):
        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]
        if data["pursuit_direction_agg"] == 1:
            if it == 0:
                ax.plot(mtraj[-1,0],mtraj[-1,1],"rx",label = "dam")
                ax.plot(vtraj[-1,0],vtraj[-1,1],"bx",label = "virgin")
            else:
                ax.plot(mtraj[-1,0],mtraj[-1,1],"rx")
                ax.plot(vtraj[-1,0],vtraj[-1,1],"bx")
        else:
            pass

    def __plot_starts_mled_internal(self,data,fig,ax,it):
        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]
        if data["pursuit_direction_agg"] == -1:
            if it == 0:
                ax.plot(mtraj[0,0],mtraj[0,1],"ro",label = "dam")
                ax.plot(vtraj[0,0],vtraj[0,1],"bo",label = "virgin")
            else:
                ax.plot(mtraj[0,0],mtraj[0,1],"ro")
                ax.plot(vtraj[0,0],vtraj[0,1],"bo")
        else:
            pass

    def __plot_ends_mled_internal(self,data,fig,ax,it):
        ## Get individual trajectories
        mtraj = data["mtraj"]
        vtraj = data["vtraj"]
        if data["pursuit_direction_agg"] == -1:
            if it == 0:
                ax.plot(mtraj[-1,0],mtraj[-1,1],"rx",label = "dam")
                ax.plot(vtraj[-1,0],vtraj[-1,1],"bx",label = "virgin")
            else:
                ax.plot(mtraj[-1,0],mtraj[-1,1],"rx")
                ax.plot(vtraj[-1,0],vtraj[-1,1],"bx")

    def plot_all_trajectories_in_traceset(self,experimentname,r,p,filtername = None,plotpath = None):
        """ Plot all trajectories in the same traceset into a single frame.

        """
        if filtername is None:
            tracedirectory = self.get_trace_dir(experimentname,r,p)
        else:
            tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

        if plotpath is None:
            plotsdir = os.path.join(tracedirectory,"plots")
            plotpath = os.path.join(plotsdir,"Aggregate_Plot_ROI_{r}_PART_{p}".format(r=r,p=p))
            mkdir_notexists(plotsdir)
        else:
            pass
        all_contents = os.listdir(tracedirectory)
        all_pursuits = [os.path.join(tracedirectory,a) for a in all_contents if a.endswith(".npy")]
        self.plot_trajectories_from_filenames(experimentname,r,p,all_pursuits,plotpath)

    def plot_vled_trajectories_in_traceset(self,experimentname,r,p,filtername = None,plotpath = None):
        """ Plot all trajectories in the same traceset into a single frame.

        """
        if filtername is None:
            tracedirectory = self.get_trace_dir(experimentname,r,p)
        else:
            tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

        if plotpath is None:
            plotsdir = os.path.join(tracedirectory,"plots")
            plotpath = os.path.join(plotsdir,"Aggregate_Plot_ROI_{r}_PART_{p}_vled".format(r=r,p=p))
            mkdir_notexists(plotsdir)
        else:
            pass
        all_contents = os.listdir(tracedirectory)
        all_pursuits = [os.path.join(tracedirectory,a) for a in all_contents if a.endswith(".npy")]
        self.plot_trajectories_from_filenames(experimentname,r,p,all_pursuits,plotpath,self.__plot_vled_internal)

    def plot_mled_trajectories_in_traceset(self,experimentname,r,p,filtername = None,plotpath = None):
        """ Plot all trajectories in the same traceset into a single frame.

        """
        if filtername is None:
            tracedirectory = self.get_trace_dir(experimentname,r,p)
        else:
            tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

        if plotpath is None:
            plotsdir = os.path.join(tracedirectory,"plots")
            plotpath = os.path.join(plotsdir,"Aggregate_Plot_ROI_{r}_PART_{p}_mled".format(r=r,p=p))
            mkdir_notexists(plotsdir)
        else:
            pass
        all_contents = os.listdir(tracedirectory)
        all_pursuits = [os.path.join(tracedirectory,a) for a in all_contents if a.endswith(".npy")]
        self.plot_trajectories_from_filenames(experimentname,r,None,all_pursuits,plotpath,self.__plot_mled_internal)

    def plot_all_starts_in_traceset(self,experimentname,r,p,filtername = None,plotpath = None):
        """ Plot all trajectories in the same interval into a single frame.

        """
        if filtername is None:
            tracedirectory = self.get_trace_dir(experimentname,r,p)
        else:
            tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

        if plotpath is None:
            plotsdir = os.path.join(tracedirectory,"plots")
            plotpath = os.path.join(plotsdir,"Aggregate_Plot_ROI_{r}_PART_{p}STARTS".format(r=r,p=p))
            mkdir_notexists(plotsdir)
        else:
            pass
        all_contents = os.listdir(tracedirectory)
        all_pursuits = [os.path.join(tracedirectory,a) for a in all_contents if a.endswith(".npy")]
        self.plot_trajectories_from_filenames(experimentname,r,p,all_pursuits,plotpath,self.__plot_starts_internal)

    def plot_all_ends_in_traceset(self,experimentname,r,p,filtername = None,plotpath = None):
        """ Plot all trajectories in the same interval into a single frame.

        """
        if filtername is None:
            tracedirectory = self.get_trace_dir(experimentname,r,p)
        else:
            tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

        if plotpath is None:
            plotsdir = os.path.join(tracedirectory,"plots")
            plotpath = os.path.join(plotsdir,"Aggregate_Plot_ROI_{r}_PART_{p}ENDS".format(r=r,p=p))
            mkdir_notexists(plotsdir)
        else:
            pass
        all_contents = os.listdir(tracedirectory)
        all_pursuits = [os.path.join(tracedirectory,a) for a in all_contents if a.endswith(".npy")]
        self.plot_trajectories_from_filenames(experimentname,r,p,all_pursuits,plotpath,self.__plot_ends_internal)

    def plot_all_virgin_ends_in_experiment(self,experimentname,r,plotpath,filtername = None):
        """ Plot all trajectories in the same interval into a single frame.

        """
        all_pursuits = []

        for p in range(self.experiments[experimentname]["nb_trace_sets"]):
            if filtername is None:
                tracedirectory = self.get_trace_dir(experimentname,r,p)
            else:
                tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

            p_contents = os.listdir(tracedirectory)
            p_pursuits = [os.path.join(tracedirectory,a) for a in p_contents if a.endswith(".npy")]
            all_pursuits = all_pursuits+p_pursuits
        self.plot_trajectories_from_filenames(experimentname,r,None,all_pursuits,plotpath,self.__plot_ends_vled_internal)

    def plot_all_mother_ends_in_experiment(self,experimentname,r,plotpath,filtername = None):
        """ Plot all trajectories in the same interval into a single frame.

        """
        all_pursuits = []

        for p in range(self.experiments[experimentname]["nb_trace_sets"]):
            if filtername is None:
                tracedirectory = self.get_trace_dir(experimentname,r,p)
            else:
                tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

            p_contents = os.listdir(tracedirectory)
            p_pursuits = [os.path.join(tracedirectory,a) for a in p_contents if a.endswith(".npy")]
            all_pursuits = all_pursuits+p_pursuits
        self.plot_trajectories_from_filenames(experimentname,r,None,all_pursuits,plotpath,self.__plot_ends_mled_internal)

    def plot_all_virgin_starts_in_experiment(self,experimentname,r,plotpath,filtername = None):
        """ Plot all trajectories in the same interval into a single frame.

        """
        all_pursuits = []

        for p in range(self.experiments[experimentname]["nb_trace_sets"]):
            if filtername is None:
                tracedirectory = self.get_trace_dir(experimentname,r,p)
            else:
                tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

            p_contents = os.listdir(tracedirectory)
            p_pursuits = [os.path.join(tracedirectory,a) for a in p_contents if a.endswith(".npy")]
            all_pursuits = all_pursuits+p_pursuits
        self.plot_trajectories_from_filenames(experimentname,r,None,all_pursuits,plotpath,self.__plot_starts_vled_internal)

    def plot_all_mother_starts_in_experiment(self,experimentname,r,plotpath,filtername = None):
        """ Plot all trajectories in the same interval into a single frame.

        """
        all_pursuits = []

        for p in range(self.experiments[experimentname]["nb_trace_sets"]):
            if filtername is None:
                tracedirectory = self.get_trace_dir(experimentname,r,p)
            else:
                tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)

            p_contents = os.listdir(tracedirectory)
            p_pursuits = [os.path.join(tracedirectory,a) for a in p_contents if a.endswith(".npy")]
            all_pursuits = all_pursuits+p_pursuits
        self.plot_trajectories_from_filenames(experimentname,r,None,all_pursuits,plotpath,self.__plot_starts_mled_internal)

    def load_trace_data(self,experimentname,r,p,interval,filtername = None):
        """Aggregates the construction of a data path, and loading of the data under that path 

        :param experimentname: Name of the experiment we will be analyzing
        :param r: Integer giving the roi we are interested in. 
        :param p: Integer giving the video part we are interested in.
        :param interval: Interval of frames that define this event. 
        """
        if filtername is None:
            tracedirectory = self.get_trace_dir(experimentname,r,p)
        else: 
            tracedirectory = self.get_trace_filter_dir(experimentname,r,p,filtername)
        filename = self.get_pursuit_eventname(tracedirectory,interval)
        data = np.load(filename,allow_pickle = True)[()]
        return data

    def get_trace_dir(self,experimentname,r,p):
        tracedirectory = os.path.join(self.path,"{}_Pursuit_Events".format(experimentname),"ROI_{}".format(r),"PART_{}".format(p))
        return tracedirectory

    def get_trace_filter_dir(self,experimentname,r,p,filtername):
        tracedirectory = os.path.join(self.path,"{}_Pursuit_Events".format(experimentname),"FILTER_{f}".format(f=filtername),"ROI_{}".format(r),"PART_{}".format(p))
        return tracedirectory

    def get_pursuit_eventname(self,tracedirectory,interval):
        path = os.path.join(tracedirectory,"Pursuit{i}".format(i=interval)+"_Direction{}.npy")
        vlead = path.format(1.0)
        mlead = path.format(-1.0)
        paths = []
        for candidate in [vlead,mlead]:
            if os.path.exists(candidate):
                paths.append(candidate)
            else:
                pass
        assert len(paths) == 1, "File does not exist, or is not correctly saved."
        filename = paths[0]
        return filename

    def get_boxdims(self,config,r):
        """Get the dimensions of a box with bottom left corner at 0,0. 

        :param config: configuration dictionary used to generate analysis.
        :param r: the index of the box for which we are getting dimensions. 
        """
        boxkey = "box{}".format(r)
        coordinates = config["coordinates"][boxkey]
        bottom_left = np.array([coordinates["x0"],coordinates["y0"]])
        top_right = np.array([coordinates["x1"],coordinates["y1"]])
        dims = top_right-bottom_left

        return dims

    def get_nestcoords(self,config,r):
        """Get the position of a nest relative to a box with bottom left corner at 0,0. 

        :param config: configuration dictionary used to generate analysis.
        :param r: the index of the box for which we are getting dimensions. 
        """
        boxkey = "box{}".format(r)
        coordinates = config["coordinates"][boxkey]
        bottom_left = np.array([coordinates["x0"],coordinates["y0"]])
        top_right = np.array([coordinates["x1"],coordinates["y1"]])
        dims = top_right-bottom_left
        
        nest_coordinates = config["nests"][boxkey]
        nest_bottom_left = np.array([nest_coordinates["xn0"],nest_coordinates["yn0"]])
        nest_top_right = np.array([nest_coordinates["xn1"],nest_coordinates["yn1"]])
        nest_bl = nest_bottom_left - bottom_left
        nest_tr = nest_top_right-bottom_left

        nest_dims = nest_tr-nest_bl

        ## Format nest for drawing rectangle
        left,bottom = nest_bl
        width = nest_dims[0]
        height = nest_dims[1] 
        return left,bottom,width,height 

    def filter_pursuits(self,experiment_name,filterfunc,filtername):
        """In addition to an experiment name, takes a particular filter function as input, as well as a name for that function. Generates subfolders for that experiment with name filtername, that filter events according to that filterfunction.  
        
        """
        ## Vet input
        assert type(experiment_name) is str,"experiment name must be string."
        try:
            edict = self.experiments[experiment_name]
        except KeyError:
            print("specified experiment name does not exist or was not cleared for analysis.")
            raise
        ## Now get total number of rois and parts:
        parts = edict["nb_trace_sets"]
        rois = self.__get_box_indices(edict["config"])
        for r in rois:
            for p in range(parts):
                self.filter_pursuit_traceset(edict,r,p,filterfunc,filtername)

    def filter_pursuit_traceset(self,edict,r,p,filterfunc,filtername):
        """Provided with an experiment dictionary, roi, and part number, filter function and name of filter, 
        applies the filter to all pursuit events in the specified trace set copies them to a subdirectory. 

        """
        trace_directory = self.get_trace_dir(edict["ExperimentName"],r,p)
        ## Get all pursuit files: 
        all_contents = os.listdir(trace_directory)
        all_pursuits = [a for a in all_contents if a.endswith(".npy")]

        ## First make directory if not existing: 
        filtered_dir =  self.get_trace_filter_dir(edict["ExperimentName"],r,p,filtername)
        mkdir_notexists(filtered_dir)

        for pursuit in all_pursuits:
            filename = os.path.join(trace_directory,pursuit)
            data = np.load(filename,allow_pickle = True)[()]
            cond = filterfunc(data)
            if cond:
                dest = os.path.join(filtered_dir,pursuit)
                copyfile(filename,dest)
            else:
                pass

    def filter_distance(self,distance_thresh,pursuitevent):
        """Reject those pursuit events where the two mice are not within the distance threshold provided at some point. 

        :param distance: threshold distance for this filtering. 
        :param pursuitevent: the dictionary we are evaluating against given distance criterion. 
        """
        ## Get distance:
        difference = pursuitevent["vtraj"]-pursuitevent["mtraj"]
        distance = np.linalg.norm(difference,axis = 1)
        if np.any(distance < distance_thresh):
            return True
        else:
            return False

    def filter_velocity(self,speed_thresh,pursuitevent):
        """Reject those pursuit events where the two mice are not within the distance threshold provided at some point. 

        :param distance: threshold distance for this filtering. 
        :param pursuitevent: the dictionary we are evaluating against given distance criterion. 
        """
        ## Get distance:
        cond = True
        for t in ["vtraj","mtraj"]:
            vel = np.diff(pursuitevent[t],axis = 0)
            speed = np.linalg.norm(vel,axis = 1)
            print(speed)
            if np.any(speed > speed_thresh):
                cond = False
            else:
                pass
        return cond

    def make_clip(self,experimentname,r,p,interval,plotpath):
        """Make a video clip of the specified interaction. 

        """
        videoname = self.experiments[experimentname]["VideoName"]
        clip = VideoFileClip(os.path.join(self.path,videoname))
        config = self.__load_config(os.path.join(self.path,self.experiments[experimentname]["ConfigName"]))
        config_box=config["coordinates"]["box{}".format(r)]
        ## now get the time in seconds: 
        interval_time = np.array(interval)+p*self.trace_length_frames
        print(interval,p,self.trace_length_frames)
        interval_secs = interval_time/30
        print(interval_secs)
        cropped = clip.crop(x1 = config_box["x0"],x2 = config_box["x1"],y1 = config_box["y0"],y2 = config_box["y1"])
        cutout = cropped.subclip(t_start = interval_secs[0],t_end=interval_secs[1])
        cutout.write_videofile(os.path.join(plotpath,experimentname+"ROI_{}".format(r)+"PART_{}".format(p)+"Interval_{}".format(interval)+'.mp4'),codec= 'mpeg4',bitrate = '1000k')

    def split_crossing_part(self,part,index):
        """Given a pair (part,index) array of shape (2,) that has been identified as problematic, splits them into arrays of shape (2,2) that correspond to a splitting of the indicated interval event over parts. 

        """
        assert part.shape == index.shape == (2,),"shape must be correct."
        assert part[1]-part[0] == 1,"there must be a crossing event across parts to fix." ## Lets assume no pursuits last for a full 40 mins. 

        partarray_split = np.array([[part[0],part[0]],[part[1],part[1]]])
        indexarray_split = np.array([[index[0],self.trace_length_frames],[0,index[1]]])
        return partarray_split,indexarray_split
        
    def check_pursuit_parts(self,part,index):
        """Given the video parts (video segmentation) and frame indices for each pursuit event, checks that pursuits do not cross segmentation boundaries. 
        If they do, resolves them into two pursuit events that respect those segmentation boundaries.
        """
        assert part.shape[1] == 2 
        assert index.shape[1] == 2 

        part_diff = part[:,1]-part[:,0]
        crossing = np.where(part_diff)[0]
        rpart = part
        rindex = index
        for ci,c in enumerate(crossing):
            split_part,split_index = self.split_crossing_part(part[c,:],index[c,:])
            print(rpart.shape,rindex.shape,"shape")
            rpart[c+ci,:] = split_part[0,:]
            rpart = np.insert(rpart,c+ci+1,split_part[1,:],axis = 0)
            rindex[c+ci,:] = split_index[0,:]
            rindex = np.insert(rindex,c+ci+1,split_index[1,:],axis = 0)

        return rpart,rindex

    def process_groundtruth(self,experimentname):
        """Get the groundtruth xlsx file, and the process it into rois and parts like your automatically detected pursuits.   

        :param experimentname: the name of the experiment that we are going to process groundtruth data for. Returns none if groundtruth data does not exist for a particular experiment. 
        """
        gtdataname = pathlib.Path(self.experiments[experimentname]["GroundTruth"])
        gtdatafullpath = self.path / gtdataname
        excel = pd.read_excel(gtdatafullpath)
        ## Dictionary to index into the actual datasets that we care about.  
        pursuit_dict = {"Virgin":1,"Dam":-1}
        cage_dict = {1:0,2:1,3:2}
        index_template = "C{c} {p} Sh"
        for pi in pursuit_dict:
            for c in cage_dict:
                name = index_template.format(c = c, p = pi)
                dataset = excel.loc[excel["Behavior"] == name]
                dataset_frames = dataset[["Start (s)","Stop (s)"]]*30
                part,index = np.divmod(dataset_frames.values,self.trace_length_frames)
                filteredpart,filteredindex = self.check_pursuit_parts(part,index)

                ## Get dataset indexing params. 
                r = cage_dict[c]
                for i in range(filteredpart.shape[0]):
                    p = int(filteredpart[i,0])
                    filtered_dir =  self.get_trace_filter_dir(experimentname,r,p,"groundtruth")
                    mkdir_notexists(filtered_dir)
                    interval = filteredindex[i,:].astype(int)
                    tracename = "{e}roi_{r}cropped_part{p}".format(e=experimentname,r=r,p=p)+self.trace_suffix
                    dataset = loadmat(os.path.join(self.path,tracename))
                    intervaldict,pdir_proposed = self.get_intervaldict(dataset,interval,r,p,experimentname)
                    dict_name = "Pursuit{I}_Direction{P}.npy".format(I = interval,P = pdir_proposed)
                    full_path_to_data = os.path.join(filtered_dir,dict_name)
                    np.save(full_path_to_data,intervaldict)

    def extract_interval(self,string):
        """Function to extract the interval and pursuit direction from a string representing the name of a file. 

        """
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+",string)
        assert len(numbers) == 3, "must have two entries for the interval and one for the direction."
        return {"interval":[int(numbers[0]),int(numbers[1])], "direction":float(numbers[2])}

    def calculate_statistics_eventwise(self,compareimage,all_pursuits,buf = 5):
        """Calculate eventwise statistics on our dataset. 

        :param compareimage: numpy array to use to calculate these statistics. 
        :param all_pursuits: a list of two lists, each containing dictionaries with information about relevant pursuit events. 
        :param buf: an integer providing a frame buffer which we will use when considering if events overlap or not. 

        :return: returns a dictionary indicating the events in A that were detected by B and vice versa. 
        """
        assert compareimage.shape[0] == 2
        assert len(all_pursuits) == 2
        output = {"A_detectedby_B":[],"B_detectedby_A":[]}
        index = ["A_detectedby_B","B_detectedby_A"]
        for ti,pursuitstype in enumerate(all_pursuits):
            otherind = 1-ti
            for event in pursuitstype:
                interval = np.array(event["interval"])
                bufinterval = np.array([-buf,buf])+interval

                other = compareimage[otherind,slice(*bufinterval)]
                cond = np.any(other)
                output[index[ti]].append(cond)
        return output

    def calculate_statistics_durationwise(self,compareimage,all_pursuits):
        """Calculate durationwise statistics on our dataset. 

        :param compareimage: numpy array to use to calculate these statistics. 
        :param all_pursuits: a list of two lists, each containing dictionaries with information about relevant pursuit events. 

        :return: returns a dictionary indicating the proportion of each event in A that was detected by B and vice versa. 
        """
        assert compareimage.shape[0] == 2
        assert len(all_pursuits) == 2
        output = {"A_proportionin_B":[],"B_proportionin_A":[]}
        index = ["A_proportionin_B","B_proportionin_A"]
        for ti,pursuitstype in enumerate(all_pursuits):
            otherind = 1-ti
            for event in pursuitstype:
                interval = np.array(event["interval"])

                other = compareimage[otherind,slice(*interval)]
                prop = np.sum(abs(other))/len(other)
                output[index[ti]].append(prop)
        return output
                
    def calculate_statistics_directional(self,compareimage,all_pursuits,buf = 5):
        """Calculate durationwise statistics on our dataset. 

        :param compareimage: numpy array to use to calculate these statistics. 
        :param all_pursuits: a list of two lists, each containing dictionaries with information about relevant pursuit events. 
        :param buf: an integer providing a frame buffer which we will use when considering if events overlap or not. 

        :return: returns a dictionary indicating the proportion of each event in A that was detected by B and vice versa. 
        """
        assert compareimage.shape[0] == 2
        assert len(all_pursuits) == 2
        output = {"A_directiongiven_B":[],"B_directiongiven_A":[]}
        index = ["A_directiongiven_B","B_directiongiven_A"]
        for ti,pursuitstype in enumerate(all_pursuits):
            otherind = 1-ti
            for event in pursuitstype:
                interval = np.array(event["interval"])
                direction = event["direction"]
                bufinterval = np.array([-buf,buf])+interval

                other = compareimage[otherind,slice(*bufinterval)]
                ## Round in favor of correct if the inferred direction is exactly balanced. 
                inferreddirection = np.sign(np.sum(other)+0.1*direction).astype(int)
                dirs = {"ref":direction,"targ":inferreddirection}
                output[index[ti]].append(dirs)
        return output


    def calculate_statistics(self,compareimage,all_pursuits):
        """Calculate event wise, duration wise, and direction classification statistics on our dataset. 
        
        :param compareimage: numpy array to use to calculate these statistics. 
        :param all_pursuits: a list of two lists, each containing dictionaries with information about relevant pursuit events. 
        """
        ## Calculate eventwise stats:
        outevent = self.calculate_statistics_eventwise(compareimage,all_pursuits)
        outduration =self.calculate_statistics_durationwise(compareimage,all_pursuits)
        outdirection = self.calculate_statistics_directional(compareimage,all_pursuits)



    def compare_pursuits(self,experimentname,r,p,filternames,plotpath = None):
        """Look at two different sets of pursuit events, and compare them. Generate a table that compares the two, as well as a a plot (optional).   

        :param experimentname: the name of the experiment for which we will retrieve pursuits. 
        :param r: the integer giving the roi index we will look at. 
        :param p: the integer giving the video part we will look at. 
        :param filternames: a list with two elements, describing the names of the two filters we will compare. If it contains only one element, will compare against groundtruth. 
        :param plotpath: path to which we will plot and save out the result. 
        """

        ## Get the name of the directories that we should search for:
        assert type(filternames) == list, "filternames must be passed as list."
        if len(filternames) == 1:
            dirnames = [self.get_trace_dir(experimentname,r,p),self.get_trace_filter_dir(experimentname,r,p,filternames[0])]
            filternames_fixed = ["raw",filternames[0]]
        elif len(filternames) == 2:
            dirnames = [self.get_trace_filter_dir(experimentname,r,p,filtername) for filtername in filternames]
            filternames_fixed = filternames
        else:
            raise Exception("filternames must be a list of length 1 or 2")

        ## First get the names and detected pursuit leaders for each interval in both sets of pursuit events. 
        all_pursuits = []
        compareimage = np.zeros((2,self.trace_length_frames))
        for n,name in enumerate(dirnames):
            pursuits = list(pathlib.Path(name).glob("*.npy"))
            if len(pursuits) > 0: 
                pursuitnames = [p.stem for p in pursuits]
                pursuitnumbers = [self.extract_interval(pn) for pn in pursuitnames]
                all_pursuits.append(pursuitnumbers)
                for pn in pursuitnumbers:
                    interval = pn["interval"]
                    direction = pn["direction"]
                    print(interval)
                    compareimage[n,slice(*interval)] = direction

            else:
                raise FileNotFoundError("filter results not yet generated for filtername {}.".format(name)) 
        if plotpath:
            plt.imshow(compareimage,aspect = 10000)
            plt.axhline(0.5,color = "black")
            plt.title("Ethogram Comparison: Experiment {}, ROI {}, Part {}; Top: {} Bottom: {}".format(experimentname,r,p,*filternames_fixed))
            plt.show()


            
            

        



class PursuitVideo(object):
    """
    Data class for videos of behavioral pursuit. Initialized from json files providing metadata about the location of the data  
    """

    def __init__(self,filename):
        """
        Initialization for this class takes input from a json object specified relevant metadata, as given below.  
        The only field that is required from the provided metata object is the video path. Not providing others will trigger errors in downstream processing however. 
        """
        self.metadata = self.__load_spec(filename)
        self.__parse_videopaths()


    def __load_spec(self,filename):
        with open(filename,"r") as f:
            spec = json.load(f)
        return spec

    def __parse_videopaths(self):
        """
        Parse the provided path to the video data.  
        Paths can be of the form: 
        s3://bucketname/path
        or 
        localdirectory/path/

        """
        try:
            self.metadata["source"]
            assert self.metadata["source"] in ["s3","local"], "source must be specified as 's3' or 'local'"

        except KeyError:
            print("video source location not provided. inferring from path")
            s3_prefix = "s3://"
            ## determine where video is stored
            if self.metadata["path"].startswith(s3_prefix):
                self.metadata["source"] = "s3"
            else:
                self.metadata["source"] = "local"

    def write_frame(self,timestamp,delete=True):
        """
        Get a frame from the video provided. 
        One of the main methods for this class. Returns user-viewable clips in the specified output location (s3 location or local)

        :param timestamp: time (in seconds, or as a string)  for the frame we want. 
        """
        if self.metadata["source"] == "s3":
            write_frame_s3(self.metadata["path"],timestamp,delete=delete)
        elif self.metadata["source"] == "local":
            write_frame_local(self.metadata["path"],timestamp)

    def write_frame_local(self,path,timestamp):
        """
        Gets the frame specified, and saves it. 
        """
        clipname = os.path.basename(os.path.splitext(path)[0])
        
        framename = clipname+"frame{}.png".format(timestamp)
        with VideoFileClip(path) as clip:
            clip.save_frame(framename,t = timestamp)

    def write_frame_s3(self,path,timestamp):
        pass




