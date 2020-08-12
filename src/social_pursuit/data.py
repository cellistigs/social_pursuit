import json 
import os
import yaml
#from moviepy.editor import VideoFileClip
#import boto3
#from botocore.exceptions import ClientError

#s3_client = boto3.client("s3")

package_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(package_dir,"local_paths.json"),"r") as f:
    pathdict = json.load(f)

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

## Class for keeping data organized. 
class PursuitTraces(object):
    """PursuitTraces."""

    """
    Data class for traces of behavioral pursuit. Initialized from json files providing metadata about the data location. 

    :param filename: the name of the file where the trace metadata is stored.  
    """
    def __init__(self,filename):
        """Initialzation for this class takes in

        """
        self.metadata = self.__load_spec(filename)
        clean_experiments = self.__check_data_integrity()
        print(clean_experiments)

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
        self.path = path
        ## get all files in the trace directory. 
        all_contents = os.listdir(path)

        cleared_traces = []
        all_resources = ["config","video","traces"]
        for trace_dict in all_traces:
            config_exists = self.__check_config(trace_dict,all_contents)
            video_exists = self.__check_video(trace_dict,all_contents)
            traces_exist,trace_nb = self.__check_traces(trace_dict,all_contents)
            all_conditions = [config_exists,video_exists,traces_exist]
            approved = all(all_conditions)
            if approved:
                trace_dict["nb_trace_sets"] = trace_nb
                cleared_traces.append(trace_dict)
            else:
                "Name missing resources:"
                missing_resources = [a for i,a in enumerate(all_resources) if all_conditions[i] is False]
                print("ATTENTION: Experiment {e} has errors with the following resources: {l}. It will not be included in further analysis.".format(e = trace_dict["ExperimentName"],l = missing_resources))

        return cleared_traces

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
            boxes = config["coordinates"].keys()
            box_indices = [int(list(filter(str.isdigit,b))[0]) for b in boxes]
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




