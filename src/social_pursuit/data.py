import json 
import os
from moviepy.editor import VideoFileClip
import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client("s3")

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
        assert self.__check_data_integrity()

    def __load_spec(self,filename):
        with open(filename,"r") as f:
            spec = json.load(f)
        return spec

    def __check_data_integrity(self):
        try:
            self.metadata
        except AttributeError:
            print("metadata does not exist.")
            raise
        all_traces = self.metadata["traces_expected"]

        path = self.metadata["trace_directory"]
        

        for trace_dict in all_traces:
            print(trace_dict)
        assert false



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




