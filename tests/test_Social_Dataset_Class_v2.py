from social_pursuit.Social_Dataset_Class_v2 import Social_Dataset
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import pytest

#tracepath = "/Volumes/TOSHIBA EXT STO/Legacy_Traces/V116_03082018/pursuits/part13/interval23796_23903/traces"
tracepath = "/Volumes/TOSHIBA EXT STO/Legacy_Traces/V116_03082018/pursuits/part13/interval22765_23181/traces"
labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
scorer = "DeepCut_resnet50_social_NewJul8shuffle1_1030000"

output_dir = pathlib.Path("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/tempdir")
fixture_dir = pathlib.Path("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/test_dir")

class Test_Social_Dataset():
    def test_Social_Dataset(self):
        sd = Social_Dataset(tracepath,networkname = scorer)

    def test_Social_Dataset_filter_full_new(self):
        sd = Social_Dataset(tracepath,vers =1,networkname = scorer)
        sd.filter_full_new(labeled_data)

    def test_Social_Dataset_render_trajectory_nan(self):
        sd = Social_Dataset(tracepath,vers =1,networkname = scorer)
        sd.filter_full_new(labeled_data)
        partind = 4
        traj = sd.render_trajectory_nan(partind)
        plt.plot(sd.dataset[sd.scorer].values[:,partind*3],sd.dataset[sd.scorer].values[:,partind*3+1],label = "original")
        plt.plot(traj[:,0],traj[:,1],label = "corrected")
        plt.legend()
        plt.savefig(os.path.join(output_dir,"test_Social_Dataset_render_trajectory_nan.png"))

    def test_Social_Dataset_render_trajectories_nan(self):
        sd = Social_Dataset(tracepath,vers =1,networkname = scorer)
        sd.filter_full_new(labeled_data)
        partind = 4
        traj = sd.render_trajectory_nan(partind)
        trajectories = sd.render_trajectories_nan()
        assert np.all(np.nanmean(trajectories[:,:,4,0]-traj,axis = 0) == np.array([0,0]))
        assert trajectories.shape[1:] == (2,5,2)


