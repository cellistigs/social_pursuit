# Test the labeled data module
import os
import numpy as np
import pytest
import pathlib
import matplotlib.pyplot as plt
#from botocore.stub import Stubber
from social_pursuit.data import PursuitTraces,Polar,PursuitVideo,ExperimentInitializer#,PursuitTraces,s3_client,transfer_if_not_found
from social_pursuit.labeled import LabeledData
import social_pursuit.data


labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"
class Test_LabeledData():
    def test_LabeledData(self):
        data = LabeledData(labeled_data,additionalpath)

    def test_LabeledData_get_dataarray(self):
        data = LabeledData(labeled_data,additionalpath)
        np.testing.assert_equal(data.data.values[:,:2] , data.dataarray[:,:,0,0])
        np.testing.assert_equal(data.data.values[:,-2:] , data.dataarray[:,:,4,1])

    def test_LabeledData_datasets_indices(self):
        data = LabeledData(labeled_data,additionalpath)
        datamapping = data.datasets_indices()
        for d,drange in datamapping.items():
            assert np.all(drange<data.data.shape[0])

    def test_LabeledData_get_positions(self):
        data = LabeledData(labeled_data,additionalpath)
        part_locs = data.get_positions(np.arange(0,2),0)
        assert part_locs.shape == (2,2)
        np.testing.assert_equal(part_locs,data.data.values[0:2,0:2])

    def test_LabeledData_get_positions_last(self):
        data = LabeledData(labeled_data,additionalpath)
        part_locs = data.get_positions(np.arange(0,2),9)
        assert part_locs.shape == (2,2)
        np.testing.assert_equal(part_locs,data.data.values[0:2,18:20])

    def test_LabeledData_distances(self):
        data = LabeledData(labeled_data,additionalpath)
        part_dists = data.distances(np.arange(0,2),0,1)
        norm = np.linalg.norm(data.get_positions(np.arange(0,2),0)-data.get_positions(np.arange(0,2),1),axis =1)
        assert np.all(part_dists == norm)

    def test_LabeledData_distances_mean(self):
        data = LabeledData(labeled_data,additionalpath)
        part_mean = data.distances_mean(np.arange(0,2),0,1)
        norm = np.linalg.norm(data.get_positions(np.arange(0,2),0)-data.get_positions(np.arange(0,2),1),axis =1)
        assert np.all(part_mean == np.mean(norm))

    def test_LabeledData_distances_std(self):
        data = LabeledData(labeled_data,additionalpath)
        part_std = data.distances_std(np.arange(0,2),0,1)
        norm = np.linalg.norm(data.get_positions(np.arange(0,2),0)-data.get_positions(np.arange(0,2),1),axis =1)
        assert np.all(part_std == np.std(norm))

    def test_LabeledData_distances_wholemouse(self):
        data = LabeledData(labeled_data,additionalpath)
        interval = np.arange(0,3)
        part0 = 0
        part1 = 1
        part_dists = data.distances(interval,part0,part1)
        mouse_dists = data.distances_wholemouse(interval,0)
        np.testing.assert_equal(mouse_dists[(part1,part0)],part_dists)

    def test_LabeledData_stats_wholemouse(self):
        data = LabeledData(labeled_data,additionalpath)
        interval = np.arange(0,3)
        part0 = 0
        part1 = 1
        part_mean = data.distances_mean(interval,part0,part1)
        part_std = data.distances_std(interval,part0,part1)
        mouse_stats = data.stats_wholemouse(interval,0) 
        np.testing.assert_equal(mouse_stats[(part1,part0)][0],part_mean)
        np.testing.assert_equal(mouse_stats[(part1,part0)][1],part_std)

    def test_LabeledData_sample(self):
        data = LabeledData(labeled_data,additionalpath)
        n = 5
        ds = data.sample(n)
        assert ds.shape == (n,2,5,2)

    def test_LabeledData_sample_radius(self):
        data = LabeledData(labeled_data,additionalpath)
        n = 5
        length = 10.
        ds = data.sample_radius(n,length)
        assert ds.shape == (n,2,5,2)
        mean_pos = np.nanmean(ds,axis = 2)
        dir_vecs = np.diff(mean_pos,axis = -1)
        lengths = np.linalg.norm(dir_vecs,axis = 1).flatten()
        diff = lengths=length
        ## Nans should compare to nan, all others should be zero.
        baseline = lengths-lengths
        assert np.isclose(lengths - length,baseline,equal_nan = True)

    def test_LabeledData_sample_radius_ori(self):
        data = LabeledData(labeled_data,additionalpath)
        n = 5
        length = 10.
        ds = data.sample_radius_orientation(n,length)
        assert ds.shape == (n,2,5,2)
        mean_pos = np.nanmean(ds,axis = 2)
        dir_vecs = np.diff(mean_pos,axis = -1)
        lengths = np.linalg.norm(dir_vecs,axis = 1).flatten()
        diff = lengths=length
        ## Nans should compare to nan, all others should be zero.
        baseline = lengths-lengths
        assert np.isclose(lengths - length,baseline,equal_nan = True)
