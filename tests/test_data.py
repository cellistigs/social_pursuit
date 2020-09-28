# Test the data module
import os
import numpy as np
import pytest
import pathlib
#from botocore.stub import Stubber
from social_pursuit.data import PursuitTraces,Polar#PursuitVideo,PursuitTraces,s3_client,transfer_if_not_found
import social_pursuit.data

test_raw1= "TempTrial2roi_2cropped_part2DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat"
test_raw2= "TempTrial2roi_2cropped_part6DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat"

output_dir = pathlib.Path("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/tempdir")

close_experiment = {"ExperimentName":"TempTrial2",
        "ROI":0,
        "PART":0,
        "Interval":[54078, 54107]
        }

far_experiment = {"ExperimentName":"TempTrial2",
        "ROI":0,
        "PART":1,
        "Interval":[30081, 30093]# [68816, 68843]
        }

clip_experiment = {"ExperimentName":"TempTrial2",
        "ROI":0,
        "PART":1,
        "Interval":[1, 2]# [68816, 68843]
        }

test_fixture_template = pathlib.Path(".").resolve() / pathlib.Path("test_fixtures/traces_test.json")

class Test_PursuitTraces():
    def test_PursuitTraces(self):
        a = PursuitTraces(test_fixture_template)
        assert list(a.experiments.keys()) == ["TempTrial2","TempTrial16"]

    def test_PursuitTraces_get_tracedicts(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        traces = a.get_tracedicts([close_experiment["ExperimentName"]])
        
    def test_PursuitTraces_get_pursuit_traceset(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        path = output_dir
        traces = a.get_pursuit_traceset(a.experiments[close_experiment["ExperimentName"]],r=close_experiment["ROI"],p =close_experiment["PART"],part_fullpath=path)
        
    def test_PursuitTraces_get_pursuit(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        traces = a.get_pursuits(close_experiment["ExperimentName"])

    def test_PursuitTraces_plot_far(self,monkeypatch):
        """TODO make this more fully featured. 
        
        """
        def mockplot():
            return "plotting"
        monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        a.plot_trajectory(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])

    def test_PursuitTraces_plot_close(self,monkeypatch):
        """TODO make this more fully featured. 
        
        """
        def mockplot():
            return "plotting"
        monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        a.plot_trajectory(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])

    def test_PursuitTraces_distance_filter_close_accept(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        tracedir0 = a.load_trace_data(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])
        assert a.filter_distance(5,tracedir0)

    def test_PursuitTraces_distance_filter_close_reject(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        tracedir0 = a.load_trace_data(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])
        assert not a.filter_distance(1,tracedir0)

    def test_PursuitTraces_distance_filter_far_accept(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        tracedir0 = a.load_trace_data(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])
        assert a.filter_distance(10,tracedir0)

    def test_PursuitTraces_distance_filter_far_accept(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        tracedir0 = a.load_trace_data(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])
        print(tracedir0)
        assert not a.filter_distance(1,tracedir0)

    def test_PursuitTraces_velocity_filter_far_accept(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        tracedir0 = a.load_trace_data(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])
        assert a.filter_velocity(31,tracedir0)
        
    def test_PursuitTraces_velocity_filter_far_reject(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        tracedir0 = a.load_trace_data(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])
        assert not a.filter_velocity(1,tracedir0)

    def test_PursuitTraces_velocity_filter_close_accept(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        tracedir0 = a.load_trace_data(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])
        assert a.filter_velocity(10,tracedir0)

    def test_PursuitTraces_velocity_filter_close_reject(self):
        a = PursuitTraces(test_fixture_template)
        ## don't load everything, just first
        tracedir0 = a.load_trace_data(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])
        assert not a.filter_velocity(1,tracedir0)

    def test_PursuitTraces_filter_pursuits_traceset(self):
        a = PursuitTraces(test_fixture_template)
        edict = a.experiments[far_experiment["ExperimentName"]]
        thresh10 = lambda data: a.filter_velocity(10,data)
        a.filter_pursuit_traceset(edict,far_experiment["ROI"],far_experiment["PART"],thresh10,"examplefilter")
        
    def test_PursuitTraces_plot_aggregate(self,monkeypatch):
        def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal):
            pass
        def mockplot():
            return mock
        monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
        a = PursuitTraces(test_fixture_template)
        a.plot_all_trajectories_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])

    def test_PursuitTraces_plot_aggregate_mled(self,monkeypatch):
        def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
            pass
        def mockplot():
            return mock
        monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
        a = PursuitTraces(test_fixture_template)
        a.plot_mled_trajectories_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])

    def test_PursuitTraces_plot_aggregate_vled(self,monkeypatch):
        def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
            pass
        def mockplot():
            return mock
        monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
        a = PursuitTraces(test_fixture_template)
        a.plot_vled_trajectories_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])

    def test_PursuitTraces_plot_aggregate_starts(self,monkeypatch):
        def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
            pass
        def mockplot():
            return mock
        monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
        a = PursuitTraces(test_fixture_template)
        a.plot_all_starts_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])
        
    def test_PursuitTraces_plot_aggregate_ends(self,monkeypatch):
        def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
            pass
        def mockplot():
            return mock
        monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
        a = PursuitTraces(test_fixture_template)
        a.plot_all_ends_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])

    def test_PursuitTraces_plot_aggregate_ends_vled(self,monkeypatch):
        def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
            pass
        def mockplot():
            return mock
        monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
        a = PursuitTraces(test_fixture_template)
        a.plot_all_virgin_ends_in_experiment(close_experiment["ExperimentName"],close_experiment["ROI"],"./here.png")

    def test_PursuitTraces_plot_aggregate_ends_mled(self,monkeypatch):
        def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
            pass
        def mockplot():
            return mock
        monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
        a = PursuitTraces(test_fixture_template)
        a.plot_all_mother_ends_in_experiment(close_experiment["ExperimentName"],close_experiment["ROI"],"./here.png")
        
    def test_PursuitTraces_make_clip_far(self,monkeypatch):
        def mock_videowrite(self,name,codec,bitrate):
            print(name)
        monkeypatch.setattr(social_pursuit.data.VideoFileClip,"write_videofile",mock_videowrite)
        a = PursuitTraces(test_fixture_template)
        a.make_clip(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"],"./")

    def test_PursuitTraces_check_pursuit_parts_clear(self):
        a = PursuitTraces(test_fixture_template)
        parts = np.array([[0,0],[1,1],[1,1]])
        indices = np.array([[10,20],[1,2],[4,10]])
        nparts,nindices=a.check_pursuit_parts(parts,indices)
        assert np.array_equal(nparts,parts)
        assert np.array_equal(nindices,indices)
        
    def test_PursuitTraces_check_pursuit_parts_fail(self):
        a = PursuitTraces(test_fixture_template)
        parts = np.array([[0,0],[0,1],[1,1],[1,2],[3,3]])
        indices = np.array([[10,20],[71000,2],[4,10],[71999,1],[4,5]])
        nparts,nindices=a.check_pursuit_parts(parts,indices)
        print(nparts,nindices)
        assert np.array_equal(nparts,np.array([[0,0],[0,0],[1,1],[1,1],[1,1],[2,2],[3,3]]))
        assert np.array_equal(nindices,np.array([[10,20],[71000,a.trace_length_frames],[0,2],[4,10],[71999,a.trace_length_frames],[0,1],[4,5]]))
        

    def test_PursuitTraces_process_groundtruth(self):
        a = PursuitTraces(test_fixture_template)
        a.process_groundtruth(close_experiment["ExperimentName"])
        
    def test_PursuitTraces_plot_groundtruth(self):
        a = PursuitTraces(test_fixture_template)
        a.plot_trajectories_traceset(close_experiment["ExperimentName"],2,close_experiment["PART"],filtername = "groundtruth")

    def test_PursuitTraces_compare_pursuits_traceset_raw(self):
        a = PursuitTraces(test_fixture_template)
        a.compare_pursuits_traceset(close_experiment["ExperimentName"],1,close_experiment["PART"],filternames = ["groundtruth"],plotpath = "somepath")

    def test_PursuitTraces_compare_pursuits_traceset(self):
        a = PursuitTraces(test_fixture_template)
        out = a.compare_pursuits_traceset(close_experiment["ExperimentName"],1,close_experiment["PART"],filternames = ["groundtruth","examplefilter"])
        assert out is None

    def test_PursuitTraces_calculate_statistics_eventwise(self):
        a = PursuitTraces(test_fixture_template)
        compareimage = np.array([[1,1,1,0,1,1],[0,0,0,0,1,0]])
        all_pursuits = [[{"interval":[0,3],"direction":1},{"interval":[4,6],"direction":1}],[{"interval":[4,5],"direction":1}]]
        output = a.calculate_statistics_eventwise(compareimage,all_pursuits,buf = 0)
        assert output == {"A_detectedby_B":[False,True],"B_detectedby_A":[True]}
        
    def test_PursuitTraces_calculate_statistics_eventwise_buf(self):
        a = PursuitTraces(test_fixture_template)
        compareimage = np.array([[1,1,1,0,1,1],[0,0,0,0,1,0]])
        all_pursuits = [[{"interval":[0,3],"direction":1},{"interval":[4,6],"direction":1}],[{"interval":[4,5],"direction":1}]]
        output = a.calculate_statistics_eventwise(compareimage,all_pursuits,buf = 3)
        assert output == {"A_detectedby_B":[True,True],"B_detectedby_A":[True]}

    def test_PursuitTraces_calculate_statistics_durationwise(self):
        a = PursuitTraces(test_fixture_template)
        compareimage = np.array([[1,1,1,0,1,1],[0,0,0,0,1,0]])
        all_pursuits = [[{"interval":[0,3],"direction":1},{"interval":[4,6],"direction":1}],[{"interval":[4,5],"direction":1}]]
        output = a.calculate_statistics_durationwise(compareimage,all_pursuits)
        assert output == {"A_proportionin_B":[0.0,0.5],"B_proportionin_A":[1.0]}

    def test_PursuitTraces_calculate_statistics_directional(self):
        a = PursuitTraces(test_fixture_template)
        compareimage = np.array([[1,1,1,0,1,1],[0,0,0,0,1,0]])
        all_pursuits = [[{"interval":[0,3],"direction":1},{"interval":[4,6],"direction":1}],[{"interval":[4,5],"direction":1}]]
        output = a.calculate_statistics_directional(compareimage,all_pursuits,buf = 0)
        assert output == {"A_directiongiven_B":[{"ref":1,"targ":0},{"ref":1,"targ":1}],"B_directiongiven_A":[{"ref":1,"targ":1}]}

    def test_PursuitTraces_calculate_statistics_directional_diffdirection(self):
        a = PursuitTraces(test_fixture_template)
        compareimage = np.array([[1,1,1,0,1,1],[0,0,0,0,-1,0]])
        all_pursuits = [[{"interval":[0,3],"direction":1},{"interval":[4,6],"direction":1}],[{"interval":[4,5],"direction":-1}]]
        output = a.calculate_statistics_directional(compareimage,all_pursuits,buf = 0)
        assert output == {"A_directiongiven_B":[{"ref":1,"targ":0},{"ref":1,"targ":-1}],"B_directiongiven_A":[{"ref":-1,"targ":1}]}

    def test_PursuitTraces_calculate_statistics_directional_rounddirection(self):
        a = PursuitTraces(test_fixture_template)
        compareimage = np.array([[1,1,1,0,-1,-1],[0,0,0,0,-1,0]])
        all_pursuits = [[{"interval":[0,3],"direction":1},{"interval":[4,6],"direction":-1}],[{"interval":[4,5],"direction":-1}]]
        output = a.calculate_statistics_directional(compareimage,all_pursuits,buf = 3)
        assert output == {"A_directiongiven_B":[{"ref":1,"targ":-1},{"ref":-1,"targ":-1}],"B_directiongiven_A":[{"ref":-1,"targ":-1}]}

    def test_PursuitTraces_calculate_statistics(self):
        a = PursuitTraces(test_fixture_template)
        compareimage = np.array([[1,1,1,0,1,1],[0,0,0,0,1,0]])
        all_pursuits = [[{"interval":[0,3],"direction":1},{"interval":[4,6],"direction":1}],[{"interval":[4,5],"direction":1}]]
        output = a.calculate_statistics(compareimage,all_pursuits,buf = 0)
        assert output["eventwise"] == {"A_detectedby_B":[False,True],"B_detectedby_A":[True]}
        assert output["durationwise"] == {"A_proportionin_B":[0.0,0.5],"B_proportionin_A":[1.0]}
        assert output["directional"] == {"A_directiongiven_B":[{"ref":1,"targ":0},{"ref":1,"targ":1}],"B_directiongiven_A":[{"ref":1,"targ":1}]}

    def test_PursuitTraces_get_trace_filter_compare_dir(self):
        a = PursuitTraces(test_fixture_template)
        experimentname = "mockexperiment"
        r = 1
        p = 2
        filternames = ["mockfilter1","mockfilter2"]
        output = a.get_trace_filter_compare_dir(experimentname,r,p,filternames)
        assert output == "/Volumes/TOSHIBA EXT STO/RTTemp_Traces/test_dir/mockexperiment_Pursuit_Events/FILTER_A_mockfilter1_FILTER_B_mockfilter2/ROI_1/PART_2"

    def test_PursuitTraces_get_trace_filter_compare_dir_singlet(self):
        a = PursuitTraces(test_fixture_template)
        experimentname = "mockexperiment"
        r = 1
        p = 2
        filternames = ["mockfilter2"]
        output = a.get_trace_filter_compare_dir(experimentname,r,p,filternames)
        assert output == "/Volumes/TOSHIBA EXT STO/RTTemp_Traces/test_dir/mockexperiment_Pursuit_Events/FILTER_A_raw_FILTER_B_mockfilter2/ROI_1/PART_2"

    def test_PursuitTraces_get_trace_filter_compare_dir_fail(self):
        a = PursuitTraces(test_fixture_template)
        experimentname = "mockexperiment"
        r = 1
        p = 2
        filternames = "mockfilter2"
        with pytest.raises(AssertionError):
            output = a.get_trace_filter_compare_dir(experimentname,r,p,filternames)

    def test_PursuitTraces_retrieve_groundtruth_statistics_eventwise(self):
        a = PursuitTraces(test_fixture_template)
        filternames = ["groundtruth"]
        results = a.retrieve_groundtruth_statistics_eventwise(close_experiment["ExperimentName"],filternames)
        assert results["experimentname"] == "TempTrial2"
        assert results["filternames"] == filternames
        assert results[0][0] == {'false_detect':1.0,"true_detect":0.0}
        assert results[1][0] == {'false_detect':0.6,"true_detect":0.8666666666666667}
        assert results[1]["total"]
        assert results[0]["total"]
        
class Test_PursuitTraces_kinematics():
    """Test suite to examine the kinematics of pursuit events.  

    """
    def test_PursuitTraces_skeleton_clustering(self):
        a = PursuitTraces(test_fixture_template)
        a.skeleton_clustering(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"],filtername = None)

    def test_PursuitTraces_skeleton_consistency(self):
        pass

    def test_PursuitTraces_distance_stats(self):
        pass

    def test_PursuitTraces_velocity_stats(self):
        pass

    def test_PursuitTraces_acceleration_stats(self):
        pass

    def test_PursuitTraces_tortuosity_stats(self):
        pass

class TestPolar():
    def test_Polar(self):
        datapath = os.path.join(output_dir,test_raw1)
        data = Polar(datapath)

    def test_Polar_wrongpath(self):
        datapath = os.path.join(output_dir,test_raw1,".py")
        with pytest.raises(AssertionError):
            data = Polar(datapath)

    def test_Polar_load_data(self):
        datapath = os.path.join(output_dir,test_raw1)
        data = Polar(datapath)
        traj = data.load_data()
        assert traj.shape == (72000,2,5,2)


    def test_Polar_get_polar_average(self):
        datapath = os.path.join(output_dir,test_raw1)
        data = Polar(datapath)
        traj = data.load_data()
        avg = data.get_polar_average(traj)
        assert avg.shape == (72000,2)


#def test_load_spec():
#    a = PursuitVideo("test_fixtures/template.json")
#
#def test_parse_videopaths():
#    a = PursuitVideo("test_fixtures/template.json")
#    a = PursuitVideo("test_fixtures/template_nosource.json")
#    with pytest.raises(AssertionError):
#        assert PursuitVideo("test_fixtures/template_source_misformat.json")

