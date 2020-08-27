# Test the data module
import numpy as np
import pytest
import pathlib
#from botocore.stub import Stubber
from social_pursuit.data import PursuitTraces#PursuitVideo,PursuitTraces,s3_client,transfer_if_not_found
import social_pursuit.data

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
output_dir = pathlib.Path("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/tempdir")

def test_PursuitTraces():
    a = PursuitTraces(test_fixture_template)
    assert list(a.experiments.keys()) == ["TempTrial2","TempTrial16"]

def test_PursuitTraces_get_tracedicts():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    traces = a.get_tracedicts([close_experiment["ExperimentName"]])
    
def test_PursuitTraces_get_pursuit_traceset():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    path = output_dir
    traces = a.get_pursuit_traceset(a.experiments[close_experiment["ExperimentName"]],r=close_experiment["ROI"],p =close_experiment["PART"],part_fullpath=path)
    
def test_PursuitTraces_get_pursuit():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    traces = a.get_pursuits(close_experiment["ExperimentName"])

def test_PursuitTraces_plot_far(monkeypatch):
    """TODO make this more fully featured. 
    
    """
    def mockplot():
        return "plotting"
    monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    a.plot_trajectory(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])

def test_PursuitTraces_plot_close(monkeypatch):
    """TODO make this more fully featured. 
    
    """
    def mockplot():
        return "plotting"
    monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    a.plot_trajectory(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])

def test_PursuitTraces_distance_filter_close_accept():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    tracedir0 = a.load_trace_data(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])
    assert a.filter_distance(5,tracedir0)

def test_PursuitTraces_distance_filter_close_reject():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    tracedir0 = a.load_trace_data(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])
    assert not a.filter_distance(1,tracedir0)

def test_PursuitTraces_distance_filter_far_accept():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    tracedir0 = a.load_trace_data(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])
    assert a.filter_distance(10,tracedir0)

def test_PursuitTraces_distance_filter_far_accept():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    tracedir0 = a.load_trace_data(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])
    print(tracedir0)
    assert not a.filter_distance(1,tracedir0)

def test_PursuitTraces_velocity_filter_far_accept():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    tracedir0 = a.load_trace_data(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])
    assert a.filter_velocity(31,tracedir0)
    
def test_PursuitTraces_velocity_filter_far_reject():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    tracedir0 = a.load_trace_data(far_experiment["ExperimentName"],far_experiment["ROI"],far_experiment["PART"],far_experiment["Interval"])
    assert not a.filter_velocity(1,tracedir0)

def test_PursuitTraces_velocity_filter_close_accept():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    tracedir0 = a.load_trace_data(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])
    assert a.filter_velocity(10,tracedir0)

def test_PursuitTraces_velocity_filter_close_reject():
    a = PursuitTraces(test_fixture_template)
    ## don't load everything, just first
    tracedir0 = a.load_trace_data(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"])
    assert not a.filter_velocity(1,tracedir0)

def test_PursuitTraces_filter_pursuits_traceset():
    a = PursuitTraces(test_fixture_template)
    edict = a.experiments[far_experiment["ExperimentName"]]
    thresh10 = lambda data: a.filter_velocity(10,data)
    a.filter_pursuit_traceset(edict,far_experiment["ROI"],far_experiment["PART"],thresh10,"examplefilter")

    
def test_PursuitTraces_plot_aggregate(monkeypatch):
    def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal):
        pass
    def mockplot():
        return mock
    monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
    a = PursuitTraces(test_fixture_template)
    a.plot_all_trajectories_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])

def test_PursuitTraces_plot_aggregate_mled(monkeypatch):
    def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
        pass
    def mockplot():
        return mock
    monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
    a = PursuitTraces(test_fixture_template)
    a.plot_mled_trajectories_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])

def test_PursuitTraces_plot_aggregate_vled(monkeypatch):
    def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
        pass
    def mockplot():
        return mock
    monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
    a = PursuitTraces(test_fixture_template)
    a.plot_vled_trajectories_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])

def test_PursuitTraces_plot_aggregate_starts(monkeypatch):
    def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
        pass
    def mockplot():
        return mock
    monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
    a = PursuitTraces(test_fixture_template)
    a.plot_all_starts_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])
    
def test_PursuitTraces_plot_aggregate_ends(monkeypatch):
    def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
        pass
    def mockplot():
        return mock
    monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
    a = PursuitTraces(test_fixture_template)
    a.plot_all_ends_in_traceset(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"])

def test_PursuitTraces_plot_aggregate_ends_vled(monkeypatch):
    def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
        pass
    def mockplot():
        return mock
    monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
    a = PursuitTraces(test_fixture_template)
    a.plot_all_virgin_ends_in_experiment(close_experiment["ExperimentName"],close_experiment["ROI"],"./here.png")

def test_PursuitTraces_plot_aggregate_ends_mled(monkeypatch):
    def mock(experimentname,r,p,all_pursuits,plotpath,plot_starts_internal,path):
        pass
    def mockplot():
        return mock
    monkeypatch.setattr(PursuitTraces,"plot_trajectories_from_filenames",mockplot()) 
    a = PursuitTraces(test_fixture_template)
    a.plot_all_mother_ends_in_experiment(close_experiment["ExperimentName"],close_experiment["ROI"],"./here.png")
    
def test_PursuitTraces_make_clip_far(monkeypatch):
    def mock_videowrite(self,name,codec,bitrate):
        print(name)
    monkeypatch.setattr(social_pursuit.data.VideoFileClip,"write_videofile",mock_videowrite)
    a = PursuitTraces(test_fixture_template)
    a.make_clip(close_experiment["ExperimentName"],close_experiment["ROI"],close_experiment["PART"],close_experiment["Interval"],"./")

def test_PursuitTraces_check_pursuit_parts_clear():
    a = PursuitTraces(test_fixture_template)
    parts = np.array([[0,0],[1,1],[1,1]])
    indices = np.array([[10,20],[1,2],[4,10]])
    nparts,nindices=a.check_pursuit_parts(parts,indices)
    assert np.array_equal(nparts,parts)
    assert np.array_equal(nindices,indices)
    
def test_PursuitTraces_check_pursuit_parts_fail():
    a = PursuitTraces(test_fixture_template)
    parts = np.array([[0,0],[0,1],[1,1],[1,2],[3,3]])
    indices = np.array([[10,20],[71000,2],[4,10],[71999,1],[4,5]])
    nparts,nindices=a.check_pursuit_parts(parts,indices)
    print(nparts,nindices)
    assert np.array_equal(nparts,np.array([[0,0],[0,0],[1,1],[1,1],[1,1],[2,2],[3,3]]))
    assert np.array_equal(nindices,np.array([[10,20],[71000,a.trace_length_frames],[0,2],[4,10],[71999,a.trace_length_frames],[0,1],[4,5]]))
    

def test_PursuitTraces_process_groundtruth():
    a = PursuitTraces(test_fixture_template)
    a.process_groundtruth(close_experiment["ExperimentName"])
    
#
#def test_load_spec():
#    a = PursuitVideo("test_fixtures/template.json")
#
#def test_parse_videopaths():
#    a = PursuitVideo("test_fixtures/template.json")
#    a = PursuitVideo("test_fixtures/template_nosource.json")
#    with pytest.raises(AssertionError):
#        assert PursuitVideo("test_fixtures/template_source_misformat.json")

