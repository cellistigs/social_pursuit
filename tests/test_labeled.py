# Test the labeled data module
import os
import joblib
import numpy as np
import copy
import pytest
import pathlib
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage import color
#from botocore.stub import Stubber
from social_pursuit.data import PursuitTraces,Polar,PursuitVideo,ExperimentInitializer#,PursuitTraces,s3_client,transfer_if_not_found
from social_pursuit.labeled import LabeledData,LineSelector
import social_pursuit.data
import json

output_dir = pathlib.Path("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/tempdir")
fixture_dir = pathlib.Path("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/test_dir")

labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"
class Test_LabeledData():
    def test_LabeledData(self):
        data = LabeledData(labeled_data,additionalpath)

    def test_LabeledData_get_dataarray(self):
        data = LabeledData(labeled_data,additionalpath)
        np.testing.assert_equal(data.data.values[:,:2] , data.dataarray[:,:,0,0])
        np.testing.assert_equal(data.data.values[:,-2:] , data.dataarray[:,:,4,1])

    def test_LabeledData_get_dataarray_nonan(self):
        data = LabeledData(labeled_data,additionalpath)
        nonan = data.get_dataarray_nonan()
        assert len(np.where(np.isnan(nonan))[0]) == 0


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

    def test_LabeledData_switch_points(self):
        data = LabeledData(labeled_data,additionalpath)
        sampled = data.get_dataarray_nonan()[:3,:,:,:]
        c = copy.deepcopy(sampled)
        indices = [0]
        switched = data.switch_points(sampled,indices,2)
        ## Todo come up with right assert condition for this.

    def test_LabeledData_steal_points(self):
        data = LabeledData(labeled_data,additionalpath)

        sampled = data.get_dataarray_nonan()[:3,:,:,:]
        c = copy.deepcopy(sampled)
        indices = [0]
        switched = data.steal_points(sampled,indices,2)
        ## Todo come up with right assert condition for this.

    def test_LabeledData_remove_animal(self):
        data = LabeledData(labeled_data,additionalpath)
        sampled = data.get_dataarray_nonan()[:3,:,:,:]
        c = copy.deepcopy(sampled)
        indices = [0]
        switched = data.remove_animal(sampled,indices)
        assert np.all(switched[0,:,:,0] == switched[0,:,:,1])

    def test_LabeledData_get_images(self):
        data = LabeledData(labeled_data,additionalpath)
        images = data.get_images([0])
        assert images[0].shape[-1] == 3

    def test_LabeledData_get_images_grayscale(self):
        data = LabeledData(labeled_data,additionalpath)
        images = data.get_images_grayscale([0])
        assert images[0].shape == (480, 704)

    def test_LabeledData_convert_images_grayscale(self):
        data = LabeledData(labeled_data,additionalpath)
        images = data.get_images([0])
        gimages = data.convert_images_grayscale(images)
        assert gimages[0].shape == (480, 704) 

    def test_LabeledData_binarize_frames(self):
        data = LabeledData(labeled_data,additionalpath)
        frameinds = [0,40,41,42]
        out = data.binarize_frames(frameinds)
        assert len(out) == len(frameinds)
        assert out[0].dtype == np.bool_ 

    def test_LabeledData_binarize_frames_wrong_order(self):
        data = LabeledData(labeled_data,additionalpath)
        frameinds = [40,41,42,0]
        with pytest.raises(AssertionError):
            out = data.binarize_frames(frameinds)

    def test_LabeledData_clean_binary(self):
        data = LabeledData(labeled_data,additionalpath)
        frameinds = [0,40,41,42,85]
        out = data.binarize_frames(frameinds)
        clean = data.clean_binary(out)
        plt.imshow(clean[4])
        plt.savefig(os.path.join(output_dir,"test_LabeledData_clean_binary.png"))

        #plt.imshow(clean[1])
        #plt.show()

    def test_LabeledData_link_parts(self):
        data = LabeledData(labeled_data,additionalpath)
        poses = data.dataarray[84,:,:,0].T
        lines = data.link_parts(poses)
        plt.plot(lines[:,0],lines[:,1])
        plt.savefig(os.path.join(output_dir,"test_LabeledData_link_parts.png"))

    def test_LabeledData_generate_auxpoints(self):
        data = LabeledData(labeled_data,additionalpath)
        to_edit = ["dam","virgin"]
        out = data.generate_auxpoints(0,to_edit,dryrun = False)
        assert list(out.keys()) == to_edit
        print(out)
        assert 0 

    def test_LabeledData_segment_animals(self):
        data = LabeledData(labeled_data,additionalpath)
        frameinds = [0,40,41,42,85,98]
        selectid = 4
        frameselect = frameinds[selectid]
        fig,ax = plt.subplots(2,2)
        ax[0,0].imshow(data.get_images(frameinds)[selectid])
        out = data.binarize_frames(frameinds)
        clean = data.clean_binary(out)
        virgin_data = data.dataarray[frameselect,:,:,0]
        dam_data = data.dataarray[frameselect,:,:,1]
        dcent = data.get_positions(np.array([frameselect]),8)
        vcent = data.get_positions(np.array([frameselect]),3)
        ax[0,1].imshow(clean[selectid])
        #plt.plot(dcent[:,1],dcent[:,0],"x")
        #plt.plot(vcent[:,1],vcent[:,0],"x")
        ax[0,1].plot(dcent[:,0],dcent[:,1],"x")
        ax[0,1].plot(vcent[:,0],vcent[:,1],"x")
        labeled,markers = data.segment_animals(out[selectid],virgin_data,dam_data)
        ax[1,0].imshow(markers)
        ax[1,1].imshow(labeled)
        ax[0,0].set_title("Original image")
        ax[0,1].set_title("Cleaned binarized image")
        ax[1,0].set_title("Markers for watershed.")
        ax[1,1].set_title("Segmented Image.")
        plt.savefig(os.path.join(output_dir,"test_LabeledData_segment_animals"))

    def test_LabeledData_segment_animals_interactive(self):
        data = LabeledData(labeled_data,additionalpath)
        frameinds = [0,40,41,42,85,98]
        selectid = 4
        frameselect = frameinds[selectid]
        auxpoints = data.generate_auxpoints(frameselect,["virgin"])
        out = data.binarize_frames(frameinds)
        clean = data.clean_binary(out)
        virgin_data = data.dataarray[frameselect,:,:,0]
        dam_data = data.dataarray[frameselect,:,:,1]
        dcent = data.get_positions(np.array([frameselect]),8)
        vcent = data.get_positions(np.array([frameselect]),3)

        labeled,markers = data.segment_animals(out[selectid],virgin_data,dam_data,auxpoints)
        fig,ax = plt.subplots(2,2,figsize = (20,20))
        ax[0,1].imshow(clean[selectid])
        ax[0,1].plot(dcent[:,0],dcent[:,1],"x")
        ax[0,1].plot(vcent[:,0],vcent[:,1],"x")
        ax[0,0].imshow(data.get_images(frameinds)[selectid])
        ax[1,0].imshow(markers)
        ax[1,1].imshow(labeled)
        ax[0,0].set_title("Original image")
        ax[0,1].set_title("Cleaned binarized image")
        ax[1,0].set_title("Markers for watershed.")
        ax[1,1].set_title("Segmented Image.")
        plt.savefig(os.path.join(output_dir,"test_LabeledData_separate_animals_interactive"))

    def test_LabeledData_save_auxpoints(self):
        data = LabeledData(labeled_data,additionalpath)
        improvement_dict = {1:["virgin","dam"],89:["dam"]}
        #data.save_auxpoints(improvement_dict,"testsave")
        with open(os.path.join(data.additionalpath,"testsave","dict_index_1"),"r") as f:
            d = json.load(f)
        assert list(d.keys()) == ["virgin","dam"]
        with open(os.path.join(data.additionalpath,"testsave","dict_index_89"),"r") as f:
            d = json.load(f)
        assert list(d.keys()) == ["dam"]
    
    def test_LabeledData_smooth_segmentation(self):
        data = LabeledData(labeled_data,additionalpath)
        frameinds = [0,40,41,42,98]
        selectid = 1 
        frameselect = frameinds[selectid]
        out = data.binarize_frames(frameinds)
        clean = data.clean_binary(out)
        virgin_data = data.dataarray[frameselect,:,:,0]
        dam_data = data.dataarray[frameselect,:,:,1]
        dcent = data.get_positions(np.array([frameselect]),8)
        vcent = data.get_positions(np.array([frameselect]),3)
        labeled,markers = data.segment_animals(out[selectid],virgin_data,dam_data)
        fig,ax = plt.subplots(3,1)
        ax[0].imshow(labeled == 1)
        smoothed = data.smooth_segmentation(labeled)
        ax[1].imshow(smoothed[:,:,0])
        ax[2].imshow(median(labeled == 1,selem = np.ones((10,10))))
        plt.savefig(os.path.join(output_dir,"test_LabeledData_smooth_segmentation.png"))

    def test_LabeledData_get_contour_from_seg(self):
        data = LabeledData(labeled_data,additionalpath)
        frameinds = [0,40,41,42,98,84,85]
        selectid = 5
        frameselect = frameinds[selectid]
        out = data.binarize_frames(frameinds)
        clean = data.clean_binary(out)
        virgin_data = data.dataarray[frameselect,:,:,0]
        dam_data = data.dataarray[frameselect,:,:,1]
        dcent = data.get_positions(np.array([frameselect]),8)
        vcent = data.get_positions(np.array([frameselect]),3)
        labeled,markers = data.segment_animals(out[selectid],virgin_data,dam_data)
        smoothed = data.smooth_segmentation(labeled)
        contours = data.get_contour_from_seg(smoothed)
        mouse1 = contours[0][0]
        mouse2 = contours[1][0]
        plt.plot(mouse1[:,0],mouse1[:,1])
        plt.plot(mouse2[:,0],mouse2[:,1])
        plt.savefig(os.path.join(output_dir,"test_LabeledData_get_contour_from_seg.png"))

    def test_LabeledData_get_contour_fourier_rep(self):
        """NOTE: This method currently just takes the 0th entry of the contour list for each mouse. Do something smarter to eliminate the need for this index and account for other contours upstream.

        """
        data = LabeledData(labeled_data,additionalpath)
        dictentries = joblib.load(os.path.join(fixture_dir,"test_contours"))
        contourdict = {i:dictentries[i] for i in range(len(dictentries))}
        contours = data.get_contour_fourier_rep(contourdict)
    
        assert len(contours) == len(dictentries) 
        assert list(contours[0].keys()) == ["virgin","dam"]
        assert list(contours[0]["virgin"].keys()) == ["coefs","freqs"]
        inverse = np.fft.ifft(contours[0]["virgin"]["coefs"])
        traced_inverse = np.concatenate([np.real(inverse)[:,None],np.imag(inverse)[:,None]],axis = 1)
        assert np.all(np.isclose(traced_inverse,contourdict[0][0][0]))

    def test_LabeledData_find_startpoint(self):
        data = LabeledData(labeled_data,additionalpath)
        dictentries = joblib.load(os.path.join(fixture_dir,"test_contours"))
        contourdict = {i:dictentries[i] for i in range(len(dictentries))}
        contourfs = data.get_contour_fourier_rep(contourdict)
        ind = 0
        contourf = contourfs[ind]
        tips = data.dataarray[ind,:,0,:]
        cents = data.dataarray[ind,:,3,:]
        orig_contour = np.fft.ifft(contourf["dam"]["coefs"])
        plt.plot(np.real(orig_contour),np.imag(orig_contour),"bx",markersize = 0.5)
        plt.plot(np.real(orig_contour)[0],np.imag(orig_contour)[0],"bo")
        rotcontourf = data.find_startpoint(contourf,tips,cents)
        rot_contour = np.fft.ifft(rotcontourf["dam"]["coefs"])
        plt.plot(np.real(rot_contour),np.imag(rot_contour),"rx",markersize = 0.5)
        plt.plot(np.real(rot_contour)[0],np.imag(rot_contour)[0],"ro")
        plt.plot(*reversed(tips[:,1]),"x")
        plt.savefig(os.path.join(output_dir,"test_LabeledData_find_startpoint.png"))

    def test_LabeledData_center_contour(self):
        data = LabeledData(labeled_data,additionalpath)
        dictentries = joblib.load(os.path.join(fixture_dir,"test_contours"))
        contourdict = {i:dictentries[i] for i in range(len(dictentries))}
        contourfs = data.get_contour_fourier_rep(contourdict)
        ind = 0
        contourf = contourfs[ind]
        tips = data.dataarray[ind,:,0,:]
        cents = data.dataarray[ind,:,3,:]
        orig_contour = np.fft.ifft(contourf["dam"]["coefs"])
        plt.plot(np.real(orig_contour),np.imag(orig_contour),"bx",markersize = 0.5)
        centcontourf = data.center_contour(contourf,cents)
        cent_contour = np.fft.ifft(centcontourf["dam"]["coefs"])
        plt.plot(np.real(cent_contour),np.imag(cent_contour),"bx",markersize = 0.5)
        plt.savefig(os.path.join(output_dir,"test_LabeledData_center_contour.png"))

    def test_LabeledData_rotate_contour(self):
        data = LabeledData(labeled_data,additionalpath)
        dictentries = joblib.load(os.path.join(fixture_dir,"test_contours"))
        contourdict = {i:dictentries[i] for i in range(len(dictentries))}
        contourfs = data.get_contour_fourier_rep(contourdict)
        ind = 0
        contourf = contourfs[ind]
        tips = data.dataarray[ind,:,0,:]
        cents = data.dataarray[ind,:,3,:]
        centcontourf = data.center_contour(contourf,cents)
        orig_contour = np.fft.ifft(centcontourf["dam"]["coefs"])
        plt.plot(np.real(orig_contour),np.imag(orig_contour),"bx",markersize = 0.5)
        aligncontourf = data.rotate_contour(centcontourf,tips,cents)
        align_contour = np.fft.ifft(centcontourf["dam"]["coefs"])
        plt.plot(np.real(align_contour),np.imag(align_contour),"rx",markersize = 0.5)
        plt.plot(0,0,"o")
        plt.show()
        plt.savefig(os.path.join(output_dir,"test_LabeledData_rotate_contour.png"))

class Test_LineSelector():
    def test_LineSelector(self):
        fig,ax = plt.subplots()
        rline = np.random.randn(100)*100
        ax.plot(rline)
        ls = LineSelector(fig)
        ls.connect()
        plt.show()
        lines = ls.get_lines()
        plt.plot(rline)
        for l in lines:
            plt.plot(l[:,0],l[:,1])
        plt.savefig(os.path.join(output_dir,"test_LineSelector.png"))

    def test_LineSelector_get_all_points(self):
        fig,ax = plt.subplots()
        rline = np.random.randn(100)*100
        ax.plot(rline)
        ls = LineSelector(fig)
        ls.connect()
        plt.show()
        lines = ls.get_all_points()
        assert type(lines) == list
