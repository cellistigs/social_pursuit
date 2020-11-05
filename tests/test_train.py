# Test the labeled data module
import os
import joblib
import numpy as np
import jax
import jax.numpy as jnp
import copy
import pytest
import pathlib
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage import color
#from botocore.stub import Stubber
from social_pursuit.labeled import LabeledData,LineSelector,FourierDescriptor,rotate_coefficients,calculate_orientation
from social_pursuit.train import Contour_Optimizer,place_gaussian,place_gaussians
import social_pursuit.data
import json

output_dir = pathlib.Path("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/tempdir")
fixture_dir = pathlib.Path("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/test_dir")

labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"

class Test_Contour_Optimizer():
    def test_Contour_Optimizer(self):
        all_fds = joblib.load(os.path.join(fixture_dir,"all_fds_elliptic_5_elements"))
        pcadict = joblib.load(os.path.join(fixture_dir,"pcadict_elliptic_5_elements_3_components"))
        data = LabeledData(labeled_data,additionalpath)

        animal = "dam"
        mean,cov = data.get_gaussian_contour_pose_distribution(animal,pcadict[animal]["weights"],all_fds)
        co = Contour_Optimizer(mean,cov,pcadict[animal]["pca"])
        weights = pcadict[animal]["weights"][0]


        assert np.all(np.abs(co.pca_imap(weights) - pcadict[animal]["pca"].inverse_transform(weights))<1e-3)
        assert np.all(np.abs(co.pca_map(co.pca_imap(weights))-weights < 1e-5))

    def test_eval_posterior(self):
        all_fds = joblib.load(os.path.join(fixture_dir,"all_fds_elliptic_5_elements"))
        pcadict = joblib.load(os.path.join(fixture_dir,"pcadict_elliptic_5_elements_3_components"))
        data = LabeledData(labeled_data,additionalpath)

        animal = "dam"
        animalind = {"dam":1,"virgin":0}[animal]
        mean,cov = data.get_gaussian_contour_pose_distribution(animal,pcadict[animal]["weights"],all_fds)
        co = Contour_Optimizer(mean,cov,pcadict[animal]["pca"])
        weights = pcadict[animal]["weights"][0]
        points = all_fds[0].process_points()
        post_eval_func = co.eval_posterior(points[:,:,animalind],mean,cov)
        assert post_eval_func((weights))-1 < 1e-6

        from_other = data.get_MAP_weights(mean,cov,points[None,:,:,animalind])

    def test_place_contour(self):
        all_fds = joblib.load(os.path.join(fixture_dir,"all_fds_elliptic_5_elements"))
        pcadict = joblib.load(os.path.join(fixture_dir,"pcadict_elliptic_5_elements_3_components"))
        data = LabeledData(labeled_data,additionalpath)

        animal = "virgin"
        animal_nb = {"dam":1,"virgin":0}[animal]
        mean,cov = data.get_gaussian_contour_pose_distribution(animal,pcadict[animal]["weights"],all_fds)
        co = Contour_Optimizer(mean,cov,pcadict[animal]["pca"])

        descriptor = all_fds[0]

        contour = data.get_contour_from_pc(pcadict[animal]["pca"],pcadict[animal]["weights"][0:1,:])[0]
        
        angle = descriptor.mouseangles[animal_nb]
        points = descriptor.points[:,3,animal_nb]

        eval_image_contour = co.place_contour((points,angle))
        print(eval_image_contour(contour).shape)

        reference = descriptor.superimpose_centered_contour(animal,contour)
        fig,ax = plt.subplots(2,1)
        ax[0].plot(*reference)
        ax[1].plot(*eval_image_contour(contour))
        plt.savefig(os.path.join(output_dir,"test_place_contour.png"))

        assert np.all(np.abs(eval_image_contour(contour) - reference) < 1e-3)
        

    def test_eval_image(self):
        all_fds = joblib.load(os.path.join(fixture_dir,"all_fds_elliptic_5_elements"))
        pcadict = joblib.load(os.path.join(fixture_dir,"pcadict_elliptic_5_elements_3_components"))
        data = LabeledData(labeled_data,additionalpath)

        animal = "virgin"
        animal_nb = {"dam":1,"virgin":0}[animal]
        mean,cov = data.get_gaussian_contour_pose_distribution(animal,pcadict[animal]["weights"],all_fds)
        co = Contour_Optimizer(mean,cov,pcadict[animal]["pca"])

        fft = co.pca_imap(pcadict[animal]["weights"][0,:])
        fft_reshape = fft.reshape(2,-1)
        featurelength = int(fft_reshape.shape[-1]/2)
        fft_reshape_complex = fft_reshape[:,:featurelength] +1j*fft_reshape[:,featurelength:]
        contour_reconstruct = jnp.fft.irfft(fft_reshape_complex)

        descriptor = all_fds[0]
        angle = descriptor.mouseangles[animal_nb]
        points = descriptor.points[:,3,animal_nb]

        eval_image_contour = co.place_contour((points,angle))

        contour = data.get_contour_from_pc(pcadict[animal]["pca"],pcadict[animal]["weights"][0:1,:])[0]
        
        reference = descriptor.superimpose_centered_contour(animal,contour)

        image = descriptor.image
        eval_image = co.eval_image((points,angle),image.shape[:2],10)

        fig,ax = plt.subplots(3,1)
        ax[0].imshow(eval_image(fft_reshape_complex))
        ax[1].imshow(image)
        ax[2].plot(*eval_image_contour(contour))
        plt.savefig(os.path.join(output_dir,"test_eval_image.png"))

        assert np.all(np.abs(eval_image_contour(contour) - reference) < 1e-3)

    def test_train(self):
        all_fds = joblib.load(os.path.join(fixture_dir,"all_fds_elliptic_5_elements"))
        pcadict = joblib.load(os.path.join(fixture_dir,"pcadict_elliptic_5_elements_3_components"))
        data = LabeledData(labeled_data,additionalpath)

        animal = "dam"
        animal_nb = {"dam":1,"virgin":0}[animal]
        mean,cov = data.get_gaussian_contour_pose_distribution(animal,pcadict[animal]["weights"],all_fds)
        co = Contour_Optimizer(mean,cov,pcadict[animal]["pca"])

        contour_ind = 1

        fft = co.pca_imap(pcadict[animal]["weights"][contour_ind,:])

        descriptor = all_fds[contour_ind]
        image_crop = {"y":(200,370),"x":(0,70)}

        vals,contours = co.train(descriptor,fft,animal,10,image_crop)

        assert 0

def test_place_gaussian():
    coord = np.array([10,10])
    dims = (30,50)
    sigma = 5
    output = place_gaussian(coord,dims,sigma)
    assert np.min(output) >= 0
    assert np.max(output) <= 1000
    plt.imshow(output)
    plt.savefig(os.path.join(output_dir,"test_place_gaussian.png"))


def test_place_gaussians():
    coords = np.stack([np.linspace(0,10),np.linspace(50,20)],axis = 1)
    dims = (100,50)
    sigma = 0.1 
    output = place_gaussians(coords,dims,sigma)
    assert np.min(output) >= 0
    assert np.max(output) <= 1000
    plt.imshow(output)
    plt.savefig(os.path.join(output_dir,"test_place_gaussians.png"))
        














