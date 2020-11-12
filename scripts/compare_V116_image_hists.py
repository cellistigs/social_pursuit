import script_doc_utils 
import json
import matplotlib.pyplot as plt
import os
from social_pursuit.labeled import PoseDistribution,FourierDescriptor
from scipy.spatial import distance as scidist
from scipy.stats import wasserstein_distance
import numpy as np
import joblib
from joblib import Memory
cachedir = "/Volumes/TOSHIBA EXT STO/cache"
memory = Memory(cachedir, verbose=1)
experimentdictpath = "../src/social_pursuit/trace_template_V116.json"
labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"

def get_dist(fd,reference_hist,animal):
    """Given a fourier descriptor and a reference histogram, outputs the emd between them. 

    :param fd: FourierDescriptor object from which we will generate a contour.
    :param reference_hist: a reference histogram that we will be calculating the distance against. 
    :param animal: the animal we are running analysis for
    """
    contour_im = fd.get_contour_image(animal)
    assert np.any(np.isnan(contour_im))
    H = get_image_histogram(contour_im,normalize = False)
    Hnorm = H/np.sum(H)
    dist = wasserstein_distance(Hnorm.flatten(),reference_hist)
    return dist,Hnorm

def get_image_histogram(image,normalize = True):
    edges = [np.arange(0,1+1/8,1/8) for i in range(3)]
    reshaped = image.reshape(-1,3,order = "F").T
    content = reshaped[:,np.where(~np.isnan(reshaped))[1]]
    assert not np.any(np.isnan(content))
    H,edges = np.histogramdd(content.T,bins = edges,density = False)
    if normalize:
        hnorm = H/np.sum(H)
    else:
        hnorm = H
    return hnorm

@memory.cache
def create_pose_distribution(nb_components = 10):
    """Gets out the trained contours that you created previously.

    """
    contourpath = "./trained_contours/"
    all_trained_fds = os.listdir(contourpath)
    iterations = 2000
    virgin_fds = {} 
    dam_fds = {}
    for fd in all_trained_fds:
        if fd.endswith(str(iterations)):
            filecomps = fd.split("_")
            frame = int(filecomps[3].split("frame")[-1])
            animal = filecomps[0]
            if animal == "virgin":
                virgin_fds[frame] = fd
            elif animal == "dam":
                dam_fds[frame] = fd
        else:
            pass

    max_ind = np.max([*list(virgin_fds.keys()),*list(dam_fds.keys())])
    all_improved_contours = {}

    for i in range(max_ind+1): 
        vfdpath,dfdpath =  virgin_fds.get(i,False),dam_fds.get(i,False)
        if vfdpath and dfdpath:
            vfd = joblib.load(os.path.join(contourpath,vfdpath))
            dfd = joblib.load(os.path.join(contourpath,dfdpath))
            fd_orig = vfd["fd"]
            vtrained = fd_orig.superimpose_centered_contour("virgin",np.fft.irfft(vfd["trained_contour_fft"],axis = -1))
            dtrained = fd_orig.superimpose_centered_contour("dam",np.fft.irfft(dfd["trained_contour_fft"],axis = -1))

        ## Now, we will replace the contours originally saved with the fd:
        fd_orig.contours["virgin"] = vtrained.T
        fd_orig.contours["dam"] = dtrained.T
        all_improved_contours[i] = fd_orig
    pd_improved = PoseDistribution(all_improved_contours,nb_components)

    return pd_improved 


if __name__ == "__main__":
    md = script_doc_utils.initialize_doc({"prev":"make_V116_fds"})
    md.new_header(title = "Summary",level = 1)
    md.new_paragraph("This code implements application 1 of the cosyne abstract goals: create image-based validation methods. In a large number of contexts, it is extremely difficult to validate the accuracy of an animal tracking algorithm, or subsequent postprocessing steps. Especially on large datasets, it can be prohibitive to examine the results of a tracking algorithm by eye. Here, we use image contours to generate validation metrics based on the image underneath the detected contour. By first translating the detected points into an image contour, we can get a holistic measure of how well the detected pose captures the whole of the animal detected beneath it. By using an image based validation method, we can capture the invariance in the animal's appearance as a whole, even as individual parts move and shift.")
    md.new_paragraph("To initialize our validation method, we first create an color intensity histogram for individual animals from our training data. This is a three dimensional histogram for 3D data. Then, for each frame of a pursuit behavior, we generate an analogous color intensity histogram and measure the wasserstein distance between this histogram and the reference histogram. The use of wasserstein on image histograms is common in the image processing literature. Finally, we quantify the per-pursuit distance in each pursuit event as a time series, and show some example frames where we detect a jumps in the wasserstein distance.")

    md.new_paragraph("First, get out the histograms from the training set:")

    ## See make_V116_fds for an in depth explanation of this function
    pd = create_pose_distribution(10)

    ## get the fds:
    all_fds = pd.all_fds
    
    @memory.cache
    def get_reference_histogram(all_fds):
        """Get the 3D reference histogram giving the likelihood of certain pixel intensities in the image.

        """
        ## Initialize histogram parameters:
        edges = [np.arange(0,1+1/8,1/8) for i in range(3)]

        all_H = {"virgin":[],"dam":[]} 

        for fdind,fd in all_fds.items():
            for animal in ["virgin","dam"]:
                image = fd.get_contour_image(animal)
                H = get_image_histogram(image,normalize = False)
                all_H[animal].append(H)
        total_H = {animal:np.sum(np.array(all_H[animal]),axis = 0) for animal in all_H}
        ## These are the color histograms we will compare against for each animal. 
        dist_H = {animal:tot_H/np.sum(tot_H) for animal,tot_H in total_H.items()} 
        return dist_H,all_H

    dist_H,all_H = get_reference_histogram(all_fds)

    tracepath = "/Volumes/TOSHIBA EXT STO/Legacy_Traces/V116_03082018/pursuits/part27/interval31087_31198/pursuit_fds"

    md.new_paragraph("We now have our 3D training data color histograms in hand. We will go frame by frame and compare these to the data we get from the raw data. First lets chooose a reference pursuit to work with: we like: {}".format(tracepath))
    pursuit_fds = joblib.load(tracepath)

    @memory.cache
    def get_all_emd_dists(tracepath,dist_H):
        """Computes the EMD distance between each contour in a pursuit sequence and 1) the average histogram constructed from the training set and 2) the previous frame. 

        :param tracepath: path to a joblib file containing a dictionary of fd objects, each representing a frame for the pursuit traces we want to apply this analysis to. 
        :param dist_H: a dictionary with keys representing virgin and dam, and values representing 3d histograms of color values. 

        """
        pursuit_fds = joblib.load(tracepath)

        animal_dists = {"virgin":[],"dam":[]}
        animal_dists_tmp = {"virgin":[],"dam":[]}
        animal_dists_sd = {"virgin":[],"dam":[]}
        animal_dists_tmp_sd = {"virgin":[],"dam":[]}
        for animal in animal_dists:
            Hpre = dist_H[animal].flatten()
            Hpre_tmp_sd = dist_H[animal].flatten()
            animalid = {"virgin":0,"dam":1}[animal]
            for index in pursuit_fds:
                orig_fd = pursuit_fds[index]["fd_orig"]
                fd_sd = pursuit_fds[index]["fd_sd"]

                dist,Hnorm = get_dist(orig_fd,dist_H[animal].flatten(),animal)
                dist_tmp,Hnorm_tmp = get_dist(orig_fd,Hpre,animal)
                dist_sd,Hnorm_sd = get_dist(fd_sd,dist_H[animal].flatten(),animal)
                dist_tmp_sd,Hnorm_tmp_sd = get_dist(fd_sd,Hpre_tmp_sd,animal)

                animal_dists[animal].append(dist)
                animal_dists_tmp[animal].append(dist_tmp)
                animal_dists_sd[animal].append(dist_sd)
                animal_dists_tmp_sd[animal].append(dist_tmp_sd)

                Hpre = Hnorm_tmp.flatten() 
                Hpre_tmp_sd = Hnorm_tmp_sd.flatten() 
        return animal_dists,animal_dists_tmp,animal_dists_sd,animal_dists_tmp_sd
        
    animal_dists,animal_dists_tmp,animal_dists_sd,animal_dists_tmp_sd = get_all_emd_dists(tracepath,dist_H)

    for animal in animal_dists:
        animalid = {"virgin":0,"dam":1}[animal]
        maxind = np.argmax(animal_dists[animal])
        maxfd = pursuit_fds[maxind]["fd_orig"]
        
        fig,ax = plt.subplots(1,2,figsize = (10,3))

        ax[0].plot(animal_dists[animal])
        ax[0].axvline(maxind,color = "black",linestyle = "--")
        ax[1].imshow(maxfd.get_contour_image(animal)[300:,:250])
        ax[1].plot(*maxfd.points[:,:,animalid]-np.array([[0],[300]]),"x")
        ax[1].plot(maxfd.contours[animal][:,1],maxfd.contours[animal][:,0]-300)
        ax[0].set_title("EMDs for 3D color histograms ({})".format(animal))
        ax[1].set_title("Frame {}".format(maxind))
        ax[1].axis("off")
        script_doc_utils.save_and_insert_image(md,fig,"../docs/script_docs/images/toy_wass_histogram{}.png".format(animal))
        md.new_paragraph("In this first example, we show the color histograms for the virgin animal.")

        maxind = np.argmax(animal_dists_tmp[animal][1:])+1
        maxfd = pursuit_fds[maxind]["fd_orig"]
        premaxfd = pursuit_fds[maxind-1]["fd_orig"]
        
        fig,ax = plt.subplots(1,3,figsize = (10,3))

        ax[0].plot(animal_dists_tmp[animal])
        ax[0].axvline(maxind,color = "black",linestyle = "--")
        ax[1].imshow(premaxfd.get_contour_image(animal))
        ax[2].imshow(maxfd.get_contour_image(animal))
        ax[1].plot(*premaxfd.points[:,:,animalid],"x")
        ax[1].plot(premaxfd.contours[animal][:,1],premaxfd.contours[animal][:,0])
        ax[2].plot(*maxfd.points[:,:,animalid],"x")
        ax[2].plot(maxfd.contours[animal][:,1],maxfd.contours[animal][:,0])
        ax[0].set_title("EMDs for 3D color histograms (Sequential) ({})".format(animal))
        ax[1].axis("off")
        ax[2].set_title("Frame {}".format(maxind))
        ax[2].axis("off")
        script_doc_utils.save_and_insert_image(md,fig,"../docs/script_docs/images/toy_wass_histogram{}_tmp.png".format(animal))

        maxind = np.argmax(animal_dists_sd[animal]) ## excludethe first entry
        maxfd = pursuit_fds[maxind]["fd_sd"]
        
        fig,ax = plt.subplots(1,2,figsize = (10,3))

        ax[0].plot(animal_dists_sd[animal])
        ax[0].axvline(maxind,color = "black",linestyle = "--")
        ax[1].imshow(maxfd.get_contour_image(animal)[300:,:250])
        ax[1].plot(*maxfd.points[:,:,animalid]-np.array([[0],[300]]),"x")
        ax[1].plot(maxfd.contours[animal][:,1],maxfd.contours[animal][:,0]-300)
        ax[0].set_title("EMDs for 3D color histograms (Social Dataset Filtering)({})".format(animal))
        ax[1].set_title("Frame {}".format(maxind))
        ax[1].axis("off")
        script_doc_utils.save_and_insert_image(md,fig,"../docs/script_docs/images/toy_wass_histogram{}_sg.png".format(animal))

        maxind = np.argmax(animal_dists_tmp_sd[animal][1:])+1
        maxfd = pursuit_fds[maxind]["fd_sd"]
        premaxfd = pursuit_fds[maxind-1]["fd_sd"]
        
        fig,ax = plt.subplots(1,3,figsize = (10,3))

        ax[0].plot(animal_dists_tmp_sd[animal])
        ax[0].axvline(maxind,color = "black",linestyle = "--")
        ax[1].imshow(premaxfd.get_contour_image(animal))
        ax[1].plot(*premaxfd.points[:,:,animalid],"x")
        ax[1].plot(premaxfd.contours[animal][:,1],premaxfd.contours[animal][:,0])
        ax[2].imshow(maxfd.get_contour_image(animal))
        ax[2].plot(*maxfd.points[:,:,animalid],"x")
        ax[2].plot(maxfd.contours[animal][:,1],maxfd.contours[animal][:,0])
        ax[0].set_title("EMDs for 3D color histograms (Sequential, SD filtered) ({})".format(animal))
        ax[2].set_title("Frame {}".format(maxind))
        ax[1].axis("off")
        ax[2].axis("off")
        script_doc_utils.save_and_insert_image(md,fig,"../docs/script_docs/images/toy_wass_histogram{}_tmp_sd.png".format(animal))

    md.new_paragraph("We can see that this works really well. Once properly normalized, using the EMD is a great way to detect anomalies in the whole appearance of the animal, as generated through the image histogram. Now let's aggregate this information across all of the pursuit events that we have.") 

    plot_all = False
    if plot_all:

        with open(experimentdictpath,"r") as f:
            experimentdata = json.load(f)

        localpath = experimentdata["trace_directory"]

        experiments = {f["ExperimentName"]:f for f in experimentdata["traces_expected"]} 

        for experiment in experiments:
            experimentinfo = experiments[experiment]
            pursuitpath = os.path.join(localpath,experiment,"pursuits")
            partpaths = [os.path.join(pursuitpath,d) for d in os.listdir(pursuitpath) if os.path.isdir(os.path.join(pursuitpath,d))]
            for partpath in partpaths:
                segmentpaths = [os.path.join(partpath,d) for d in os.listdir(partpath) if os.path.isdir(os.path.join(partpath,d))]
                for segmentpath in segmentpaths:
                    try:
                        animal_dists,animal_dists_tmp,animal_dists_sd,animal_dists_tmp_sd = get_all_emd_dists(os.path.join(segmentpath,"pursuit_fds"),dist_H)
                        all_dists = {"base":animal_dists,"tmp":animal_dists_tmp,"sd":animal_dists_sd,"tmp_sd":animal_dists_tmp_sd}
                        joblib.dump(all_dists,os.path.join(segmentpath,"emd_dists"))
                        

                        print(segmentpath)
                    except FileNotFoundError:
                        print("{} Not Found".format(segmentpath))
                        continue


    
    md.create_md_file()


