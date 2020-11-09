'''
A dataset class to hold training data, specifically. Useful to calculate
statistics on the training data to consider as ground truth going forwards.
'''

import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage.filters import threshold_yen,try_all_threshold,gaussian,median
from skimage.morphology import binary_closing,binary_opening,remove_small_holes
from skimage.segmentation import watershed
from skimage import measure
from skimage import color
from sklearn.decomposition import PCA
from social_pursuit.data import mkdir_notexists
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
import os,sys
import tqdm
import json

def rotate_coefficients(coefficients,b):
    """Given the fourier coefficients of a periodic time series, gives the coefficents for the same series starting from the bth entry.

    """
    n = len(coefficients)
    rotation_factor = np.exp(2*np.pi*1j*b*np.linspace(0,n-1,n)/n)
    rot_coefs = coefficients*rotation_factor
    diff_init = np.sum(rot_coefs)/n
    return rot_coefs,diff_init

def calculate_orientation(contour):
    """Given a contour (assuming closed) calculates the orientation in which it is rotating about the mean point. 
    If the sum of differences is positive, indicates most rotation is happening counterclockwise, and vice versa. 
    :param contour: the animal contour, given as a (n,2) numpy array. 

    """
    mean = np.mean(contour,axis = 0)    
    vectors = contour - mean
    ## Cast to complex to easily calculate angle. 
    compvec = vectors[:,0]+1j*vectors[:,1]
    angles = (np.angle(compvec)+np.pi)%(2*np.pi)
    ## The periodic boundary gives us some issues. Neglect the largest difference:
    diffs = np.diff(angles)
    discont = np.argmax(np.abs(diffs))
    diffs[discont] = 0
    ang_diffs = np.sum(diffs)
    return ang_diffs

class LineSelector():
    """Class to draw lines via click and drag. 

    """
    def __init__(self,fig):
        self.fig = fig
        self.lines = [] 
        self.press = None
    

    def connect(self):
        """
        Connect to required mpl events
        """
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)

    def disconnect(self):
        """Disconnect from required mpl events

        """
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)

    def on_press(self,event): 
        self.press = [np.round(event.xdata),np.round(event.ydata)]

    def on_release(self,event): 
        end = [np.round(event.xdata),np.round(event.ydata)]
        line = self.interpolate(np.array(self.press),np.array(end))
        self.press = None

        self.lines.append(line)

    def interpolate(self,p1,p2):
        print(p1,p2,"p1,p2")
        length = int(np.linalg.norm(p1-p2))
        spacing = np.linspace(p1,p2,length).astype(int)
        return spacing

    def get_lines(self):
        return self.lines

    def get_all_points(self):
        """Collapse all lines into a flat list of points.

        """
        all_lines = np.concatenate(self.lines,axis = 0)
        return all_lines.tolist()

class LabeledData(object):
    """An object to deal with trained label poses. Useful to calculate statistics on the training data, simulate from the training data for validation datasets, etc.  

    :param datapath: path to the h5 file containing manual labels.
    :param additionalpath: path to the folder that contains individual image folders. 
    """
    def __init__(self,datapath,additionalpath):
        self.data = pd.read_hdf(datapath)
        self.dataarray = self.get_dataarray()
        self.dataname = datapath.split('.h5')[0]
        self.scorer = 'Taiga'
        self.part_list = ['vtip','vlear','vrear','vcent','vtail','mtip','mlear','mrear','mcent','mtail']
        self.part_index = np.arange(len(self.part_list))
        self.part_dict = {index:self.part_list[index] for index in range(len(self.part_list))}
        self.size = self.data.shape[0]
        self.datamapping = self.datasets_indices()
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.additionalpath = additionalpath

    def get_dataarray(self):
        """Create a data array for easy array operations.

        """
        mice = np.split(self.data.values,2,axis = 1)

        micestack = np.stack(mice,axis = -1)
        # Now of shape(time,10,mice)
        parts = np.split(micestack,np.arange(2,10,2),axis =1) 
        partstack = np.stack(parts,axis = -2)
        return partstack

    def get_dataarray_nonan(self):
        """Get just the part of the data that does not have nans.

        """
        nans = np.where(np.isnan(self.dataarray))[0]
        nantimes = np.unique(nans)
        notnan = [i for i in np.arange(len(self.dataarray)) if i not in nantimes]
        return self.dataarray[np.array(notnan),:,:,:]

    def set_cropping(self,xmin,xmax,ymin,ymax):
        """
        Correctly crop the window to match up detected points with cropped frames:
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def datasets_indices(self):
        """
        Looks at the h5 file, extracts the paths to individual images, and infers the list of folders that contain image data. Furthermore, gives the range of indices from the h5 file that correspond to images in each folder. 
        :return: A dictionary with keys giving the names of each folder, and values giving an array of indices, telling us which entries in the h5 file belong to experiments in which folder.
        """
        allpaths = self.get_imagenames(range(self.size))
        ## Find unique folder names:
        datafolders = [datafolder.split('/')[-2] for datafolder in allpaths]
        unique = list(set(datafolders))

        ## Return separate index sets for each folder:
        datamapping = {}
        for folder in unique:
            folderdata = np.array([i for i in range(self.size) if datafolders[i] in folder])
            datamapping[folder] = folderdata
        return datamapping

## Atomic actions: selecting entries of training data.
    def get_imagenames(self,indices):
        """Given a range of indices corresponding to timepoints in the training data, returns the actual paths to the images (up to the parent folder)
        :param indices: an array like of indices to get the paths for. 
        :return: a corresponding list of paths to the images that were labeled.
        """
        ids = self.data.index.tolist()
        relevant = [ids[i] for i in indices]
        #relevant = [id for i,id in enumerate(ids) if i in indices]
        return relevant

    def get_positions(self,indices,part):
        """Given a range of time indices, and the integer index of a part, returns the positions of that part at the given range of indices 

        :param indices: a numpy array of the indices in the training data you care about.
        :param part: an integer corresponding to the animals body part at any given point in time. 
        """
        assert part < 10
        assert type(part) in [int,np.int64,np.int32]
        m = np.floor(part//5).astype(int)
        p = part-m*5
        position = self.dataarray[indices,:,p,m]
        return position


## Define skeleton statistic functions:
    def distances(self,indices,part0,part1):
        """Given a range of time indices and a pair of part indices, returns the distances between the given parts at the given range of indices.

        :param indices: a numpy array of the indices in the training data you care about. 
        :param part0: an integer corresponding to the first body part you care to retrieve at any given time. 
        :param part1: an integer corresponding to the second body part you care to retrieve at any given time. 
        """
        positions0 = self.get_positions(indices,part0)
        positions1 = self.get_positions(indices,part1)
        dists = np.linalg.norm(positions0-positions1,axis = 1)
        return dists

    def distances_mean(self,indices,part0,part1):
        """Given a range of time indices and a pair of part indices, returns the mean distance between the given parts across the given range of indices.

        :param indices: a numpy array of the indices in the training data you care about. 
        :param part0: an integer corresponding to the first body part you care to retrieve at any given time. 
        :param part1: an integer corresponding to the second body part you care to retrieve at any given time. 
        """
        dists = self.distances(indices,part0,part1)
        mean = np.nanmean(dists)
        return mean

    def distances_std(self,indices,part0,part1):
        """Given a range of time indices and a pair of part indices, returns the standard deviation of the distance between the given parts across the given range of indices.

        :param indices: a numpy array of the indices in the training data you care about. 
        :param part0: an integer corresponding to the first body part you care to retrieve at any given time. 
        :param part1: an integer corresponding to the second body part you care to retrieve at any given time. 
        """
        dists = self.distances(indices,part0,part1)
        mean = np.nanstd(dists)
        return mean

    def distances_hist(self,indices,part0,part1,bins=None):
        """Given a range of time indices and a pair of part indices, returns histograms of the distance between the given parts across the given range of indices.

        :param indices: a numpy array of the indices in the training data you care about. 
        :param part0: an integer corresponding to the first body part you care to retrieve at any given time. 
        :param part1: an integer corresponding to the second body part you care to retrieve at any given time. 
        :param bins: (optional) representation of the histogram bins. Takes the same representation as the bins argument of np.histogram.
        """
        dists = self.distances(indices,part0,part1)
        dists = dists[~np.isnan(dists)]
        if bins is not None:
            hist,edges = np.histogram(dists,bins)
        else:
            hist,edges = np.histogram(dists)
        return hist,edges

## Define iteration over all pairwise for a mouse:
    def distance_wholemouse_matrix(self,indices,mouse):
        """Given a range of time indices and a pair of part indices, returns distance matrices giving the euclidean distance between each pair of points..

        :param indices: a numpy array of the indices in the training data you care about. 
        :param mouse: can be 0 or 1, indicating the virgin or dam.
        :return: an array of shape (len(indices),5,5) giving a set of symmetric pairwise matrices.
        """
        data_segment = self.dataarray[indices,:,:,mouse]
        ## Subtract the values per coordinate
        coordinatewise_diff = data_segment[:,:,:,None]-data_segment[:,:,None,:]
        norm = np.linalg.norm(coordinatewise_diff,axis = 1)
        return norm

    def distances_wholemouse(self,indices,mouse):
        """Given a range of time indices and a pair of part indices, returns dictionary representation of distance between points.

        :param indices: a numpy array of the indices in the training data you care about. 
        :param mouse: can be 0 or 1, indicating the virgin or dam.
        :return: an dictionary giving pairwise distances between points.
        """
        assert mouse in [0,1]
        norm = self.distance_wholemouse_matrix(indices,mouse)
        id_0 = np.arange(5)+mouse
        pairwise_dists = {}
        for p,j in enumerate(id_0):
            for i in id_0[:p]:
                pairwise_dists[(j,i)] = norm[:,j,i]
        return pairwise_dists

## Define iteration over all pairwise for a mouse:
    def stats_wholemouse(self,indices,mouse):
        """Given a range of time indices and a pair of part indices, returns dictionary representation of distance between points.

        :param indices: a numpy array of the indices in the training data you care about. 
        :param mouse: can be 0 or 1, indicating the virgin or dam.
        :return: an dictionary giving mean and standard deviation of the distance between points across the given interval.
        """
        assert mouse in [0,1]
        norm = self.distance_wholemouse_matrix(indices,mouse)
        norm_mean = np.mean(norm,axis = 0)
        norm_std = np.std(norm,axis = 0)
        id_0 = np.arange(5)+5*mouse
        pairwise_dists = {}
        for p,j in enumerate(id_0):
            for i in id_0[:p]:
                mean = norm_mean[j,i] 
                std = norm_std[j,i]
                pairwise_dists[(j,i)] = (mean,std)
        return pairwise_dists

    def hists_wholemouse(self,indices,mouse,bins = None):
        id_0 = np.arange(5)+5*mouse
        pairwise_hists = {}
        for p,j in enumerate(id_0):
            for i in id_0[:p]:
                hist = self.distances_hist(indices,j,i,bins)
                pairwise_hists[(j,i)] = hist
        return pairwise_hists

## Likewise for both mice, for a single dataset:
    def distances_dataset(self,dataset):
        indices = self.datamapping[dataset]
        outmice = []
        for mouse in range(2):
            out = self.distances_wholemouse(indices,mouse)
            outmice.append(out)
        return outmice

    def stats_dataset(self,dataset):
        indices = self.datamapping[dataset]
        outmice = []
        for mouse in range(2):
            out = self.stats_wholemouse(indices,mouse)
            outmice.append(out)
        return outmice

    def hists_dataset(self,dataset,bins = None):
        indices = self.datamapping[dataset]
        outmice = []
        for mouse in range(2):
            out = self.hists_wholemouse(indices,mouse,bins)
            outmice.append(out)
        return outmice


## Likewise for both mice, for all datapoints:
    def distances_all(self):
        indices = np.arange(self.size)
        outmice = []
        for mouse in range(2):
            out = self.distances_wholemouse(indices,mouse)
            outmice.append(out)
        return outmice

    def stats_all(self):
        indices = np.arange(self.size)
        outmice = []
        for mouse in range(2):
            out = self.stats_wholemouse(indices,mouse)
            outmice.append(out)
        return outmice

    def hists_all(self,bins = None):
        indices = np.arange(self.size)
        outmice = []
        for mouse in range(2):
            out = self.hists_wholemouse(indices,mouse,bins)
            outmice.append(out)
        return outmice

## Done with training data statistics. Now consider how we would create surrogate data for testing.  
    def sample(self,n):
        """Sample n labeled datapoints from the training set. Returned as a numpy array of shape(n,coordinate,part,mouse) 

        """
        indices = np.random.choice(self.size,n,replace = True)
        return self.dataarray[indices,:,:,:]

    def sample_radius(self,n,r):
        """Sample n labeled training points, but hold the distance r between them fixed.  

        """
        samples_raw = self.sample(n)
        ## Estimate the distance between the two animals via their measured centroids. 
        mean_pos = np.nanmean(samples_raw,axis = 2)
        dir_vecs = np.diff(mean_pos,axis=-1)
        lengths = np.linalg.norm(dir_vecs,axis = 1,keepdims = True)
        unit_length = dir_vecs/lengths
        displacement = (r-lengths)*unit_length 
        print(np.linalg.norm(displacement,axis = -1))
        samples_raw[:,:,:,1:]+=displacement[:,:,None,:]
        return samples_raw
        

    def sample_radius_orientation(self,n,r):
        """Sample n labeled training points, but hold the distance r between them fixed and randomly rotate one around the other. 

        """
        samples_raw = self.sample(n)
        ## Estimate the distance between the two animals via their measured centroids. 
        mean_pos = np.nanmean(samples_raw,axis = 2)
        dir_vecs = np.diff(mean_pos,axis=-1)
        lengths = np.linalg.norm(dir_vecs,axis = 1,keepdims = True)
        unit_length = dir_vecs/lengths

        thetas = np.random.rand(n)*2*np.pi
        rot = lambda t: np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        rot_mats_t = np.array([rot(t).T for t in thetas])

        rot_lengths = np.einsum("...j,...jk",np.squeeze(unit_length),rot_mats_t)
        displacement = r*rot_lengths[:,:,None]-lengths*unit_length 
        samples_raw[:,:,:,1:]+=displacement[:,:,None,:]
        return samples_raw
    
    def switch_points(self,dataset,indices,p):
        """Given a sampled dataset of shape (n,coordinate,part,mouse), switches p randomly selected body parts at each index with the corresponding body part on the other animal.

        """
        #fig,ax = plt.subplots(2,1,sharex = True,sharey = True)
        to_switch = np.array([np.repeat(np.random.choice(5,p,replace = False),2) for i in indices])
        partarray = np.stack([indices]*2*p,axis = -1)
        #ax[0].plot(dataset[indices[0],0,:,0],dataset[indices[0],1,:,0],"bx")
        #ax[0].plot(dataset[indices[0],0,:,1],dataset[indices[0],1,:,1],"rx")
        animalarray = np.stack([np.array([0,1]*p)]*len(indices),axis = 0)
        print(to_switch[0],animalarray[0],partarray[0])
        dataset[partarray,:,to_switch,animalarray] = dataset[partarray,:,to_switch,np.fliplr(animalarray)]
        #ax[1].plot(dataset[indices[0],0,:,0],dataset[indices[0],1,:,0],"bo")
        #ax[1].plot(dataset[indices[0],0,:,1]+0.5,dataset[indices[0],1,:,1]+0.5,"ro")
        #fig.savefig("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/tempdir/flipimage")

        return dataset

    def steal_points(self,dataset,indices,p):
        """Given a sampled dataset of shape (n,coordinate,part,mouse), steals p randomly selected body parts at each index from one animal, and assigns to the corresponding body part on the other animal.

        """
        #fig,ax = plt.subplots(2,1,sharex = True,sharey = True)
        to_switch = np.array([np.random.choice(5,p,replace = False) for i in indices])
        partarray = np.stack([indices]*p,axis = -1)
        #ax[0].plot(dataset[indices[0],0,:,0],dataset[indices[0],1,:,0],"bx")
        #ax[0].plot(dataset[indices[0],0,:,1],dataset[indices[0],1,:,1],"rx")
        stealing = np.random.choice(2)
        animalarray = np.array([np.repeat(stealing,p) for i in indices])
        #animalarray = np.stack([np.array([0,1]*p)]*len(indices),axis = 0)
        dataset[partarray,:,to_switch,np.abs(1-animalarray)] = dataset[partarray,:,to_switch,animalarray]
        #ax[1].plot(dataset[indices[0],0,:,0],dataset[indices[0],1,:,0],"bo")
        #ax[1].plot(dataset[indices[0],0,:,1]+0.5,dataset[indices[0],1,:,1]+0.5,"ro")
        #fig.savefig("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/tempdir/flipimage")

        return dataset
        


    def remove_animal(self,dataset,indices):
        """Given a sampled dataset of shape (n,coordinate,part,mouse), removes the detetions for one of the animals, and substitutes in a jittered version of the other animal. 

        """
        to_switch = np.array([np.arange(5) for i in indices])
        partarray = np.stack([indices]*5,axis = -1)
        #ax[0].plot(dataset[indices[0],0,:,0],dataset[indices[0],1,:,0],"bx")
        #ax[0].plot(dataset[indices[0],0,:,1],dataset[indices[0],1,:,1],"rx")
        stealing = np.random.choice(2)
        animalarray = np.array([np.repeat(stealing,5) for i in indices])
        #animalarray = np.stack([np.array([0,1]*p)]*len(indices),axis = 0)
        dataset[partarray,:,to_switch,np.abs(1-animalarray)] = dataset[partarray,:,to_switch,animalarray]
        return dataset

## Done with sampling. Now consider image statistic functions:
## We assume that if all of the data folders are not subfolders in the current
## directory, that they are at least packaged together.

    def get_images(self,indices):
        """Gets the images at indicated indices as numpy arrays. 

        :param indices: Numpy array of indices corresponding to training data.
        """
        imagenames = self.get_imagenames(indices)
        ## Check if the images are somewhere else:
        if self.additionalpath is None:
            pass
        else:
            imagenames = [self.additionalpath+img for img in imagenames]

        ## Now we will load the images:
        images = [plt.imread(image) for image in imagenames]

        return images

    
    def convert_images_grayscale(self,images):
        """Converts images at indicated indices to grayscale (planar numpy arrays)

        :param images: list of images to convert to grayscale 
        """
        ## Now we will load the images:
        gimages = [color.rgb2gray(image) for image in images]

        return gimages
    
    def get_images_grayscale(self,indices):
        """Gets the images at indicated indices as grayscale (planar numpy arrays)

        :param indices: Numpy array of indices corresponding to training data.
        """
        imagenames = self.get_imagenames(indices)
        ## Check if the images are somewhere else:
        if self.additionalpath is None:
            pass
        else:
            imagenames = [self.additionalpath+img for img in imagenames]

        ## Now we will load the images:
        images = [color.rgb2gray(plt.imread(image)) for image in imagenames]

        return images

    def make_patches(self,indices,part,radius,):
        points = self.get_positions(indices,part)
        xcents,ycents = points[:,0],points[:,1]
        images = self.get_images(indices)
        all_clipped = np.zeros((len(indices),2*radius,2*radius,3)).astype(np.uint8)
        for i,image in enumerate(images):
            ysize,xsize = image.shape[:2]

            xcent,ycent = xcents[i]-self.xmin,ycents[i]-self.ymin

            xmin,xmax,ymin,ymax = int(xcent-radius),int(xcent+radius),int(ycent-radius),int(ycent+radius)
            ## do edge detection:
            pads  = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])


            clip = image[ymin:ymax,xmin:xmax]

            # print(clip,'makedocip')
            if np.any(pads < 0):
                topad = pads<0
                padding = -1*pads*topad
                clip = np.pad(clip,padding,'edge')

            all_clipped[i,:,:,:] = (np.round(255*clip)).astype(np.uint8)
        return all_clipped

    ## Calculate image histograms over all frames
    def patch_grandhist(self,frames,part,radius):
        dataarray = self.make_patches(frames,part,radius)
        hists = [np.histogram(dataarray[:,:,:,i],bins = np.linspace(0,255,256)) for i in range(3)]
        return hists

    ## Calculate image histograms over each frame
    def patch_hist(self,frames,part,radius):
        dataarray = self.make_patches(frames,part,radius)
        hists = [[np.histogram(dataarray[f,:,:,i],bins = np.linspace(0,255,256)) for i in range(3)]for f in range(len(frames))]
        return hists

    ## More serious computer vision 
    def binarize_frames(self,indices):
        """Binarize the frames at the given set of indices. Use Yen's method for thresholding (selected from visual examination of test frames). Yen's method selected based on: 
        .. highlight:: python
        .. code-block:: python
            from skimage.threshold import try_all_threshold            
            fig, ax = try_all_threshold(img, verbose=False)
            plt.show()
        '''

        :param indices:
        :return: a list of binarized frames.
        """
        frames = self.get_images_grayscale(indices)
        ## If including the registration frame, its value is not that good for the segmetnation of mice because it includes too much other garbage. Use the mean of the other thresholds instead.
        diff = np.diff(indices)
        assert np.all(diff>0),"please give indices in increasing order."
        include_registration_frame = (0 in indices)
        binarized = []
        threshes = []
        for i,image in enumerate(frames[::-1]):
            if i == len(frames)-1 and include_registration_frame:
                thresh = np.mean(threshes)
            else:
                thresh = threshold_yen(image)- 0.05
                threshes.append(thresh)
            binarize = image<thresh
            binarized.append(binarize)
        return binarized[::-1]

    def clean_binary(self,binaries):
        """Given a set of binarized images, clean them up with morphological opening. 

        :param binaries: a list of binary images that represent the training data. Note that the animal should be assigned to ONE, not 0, meaning we should use morphological closing (dilation followed by erosion.)   
        :return: list of image frames. 
        """
        cleaned = []
        for image in binaries:
            opened = binary_opening(image)
            closed = binary_closing(opened)
            cleaned.append(closed)
        return cleaned

    def link_parts(self,poses):
        """Given two numpy arrays representing the (part,coordinate) locations of both animals, will return two corresponding arrays of the parts connecting them. 

        :param poses: the (part,coordinate) array representing the body parts of a given animal. We will link the three head points, the ears to centroid, and the centroid to tail. We will also include a crossbar across the centroid, parallel to the position of the earsto help distinguish the bodies of the two animals.
        """
        get_length = lambda p1,p2: int(np.linalg.norm(p1-p2))
        get_spacing = lambda p1,p2: np.linspace(p1,p2,get_length(p1,p2)).astype(int)
        tl = get_spacing(poses[0,:],poses[1,:])
        tr = get_spacing(poses[0,:],poses[2,:])
        lr = get_spacing(poses[1,:],poses[2,:])
        midpointlr = np.mean(lr,axis = 0).astype(int)
        midpointlrc = ((midpointlr+poses[3,:])/2).astype(int)
        tc = get_spacing(poses[0,:],poses[3,:])
        ci = get_spacing(poses[3,:],poses[4,:])
        cb = (1.5*(lr - midpointlr) + poses[3]).astype(int)
        cbm = (1.5*(lr - midpointlr) + midpointlrc).astype(int)
        return np.concatenate([tl,tr,lr,tc,ci,cb,cbm],axis = 0)


    def generate_auxpoints(self,index,to_edit,dryrun = False):
        """Generate auxiliary tracking points that we can unequivocally assign to each animal for difficult to segment frames. 

        :param index: the index in the training set of the data you would like to label. 
        :param dryrun: boolean; if True, skips the actual plotting for testing purposes.
        :param to_edit: list of strings, should be dam, virgin, or both
        """
        for entry in to_edit: 
            assert entry in ["dam","virgin"]
        colors = {"virgin":"blue","dam":"red"}
        pointarray = {e:[] for e in to_edit} 

        if dryrun:
            pointarray = {"dam":[[0,0],[1,1]],"virgin":[[4,4],[40,40]]}

        else:
            for entry in to_edit:
                fig,ax = plt.subplots()
                ax.imshow(self.get_images([index])[0])
                linem = LineSelector(fig)
                linem.connect()
                plt.title("Auxiliary points for {}".format(entry))
                plt.show()
                pointarray[entry].extend(linem.get_all_points())
                
            fig,ax = plt.subplots()
            ax.imshow(self.get_images([index])[0])
            for k,vals in pointarray.items():
                valarray = np.array(vals)
                ax.plot(valarray[:,0],valarray[:,1],"x",color = colors[k],label = k)
                plt.title("Given Marker Points")
                plt.legend()
            plt.show()
        return pointarray

    def save_auxpoints(self,improvement_dict,foldername):
        """Given a dictionary of indices and corresponding animal indentities to generate improvements for, saves out individual dictionaries corresonding to each frame in a provided subfolder. 

        :param improvement_dict: a dictionary with keys providing the indices at which we should generate refinements, and values providing a list of the animals we should provide refinements for (a list with entries of virgin, dam or both). 
        :param foldername: a string giving the subfolder we should save out the auxiliary points to. 
        """
        savepath = os.path.join(self.additionalpath,foldername)
        mkdir_notexists(savepath)

        for ind,l in improvement_dict.items():
            assert type(ind) is int,"indices must be integers"
            for li in l:
                assert li in ["virgin","dam"]
            print(ind,l)
            auxpoints = self.generate_auxpoints(ind,l)
            filename = "dict_index_{}".format(ind)
            with open(os.path.join(savepath,filename),"w") as f:
                json.dump(auxpoints,f,indent = 4)
    
    def get_auxpoints(self,folderdict):
        """Given a dictionary of folder paths and values the corresponding training frame indices, retrieve all of the auxiliary points.
        :param folderdict: dictionary with foldernames as keys and lists of indices as values.
        :return: a dictionary with indices as keys and auxiliary point dictionaries as values.  
        """
        returndict = {}
        for folder,points in folderdict.items():
            for point in points:
                with open(os.path.join(self.additionalpath,folder,"dict_index_{}".format(point)),"r") as f:
                    datadict = json.load(f)
                returndict[point] = datadict
        return returndict

    def segment_animals(self,binary,vposes,dposes,auxpoints = None):
        """If animals are touching, separate them with distance transform.

        :param binary: A numpy array representing a binarized image.
        :param vposes: A numpy array (coordinate,part) representing the virgin's position in this frame (standard part ordering)
        :param mposes: A numpy array (coordinate,part) representing the dam's position in this frame (standard part ordering)
        :return: An integer valued image where 0 represents the background, 1 represents the virgin and 2 represents the dam., as well as an array of the markers used to initialize the watershed algorithm.
        """
        assert vposes.shape[0] == dposes.shape[0] == 2, "part arrays must be given as (coordinate,part)"
        if auxpoints is not None:
            for k,val in auxpoints.items():
                valarray = np.array(val)
                assert k in ["dam","virgin"]
                print(valarray.shape)
                assert valarray.shape[-1] == 2 
        vpint = vposes.astype(int)
        vpinterp = self.link_parts(vpint.T)
        dpint = dposes.astype(int)
        dpinterp = self.link_parts(dpint.T)
        allinterp = {"virgin":vpinterp,"dam":dpinterp}
        if auxpoints is not None:
            for k,val in auxpoints.items():
                allinterp[k] = np.concatenate([allinterp[k],np.array(val).astype(int)],axis = 0)

        distance = ndi.distance_transform_edt(binary)
        markers = np.zeros_like(distance)
        for ki,k in enumerate(["virgin","dam"]):
            markers[allinterp[k][:,1],allinterp[k][:,0]] = ki +1 

        labels = watershed(-distance,markers,mask = binary)
        return labels,markers

    def smooth_segmentation(self,segmentation,sigma = 0.7):
        """ Calculates a gaussian blur about the detected segmentation and then thresholds in order to smooth out the edges pleasantly. 
        :param segment: binarized image indicating the value of the virgin mouse encoded at value 1, and the value of the dam encoded at value 2. 
        :returns: return a (imagedim1,imagedim2, 2) dimensional array, with the last index giving the position of either animal.
        """

        output = np.zeros((*segmentation.shape,2))

        for i in [1,2]:
            mouseimage = (segmentation == i)
            ## Sigma chosen arbitrarily
            smoothed = gaussian(mouseimage,sigma = sigma)
            image_th = threshold_yen(smoothed)
            thresholded = smoothed > image_th
            deholed = remove_small_holes(thresholded,area_threshold = 50)
            output[:,:,i-1] = median(mouseimage,np.ones((5,5)))
            #output[:,:,i-1] = deholed
        return output


    def get_contour_from_seg(self,segment):
        """Converts a detected segmentation image into a contour

        :param segment: np array of shape (imagedim1,imagedim2,2), representing a stack of binarized images indicating the locations of each mouse. 
        :return contours: a list of lists representing the contours for each mouse (should only be 2)
        """
        contours = [measure.find_contours(segment[:,:,i],0.5) for i in range(2)]
        return contours 
 
    def get_contour(self,indices,auxpoints_trained = {}):
        """ TODO write test for this! Get the pair of contours that aligns best with the detected animal points in each frame.

        :param indices: numpy array of the frame indices in the training set for which you want contours
        :return: a list of lists, organized in depth as frame, animal, contour per animal.
        """
        binary = self.binarize_frames(indices)
        cleanbinary = self.clean_binary(binary)
        all_contours = []
        for ii,i in enumerate(indices):
            binim = cleanbinary[ii]
            vdata = self.dataarray[i,:,:,0]
            ddata = self.dataarray[i,:,:,1]
            if auxpoints_trained.get(i,False):
                labels,markers = self.segment_animals(binim,vdata,ddata,auxpoints = auxpoints_trained[i])
            else:
                labels,markers = self.segment_animals(binim,vdata,ddata)
            smoothed = self.smooth_segmentation(labels)
            contours = self.get_contour_from_seg(smoothed)
            all_contours.append(contours)
            
        return all_contours

    def get_contour_fourier_rep(self,contourdict):
        """Get a fourier coefficient representation of animal contours 

        :param contourdict: a dictionary with keys giving training frame indices and values giving the corresponding contours. 
        :return: dictionary with keys giving training frame indices and values giving the fourier representation of corresponding contours.
        """
        dictentries = {}
        for c,centry in contourdict.items():
            mouseentries = {"virgin":{},"dam":{}}
            for mi,m in enumerate(mouseentries):
                ccomplex = centry[mi][0][:,0]+1j*centry[mi][0][:,1] #take the 0th entry. Fix this upstream later. 
                cfft = np.fft.fft(ccomplex)
                freqs = np.fft.fftfreq(len(ccomplex),30)## assuming all videos are at 30 fps.
                mouseentries[m] = {"coefs":cfft,"freqs":freqs}
            dictentries[c] = mouseentries
        return dictentries
    
    def find_startpoint(self,fourierdict,tips,cents):
        """Find the point on the contour to assign as t = 0. We would like this to be as close as possible to the mouse's nose tip for consistency. 

        :param fourierdict: a dictionary with entries for the virgin and dam, containing the fourier coefficients and corresponding frequencies for both. 
        :param tips: a numpy array of shape (coordinate, mouse) giving the nose tip points for each mouse
        :param cents: a numpy array of shape (coordinate, mouse) giving the centroid points for each mouse
        """
        rotdict = {}
        for keyind,key in enumerate(fourierdict.keys()):
            rotdict[key] = fourierdict[key]
            coefficients = fourierdict[key]["coefs"]
            n = len(coefficients)
            ## The starting point of the curve is given by the sum of all the fourier coefficients (t = 0 => e^{-tik} = 1)
            rot_coefs = lambda b: coefficients*np.exp(2*np.pi*1j*b*np.linspace(0,n-1,n)/n)
            diff_init = lambda b: np.sum(rot_coefs(b))/n
            ## Measure the distance between different potential 
            dists = lambda b: np.abs(diff_init(b) - (tips[0,keyind]+1j*tips[1,keyind]))
            #dists = lambda b: np.linalg.norm(np.array([np.real(diff_init(b)),np.imag(diff_init(b))]) - tips[:,keyind])
            distvals = map(dists,np.linspace(0,n-1,n))
            z = np.argmin(list(distvals))
            rotdict[key]["coefs"] = rot_coefs(z) 
            ## Checking framework
            #for bi in np.linspace(0,n-1,n):

            #    init = diff_init(bi)
            #    x = np.real(init)
            #    y = np.imag(init)
            #    if bi == z:
            #        plt.plot(y,x,"x")
            #    else:
            #        pass
            #plt.plot(*tips[:,keyind],"o")
            #plt.plot(*cents[:,keyind],"o")
            #ci = np.fft.ifft(coefficients)
            #xci = np.real(ci)
            #yci = np.imag(ci)
            #plt.plot(yci,xci,"+")
            #plt.show()

        return rotdict 

    def center_contour(self,fourierdict,cents):
        """Center the fourier contours at the corresponding mouse centroid point. This will involve getting the dc component of the fourier representation, subtracting off the markered centroid position, and converting back. 
        
        :param fourierdict: a dictionary with entries for the virgin and dam, containing the fourier coefficients and corresponding frequencies for both. 
        :param cents: a numpy array of shape (coordinate, mouse) giving the centroid points for each mouse
        """
        centdict = {}
        for keyind,key in enumerate(fourierdict.keys()):
            centdict[key] = fourierdict[key]
            coefficients = fourierdict[key]["coefs"]
            ## The DC component is always first:
            dc = coefficients[0]
            cent = cents[:,keyind]
            complexcent = (cent[1]+1j*cent[0])*len(coefficients)
            newdc = dc-complexcent
            centdict[key]["coefs"][0] = newdc
        return centdict

    def rotate_contour(self,fourierdict,tips,cents):
        """
        :param fourierdict: a dictionary with entries for the virgin and dam, containing the fourier coefficients and corresponding frequencies for both. 
        :param tips: a numpy array of shape (coordinate, mouse) giving the nose tip points for each mouse
        :param cents: a numpy array of shape (coordinate, mouse) giving the centroid points for each mouse
        """
        ## Get angle between tips and centroids for both animals (NOTE: THIS SHOULD BE Y-X order to match coordinates for images.)
        dirvec = tips-cents
        compdirvec = dirvec[1,:]+1j*dirvec[0,:]
        mouseangles = np.angle(compdirvec)
        aligndict = {}
        for keyind,key in enumerate(fourierdict.keys()):
            aligndict[key] = fourierdict[key]
            coefficients = fourierdict[key]["coefs"]
            rotfactor = np.exp(np.pi/2*1j-1j*mouseangles[keyind])
            rotated = coefficients*rotfactor
            aligndict[key]["coefs"] = rotated

        return aligndict

    def center_and_rotate_fourier_rep(self,fourierdict):
        """Given a set of fourier coefficent representations for animal contours, centers them to the centroid point, and rotates them so that the tip of the nose is facing straight up. References the detected training points to do this transformation.

        :param fourierdict: dictionary with keys giving training frame indices and values giving the fourier representation of corresponding contours.
        """
        dictentries = {}
        for f,fentry in fourierdict.items():
            ## Get training data: 
            points = self.dataarray[f,:,:,:]
            tips = points[:,0,:]
            cents = points[:,3,:]
            ## Get the x-y coordinates of the 
            frot  = self.find_startpoint(fentry,tips,cents)
            fcent = self.center_contour(frot,cents)
            falign = self.rotate_contour(fcent,tips,cents)
            dictentries[f] = falign
        return dictentries
            
    def organize_fourierdicts(self,fourierdicts):
        """Given a set of fourier coefficient representations as a dictionary of dictionaries, arranges it into a format that is easier to parse. 

        :param fourierdicts: A dictionary of dictionaries, with keys giving training frame indices and values giving the fourier representation of corresponding contours. Each value is a dictionary specifying the coefficients and frequencies of the corresponding representation.
        """
        indices = list(fourierdicts.keys())
        animals = list(fourierdicts[indices[0]].keys())
        identifier = fourierdicts[indices[0]][animals[0]].keys() 
        if "xcoefs" in identifier:
            fdtype = "elliptic"
        elif "coefs" in identifier:
            fdtype = "complex"
        else:
            raise Exception("dictionary not formatted correctly.")
        all_animals = []
        for animal in animals: 
            if fdtype is "complex":
                all_coefs = [fourierdicts[i][animal]["coefs"] for i in indices]
                animaldata = np.array(all_coefs) ## has shape (len(indices),nb_coefficients)
            elif fdtype is "elliptic":
                x_coefs = np.stack([fourierdicts[i][animal]["xcoefs"] for i in indices],axis =0)
                y_coefs = np.stack([fourierdicts[i][animal]["ycoefs"] for i in indices],axis =0)
                animaldata = np.stack([x_coefs,y_coefs],axis = 1) ## has shape (len(indices),coordinate,nb_coefficients)
            all_animals.append(animaldata)
        ## Assume all have the same frequencies.
        all_data = {"data":np.stack(all_animals,axis = -1),"fdtype":fdtype,"indices":indices,"animals":animals,"freqs":fourierdicts[indices[0]][animal]["freqs"]}
        return all_data

    def get_pca(self,organized_fourierdicts,n_components = None):
        """Applies PCA to the organized fourier dictionaries, and returns the fitted pca object. A wrapper for sklearn.decomposition.PCA, with appropriate data preprocessing steps. For elliptic fourier series, the fourier transform of each coordinate is conjugate symmetric, so we can just take the positive part of the series without losing information. For the complex fourier series, we must take the whole thing.  
        :param organized_fourierdicts: Assumes that one has applied the method `organize_fourierdicts` to a dictionary of fourier dictionaries.  
        :param n_components: (optional) The number of components to use when initializing pca. If none, automatically calculated by sklearn module. 
        """
        dampca = PCA(n_components=n_components)
        virgpca = PCA(n_components=n_components)
        ## Determine if we are working with complex or elliptic signals. 
        fdtype = organized_fourierdicts["fdtype"]
        if fdtype == "elliptic":
            data = organized_fourierdicts["data"]
            ## Conjugate symmetric, so we throw away all negative frequencies in the feature axis except the nyquist frequency. Given in standard order, these are the first 256 entries of the frequency vector in the feature axis.
            inds_nondegenerate = np.arange(data.shape[2]//2+1)
            orig_shape = data.shape
            data_nondegen = data[:,:,inds_nondegenerate,:]
            ## Now we can stack the real and imaginary parts in the feature axis. 
            data_realimag = np.concatenate([np.real(data_nondegen),np.imag(data_nondegen)], axis = 2)
            ## Now we stack x and y in the feature axis.
            data_xy = data_realimag.reshape(orig_shape[0],orig_shape[1]*(orig_shape[2]+2),orig_shape[3])

            virginweights = virgpca.fit_transform(data_xy[:,:,0])
            damweights = dampca.fit_transform(data_xy[:,:,1])
            pcadict = {"virgin":{"pca":virgpca,"weights":virginweights},"dam":{"pca":dampca,"weights":damweights},"indices":organized_fourierdicts["indices"]}
        else:
            raise NotImplementedError("Only implemented elliptic transform features so far.")
            
        return pcadict

    def get_contour_from_pc(self,pca,weights):
        """When given a fit pca object and vector of pc weights, returns a contour of mouse weights that corresponds to that set of weights as applied to the provided pca model. 
        :param pca: A pca model that has been fit to contour data. 
        :param weights: A set of vectors, with overall shape (n_samples,n_components) where n_components is the number of components expected by the pca model and m is the number of vectors we would like to fit.

        """
        assert pca.n_components == weights.shape[1]
        datavectors = pca.inverse_transform(weights)
        reshaped = datavectors.reshape(len(weights),2,-1)
        featurelength = int(reshaped.shape[-1]/2)
        reshaped_complex = reshaped[:,:,:featurelength]+1j*reshaped[:,:,featurelength:]
        ## We need to add back to complex conjugation. 
        xy = np.fft.irfft(reshaped_complex)
        return xy#np.stack([x,y],axis = -1)

    def get_gaussian_contour_pose_distribution(self,animal,pca_weights,all_fds):
        """ Given an array of pca weights and dictionary of fourier descriptors corresponding to the training frames they belong to, constructs a multivariate gaussian distribution in which the first 8 points give the relative marker points of the animal (relative to the centroid and rotated so that the nose tip is facing upwards), and the remaining components give the pca weights of the contour's fourier decomposition. Note that we will require the existence of a trained pca object in order to make use of this distribution. 

        :param animal: An indicator of the animal we want to solve the problem for. Will be virgin or dam. 
        :param pca_weights: A set of pc weights of shape (number_frames,number_features). 
        :param all_fds: A dictionary of fourierdescriptor objects, with frame indices as keys and FourierDescriptor objects as values.
        :return: The mean and variance of a gaussian distribution corresponding to the given data. 
        """
        dataindices = {"dam":1,"virgin":0}
        assert animal in dataindices.keys()
        ## First form the data: 
        remove_centroid = np.array([0,1,2,4])
        ## Get the releant animal, remove the centroid (will be zero, and flatten the xy coordinates.)
        all_points_list = [all_fds[ind].process_points() for ind in all_fds.keys()]

        relevant_poses = np.stack([all_fds[ind].process_points()[:,remove_centroid,dataindices[animal]].flatten(order = "F") for ind in all_fds.keys()],axis = 0)
        for i in range(len(relevant_poses)):
            relevant_pose = relevant_poses[i,:]
            pose = relevant_pose.reshape(2,4,order = "F")

        concat_data = np.concatenate([relevant_poses,pca_weights],axis =1)
        mean = np.mean(concat_data,axis = 0)
        cov = np.cov(concat_data,rowvar = False)
        return mean,cov
    
    def get_MAP_markers(self,mean,cov,contour_pc_weights):
        """When given the parameters of a gaussian + a contour, will return the MAP estimate for the animal marker locations based on that contour.  
        :param mean: the mean vector for the multivariate gaussian we are trying to fit.  
        :param cov: the covariance matrix for the multivariate gaussian we are trying to fit. 
        :param contour: a set of pc weights of shape (n_samples,n_features) corresponding to the contours that we are conditioning on.
        :return: an array of shape (n_samples,xy,partid) providing the positions of the markers given the contour.  
        """
        mean_markers = mean[:8]
        mean_contour = mean[8:]
        cov_mm = cov[:8,:8]
        cov_mc = cov[:8,8:]
        cov_cm = cov[8:,:8]
        cov_cc = cov[8:,8:]

        assert len(contour_pc_weights.shape) == 2
        assert contour_pc_weights.shape[1] == len(mean_contour) 
        prefactor = contour_pc_weights-mean_contour[None,:]
        ## We have to transpose to make the matrix multiplcation work right.
        factor_d = np.matmul(cov_mc,np.linalg.solve(cov_cc,prefactor.T)).T

        markers_flat = mean_markers[None,:]+factor_d

        markers = markers_flat.reshape(len(contour_pc_weights),2,4,order = "F")

        return markers 

    def get_MAP_weights(self,mean,cov,markers):
        """When given the parameters of a gaussian + a contour, will return the MAP estimate for the animal marker locations based on that contour.  
        :param mean: the mean vector for the multivariate gaussian we are trying to fit.  
        :param cov: the covariance matrix for the multivariate gaussian we are trying to fit. 
        :param markers: the pc weights in an array of shape (n_samples,xy,partid) corresponding to the contour that we are conditioning on.
        :return: an array in of shape (n_samples,n_components) providing the positions of the markers given the contour.  
        """
        assert markers.shape[1:] == (2,5) 
        noncent = np.array([0,1,2,4])
        markers_nocent = markers[:,:,noncent]

        mean_markers = mean[:8]
        mean_contour = mean[8:]
        cov_mm = cov[:8,:8]
        cov_mc = cov[:8,8:]
        cov_cm = cov[8:,:8]
        cov_cc = cov[8:,8:]

        ## Flatten markers:
        markers_flat = markers_nocent.reshape(-1,8,order = "F")

        prefactor = markers_flat-mean_markers[None,:]
        factor = np.matmul(cov_cm,np.linalg.solve(cov_mm,prefactor.T)).T
        return mean_contour[None,:]+factor

class FourierDescriptor():
    """A data class to handle the creation and refinement of fourier descriptors for individual images.  
    Assumes that one has run the `LabeledData.get_contours` method to generate the relevant contours here. Note we are assuming all operations here are done on a pre-specified set of training frames. 

    :param image:
    :param contours: a dictionary with two entries: one for the dam and one for the virgin, with corresponding image contours.
    :param points:
    """
    def __init__(self,image,contours,points):
        """Filters out misspecified input data.
        """
        self.image = image
        self.contours = contours
        self.points = points
        assert len(self.image.shape) ==3, "must be a single rgb image"
        assert len(self.contours) == 2,"must provide a list of contours for virgin and dam."
        assert list(self.contours.keys()) == ["virgin","dam"], "must be named appropriately"
        for c,centry in self.contours.items():
            #assert len(centry) == 1,"assume one contour per animal: {} contours given".format(len(centry))
            assert centry.shape[-1] == 2
        assert self.points.shape == (2,5,2)
        tips = self.points[:,0,:]
        cents = self.points[:,3,:]
        dirvec = tips-cents
        compdirvec = dirvec[1,:]+1j*dirvec[0,:]
        self.mouseangles = np.angle(compdirvec)
        
    def normalized_interpolation(self,n = 512):
        """Returns a normalized contour linearly interpolated to a pre-specified number of points. Choose a power of two for fast computations via FFT. Additionally, will change the orientation of contours so that all rotate counterclockwise. 
        :param n: the number of points in your interpolation
        :return: returns a dictionary with keys as animal names, and values as arrays of interpolated contour points.
        """
        animals = ["virgin","dam"]
        contours = {"virgin":None,"dam":None}
        for a in animals:
            contour = self.contours[a]
            integrated_angle = calculate_orientation(contour)
            if integrated_angle < 0:
                contour = np.flipud(contour)
            x = np.arange(len(contour))
            ifunc = interp1d(x,contour,axis = 0)
            xinterp = np.linspace(0,x[-1],n)
            contours[a] = ifunc(xinterp)
        return contours

    def get_complex_fourier_features(self,signal=None):
        """Get fourier features by projecting the y coordinates to the complex plane and applying a dft. This will generate a two-sided spectrum that is asymmetric. Note that we assume a sampling frequency of 1 for uniformity (in reality 30 fps). TODO: determine the relationship between these different parametrizations.  
        :param signal: (optional) If given, a dictionary containing a two dimensional (x,y) time series that we will apply an fft to for each animal. If not given, apply to the "contour" attribute
        :return: a dictionary giving the coefficients and frequencies at which to register these frequencies. 
        """
        if signal is None:
            signal = self.contours
        assert list(signal.keys()) == ["virgin","dam"]
        transformdict = {}
        for key,ksig in signal.items():
            csignal = ksig[:,0]+1j*ksig[:,1]
            fft_coefs = np.fft.fft(csignal)
            fft_freqs = np.fft.fftfreq(len(csignal))
            transformdict[key] = {"coefs":fft_coefs,"freqs":fft_freqs}
        return transformdict 


    def get_elliptic_fourier_features(self,signal=None):
        """Get fourier features by treating x and y as two independent 1-d signals with independent transforms. This will generate two separate symmetric spectrums. TODO: determine the relationship between these different parametrizations. 
        :param signal: (optional) If given, a dictionary containing a two dimensional (x,y) time series that we will apply an fft to for each animal. If not given, apply to the "contour" attribute
        :return: a dictionary giving the coefficients and frequencies at which to register these frequencies (separate of x and y) 
        """
        if signal is None:
            signal = self.contours
        assert list(signal.keys()) == ["virgin","dam"]
        transformdict = {}
        for key,ksig in signal.items():
            xsignal = ksig[:,0]
            ysignal = ksig[:,1]
            fft_xcoefs = np.fft.fft(xsignal)
            fft_ycoefs = np.fft.fft(ysignal)
            fft_freqs = np.fft.fftfreq(len(xsignal))
            transformdict[key] = {"xcoefs":fft_xcoefs,"ycoefs":fft_ycoefs,"freqs":fft_freqs}
        return transformdict 

    def convert_to_contour(self,fourierdict):
        """Given a dictionary of fourier descriptors per mouse, converts to contours. Respects the reversed x y convention of the contours. Will have to be reversed if comparing to points. 

        """
        mousedict = {}
        for keyind,key in enumerate(fourierdict.keys()):
            if fourierdict[key].get("coefs",None) is not None:
                ifft = np.fft.ifft(fourierdict[key]["coefs"])
                coords = np.stack([np.real(ifft),np.imag(ifft)],axis = 1)
            else:
                xifft = np.real(np.fft.ifft(fourierdict[key]["xcoefs"]))
                yifft = np.real(np.fft.ifft(fourierdict[key]["ycoefs"]))
                coords = np.stack([xifft,yifft],axis = 1)
            mousedict[key] = coords 
        return mousedict 
                

    def find_startpoint(self,fourierdict,points = "processed"):
        """Find the point on the contour to assign as t = 0. We would like this to be as close as possible to the mouse's nose tip for consistency. 
        :param fourierdict: If given, a dictionary containing a two dimensional (x,y) time series that we will apply an fft to for each animal. If not given, apply to the "contour" attribute
        :param points: (optional) Can be "processed","raw", or a numpy array. Processed means assume that the given fourier dictionary corresponds to a trajectory that has been centered and rotated so the vector from centroid to tip points upwards. Raw means the points that the object was initialized with. Otherwise can be 
        :return: dictionary of the same shape as fourierdict, giving the relevant frequency coefficients. 
        """
        if points == "processed":
            points = self.process_points()
        elif points == "raw":
            points = self.points
        elif type(points) == np.ndarray:
            assert points.shape == (2,5,2),"must give in shape (coordinates,part,animal)"
            points = points
        else:
            raise Exception("points argument not recognized.")

        rotdict = {}
        for keyind,key in enumerate(fourierdict.keys()):
            rotdict[key] = {}
            rotdict[key]["freqs"] = fourierdict[key]["freqs"]
            tips = points[:,0,keyind]
            ## Depending on if these are complex or elliptic fourier features, we need different distance functions.  
            ## if complex:
            if fourierdict[key].get("coefs",None) is not None:
                ## X and Y are flipped for contours.
                dists = lambda b: np.abs(rotate_coefficients(fourierdict[key]["coefs"],b)[1] - (tips[1]+1j*tips[0]))
                n = len(fourierdict[key]["coefs"])
            else:
                xfreqstart = lambda b: rotate_coefficients(fourierdict[key]["xcoefs"],b)[1]
                yfreqstart = lambda b: rotate_coefficients(fourierdict[key]["ycoefs"],b)[1]
                n = len(fourierdict[key]["xcoefs"])
                ## X and Y are flipped for contours.
                dists = lambda b: np.linalg.norm(np.array([yfreqstart(b),xfreqstart(b)])-tips)
            distvals = map(dists,np.linspace(0,n-1,n))
            z = np.argmin(list(distvals))
            rotdict[key]["z"] = z
            if fourierdict[key].get("coefs",None) is not None:
                rotdict[key]["coefs"] = rotate_coefficients(fourierdict[key]["coefs"],z)[0]
            else:
                rotdict[key]["xcoefs"] = rotate_coefficients(fourierdict[key]["xcoefs"],z)[0]
                rotdict[key]["ycoefs"] = rotate_coefficients(fourierdict[key]["ycoefs"],z)[0]
            
        return rotdict

    def center_contour(self,fourierdict):
        """Center the fourier contours at the corresponding mouse centroid point. This will involve getting the dc component of the fourier representation, subtracting off the markered centroid position, and converting back.

        :param fourierdict: a dictionary with entries for the virgin and dam, containing the fourier coefficients and corresponding frequencies for both. can be given as complex or elliptic fourier features.
        """
        centdict = {}
        for keyind,key in enumerate(fourierdict.keys()):
            cent_marker = self.points[:,3,keyind]
            centdict[key] = fourierdict[key] 
            if fourierdict[key].get("coefs",None) is not None:
                condition = "complex"
                dc = fourierdict[key]["coefs"][0]
                newdc = dc - (cent_marker[1]+1j*cent_marker[0])*len(fourierdict[key]["coefs"])
                centdict[key]["coefs"][0] = newdc

            else:
                condition = "elliptic"
                xdc = fourierdict[key]["xcoefs"][0]
                ydc = fourierdict[key]["ycoefs"][0]
                newxdc = xdc - cent_marker[1]*len(fourierdict[key]["xcoefs"])
                newydc = ydc - cent_marker[0]*len(fourierdict[key]["xcoefs"])
                centdict[key]["xcoefs"][0] = newxdc
                centdict[key]["ycoefs"][0] = newydc
        return centdict

    def rotate_contour(self,fourierdict):
        """Rotates the fourier coefficient to align the mouse contour to the line between the centroid and the tip of the nose. 
        :param fourierdict: a dictionary with entries for the virgin and dam, containing the fourier coefficients and corresponding frequencies for both. can be given as complex or elliptic fourier features.
        """
        ## Get the angle between tips and centroids for both animals:    
        mouseangles = self.mouseangles
        aligndict = {}
        for keyind,key in enumerate(fourierdict.keys()):
            aligndict[key] = fourierdict[key]
            ## Center the angle to the vertical.
            baseangle = np.pi/2-mouseangles[keyind]
            rotfactor = np.exp(1j*baseangle)
            if fourierdict[key].get("coefs",None) is not None:
                coefficients = fourierdict[key]["coefs"]
                rotcoefs = coefficients*rotfactor
                aligndict[key]["coefs"] = rotcoefs
            else: 
                angle = -baseangle
                ## We need a bona fide rotation matrix here. 
                xcoefficients = fourierdict[key]["xcoefs"]
                ycoefficients = fourierdict[key]["ycoefs"]
                ## Combine into a matrix: 
                a = np.real(xcoefficients)[:,None,None]
                b = np.imag(xcoefficients)[:,None,None]
                c = np.real(ycoefficients)[:,None,None]
                d = np.imag(ycoefficients)[:,None,None]
                blockmatrix = np.block([[a,b],[c,d]])
                rotmatrix = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
                rotated = np.matmul(rotmatrix,blockmatrix)
                xrotcoefs = rotated[:,0,0]+1j*rotated[:,0,1] 
                yrotcoefs = rotated[:,1,0]+1j*rotated[:,1,1] 
                aligndict[key]["xcoefs"] = xrotcoefs
                aligndict[key]["ycoefs"] = yrotcoefs
        return aligndict

    def center_points(self,points=None):
        """Get the centered representation of body part markers to correspond with centered contours. 

        """
        if points is None:
            points = self.points
        return points-points[:,3:4,:]

    def rotate_points(self,points = None):
        """Get the rotated representation of points: 

        """
        angles = -self.mouseangles
        rotated = []
        for i in range(2):
            angle = angles[i]+np.pi/2
            rotmatrix = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
            rot = np.matmul(rotmatrix,(points[:,:,i]))
            rotated.append(rot)
        return np.stack(rotated,axis = -1)

    def process_points(self):
        """Pipeline to process points in conjunction with fourier descriptors. 
        """
        centered = self.center_points(self.points)
        rotated = self.rotate_points(centered)
        return rotated

    def process_contour(self,mode = "elliptic"):
        """Full pipeline to take the contour given at object initialization, and apply the following methods to it: 
        `normalized_interpolation`: Correct the orientation of the contour, and sample to a pre-determined number of points. 
        `get___fourier_features`: gets either complex or elliptic fourier features and applies further processing steps to them. 
        `center_contour`: Center the contour at the origin.
        `rotate_contour`: Rotate the contour to have the tip pointing in the y direction as much as possible
        `find_startpoint`: Start the contour at the point closest to the tip of the nose. 

        :return: a fourier descriptor for the contour associated with this object.

        """
        interpolated = self.normalized_interpolation(512)
        if mode is "elliptic":
            features = self.get_elliptic_fourier_features(interpolated)
        elif mode is "complex":
            features = self.get_complex_fourier_features(interpolated)
        else:
            raise Exception("mode must be one of elliptic or complex.")
        
        centered = self.center_contour(features)

        rotated = self.rotate_contour(centered)

        initialized = self.find_startpoint(rotated) 

        return initialized

    def superimpose_centered_contour(self,animal,centered_contour):
        """Given a centered, oriented contour, superimposes it on the original location and rotation of the object's contour. 

        """
        ## Get the contours: 
        assert centered_contour.shape[0] == 2
        ## Get animal identities:
        identitydict = {"virgin":0,"dam":1}
        ## Rotate to original position
        angle = -(self.mouseangles[identitydict[animal]]-np.pi/2) ## rotate towards the vertical axis.
        rotmatrix = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        rotated = np.matmul(rotmatrix,centered_contour)
        ## Recenter:
        recentered = rotated+self.points[::-1,3,identitydict[animal]][:,None]
        return recentered

    def evaluate_processed_coefs(self,animal,processed_contour,image = True):
        """Given a set of contour reconstructions generated by PCA, evaluates them against the corresponding frame of data. This evaluation takes three steps: 1) recenter and rotate the contour to the original position in the frame of video. 2) Calculate distance to the original contour. 3) get the image in a bounding box around the processed coefficient. 
        
        :param animal: string identifying the animal that we will be using 
        :param processed_contour: A contour of shape (2,number_features) that we will be aligning. 
        """
        recentered = self.superimpose_centered_contour(animal,processed_contour)
        ## Now get the interpolation of the original contour for comparison:
        interp = self.normalized_interpolation(recentered.shape[1])[animal].T
        start = recentered[:,0]
        z = interp-start[:,None]
        startloc = np.argmin(np.linalg.norm(z,axis = 0))

        rotated = np.concatenate([interp[:,startloc:],interp[:,:startloc]],axis = 1)
        ## Now get the root mean squared distance pointwise: 
        procrust = np.linalg.norm(recentered-rotated)
        ## Note that this is a lower bound on the procrustes distance, as the rotation and scaling is not guaranteed to be optimal (as it comes from keypoint alignment.)
        if image == True:
            ## Finally, get the image underneath the contour:
            integer_contour = np.round(recentered).astype(int)
            ## Zero mask:
            zeros_mask = np.zeros_like(self.image[:,:,0])
            print(zeros_mask.shape)
            zeros_mask[integer_contour[0,:],integer_contour[1,:]] = 1
            zeros_mask = ndi.binary_fill_holes(zeros_mask)
            only_mouse = np.zeros_like(self.image)+1
            only_mouse[zeros_mask,:] = self.image[zeros_mask,:]

            return procrust,only_mouse
        else:
            return procrust

class PoseDistribution():
    """Takes in a collection of fourier descriptors, then applies pca to them, and builds a corresponding gaussian distribution for them. Given an index, can return the relevant image, contour (real space), points, centered and normalized contour, fft (real or complex format), or pca weights. Can also return evaluation function that give the likelihood of any set of points given a complementary other set of points. 

    """
    def __init__(self,all_fds,n_components = 10):
        """Given a dictionary where keys are frame indices and values are fourierdescriptor objects (as well as the number of components for PCA), applies pca on the contours, and then creates a joint pose-contour gaussian distribution. Allows for arbitrary conditioning on some subset of points as well.  
        Much of the code here will be inherited from the later part of the LabeledData object workflow: 
        {FourierDescriptor.process_contour()}
        organize_fourierdicts()
        pcadict = get_pca
        distrib_params = mean,cov = get_gaussian_contour_pose_distribution()

        """
        assert type(all_fds) == dict,"must give dictionary, not {}".format(type(all_fds))
        for ind,item in all_fds.items():
            assert type(ind) == int 
            assert "image" in item.__dict__.keys()
            assert "contours" in item.__dict__.keys()
            assert "points" in item.__dict__.keys()

        ## First get out the fft representations: 
        all_ffts = {}
        for ind,item in all_fds.items():
            all_ffts[ind] = item.process_contour()
        
        ## Organize them in to an array for easy manipulation:
        organized_ffts = self.organize_fourierdicts(all_ffts)

        ## Get pca representation:
        pcadict = self.get_pca(organized_ffts,n_components)

        ## Now get distributional info for both animals: 
        animals = {"virgin":0,"dam":1}
        distrib_params = {}

        for animal in animals:
            mean,cov = self.get_gaussian_contour_pose_distribution(animal,pcadict[animal]["weights"],all_fds)
            distrib_params[animal] = {"mean":mean,"cov":cov}

        ## Finally assign all properties to self:
        self.all_fds = all_fds
        self.all_ffts = all_ffts
        self.ffts_array = organized_ffts
        self.pcadict = pcadict
        self.animals = animals
        self.distrib_params = distrib_params
            
    def organize_fourierdicts(self,fourierdicts):
        """Given a set of fourier coefficient representations as a dictionary of dictionaries, arranges it into a format that is easier to parse. 

        :param fourierdicts: A dictionary of dictionaries, with keys giving training frame indices and values giving the fourier representation of corresponding contours. Each value is a dictionary specifying the coefficients and frequencies of the corresponding representation.
        """
        indices = list(fourierdicts.keys())
        animals = list(fourierdicts[indices[0]].keys())
        identifier = fourierdicts[indices[0]][animals[0]].keys() 
        if "xcoefs" in identifier:
            fdtype = "elliptic"
        elif "coefs" in identifier:
            fdtype = "complex"
        else:
            raise Exception("dictionary not formatted correctly.")
        all_animals = []
        for animal in animals: 
            if fdtype is "complex":
                all_coefs = [fourierdicts[i][animal]["coefs"] for i in indices]
                animaldata = np.array(all_coefs) ## has shape (len(indices),nb_coefficients)
            elif fdtype is "elliptic":
                x_coefs = np.stack([fourierdicts[i][animal]["xcoefs"] for i in indices],axis =0)
                y_coefs = np.stack([fourierdicts[i][animal]["ycoefs"] for i in indices],axis =0)
                animaldata = np.stack([x_coefs,y_coefs],axis = 1) ## has shape (len(indices),coordinate,nb_coefficients)
            all_animals.append(animaldata)
        ## Assume all have the same frequencies.
        all_data = {"data":np.stack(all_animals,axis = -1),"fdtype":fdtype,"indices":indices,"animals":animals,"freqs":fourierdicts[indices[0]][animal]["freqs"]}
        return all_data
        
    def get_pca(self,organized_fourierdicts,n_components = None):
        """Applies PCA to the organized fourier dictionaries, and returns the fitted pca object. A wrapper for sklearn.decomposition.PCA, with appropriate data preprocessing steps. For elliptic fourier series, the fourier transform of each coordinate is conjugate symmetric, so we can just take the positive part of the series without losing information. For the complex fourier series, we must take the whole thing.  
        :param organized_fourierdicts: Assumes that one has applied the method `organize_fourierdicts` to a dictionary of fourier dictionaries.  
        :param n_components: (optional) The number of components to use when initializing pca. If none, automatically calculated by sklearn module. 
        """
        dampca = PCA(n_components=n_components)
        virgpca = PCA(n_components=n_components)
        ## Determine if we are working with complex or elliptic signals. 
        fdtype = organized_fourierdicts["fdtype"]
        if fdtype == "elliptic":
            data = organized_fourierdicts["data"]
            ## Conjugate symmetric, so we throw away all negative frequencies in the feature axis except the nyquist frequency. Given in standard order, these are the first 256 entries of the frequency vector in the feature axis.
            inds_nondegenerate = np.arange(data.shape[2]//2+1)
            orig_shape = data.shape
            data_nondegen = data[:,:,inds_nondegenerate,:]
            ## Now we can stack the real and imaginary parts in the feature axis. 
            data_realimag = np.concatenate([np.real(data_nondegen),np.imag(data_nondegen)], axis = 2)
            ## Now we stack x and y in the feature axis.
            data_xy = data_realimag.reshape(orig_shape[0],orig_shape[1]*(orig_shape[2]+2),orig_shape[3])

            virginweights = virgpca.fit_transform(data_xy[:,:,0])
            damweights = dampca.fit_transform(data_xy[:,:,1])
            pcadict = {"virgin":{"pca":virgpca,"weights":virginweights},"dam":{"pca":dampca,"weights":damweights},"indices":organized_fourierdicts["indices"]}
        else:
            raise NotImplementedError("Only implemented elliptic transform features so far.")
            
        return pcadict

    def get_gaussian_contour_pose_distribution(self,animal,pca_weights,all_fds):
        """ Given an array of pca weights and dictionary of fourier descriptors corresponding to the training frames they belong to, constructs a multivariate gaussian distribution in which the first 8 points give the relative marker points of the animal (relative to the centroid and rotated so that the nose tip is facing upwards), and the remaining components give the pca weights of the contour's fourier decomposition. Note that we will require the existence of a trained pca object in order to make use of this distribution. 
        :param animal: An indicator of the animal we want to solve the problem for. Will be virgin or dam. 
        :param pca_weights: A set of pc weights of shape (number_frames,number_features). 
        :param all_fds: A dictionary of fourierdescriptor objects, with frame indices as keys and FourierDescriptor objects as values.
        :return: The mean and variance of a gaussian distribution corresponding to the given data. 
        """
        dataindices = {"dam":1,"virgin":0}
        assert animal in dataindices.keys()
        ## First form the data: 
        remove_centroid = np.array([0,1,2,4])
        ## Get the releant animal, remove the centroid (will be zero, and flatten the xy coordinates.)
        all_points_list = [all_fds[ind].process_points() for ind in all_fds.keys()]

        relevant_poses = np.stack([all_fds[ind].process_points()[:,remove_centroid,dataindices[animal]].flatten(order = "F") for ind in all_fds.keys()],axis = 0)
        for i in range(len(relevant_poses)):
            relevant_pose = relevant_poses[i,:]
            pose = relevant_pose.reshape(2,4,order = "F")

        concat_data = np.concatenate([relevant_poses,pca_weights],axis =1)
        mean = np.mean(concat_data,axis = 0)
        cov = np.cov(concat_data,rowvar = False)
        return mean,cov

    def get_MAP_features(self,vector,animal):
        """When given a vector of the right shape with some true entries and some nans, will return the same vector with those NaN entries filled in by the MAP estimate given the properties of the appropriate distribution. 

        :param vectors: a set of vectors of shape (nb_vectors,feature_length) that we will be deriving map estimates for. Must be consistent in what entries contain NaNs.  
        :param animal: a string specifying what animal to regress against. 
        """
        assert len(vector.shape) == 2
        assert vector.shape[1] == len(self.distrib_params[animal]["mean"])
        ## Assert consistency of Nans 
        nans_prime = np.where(np.isnan(vector[0,:]))[0]
        for n in nans_prime:
            print(vector[:,n])
            assert np.all(np.isnan(vector[:,n]))

        length = len(self.distrib_params[animal]["mean"])
        indexvec = np.arange(length)

        ## Where is this vector nan, collect terms and move them (order maintained) to the end of the sequence
        nanvec = np.isnan(vector[0,:])
        nb_nans = np.sum(nanvec)
        nb_given = length-nb_nans
        addvec = nanvec*length
        ## Permute gives the ordering under which to perform conditioning, and reference gives the ordering to bring you back.
        permute = np.argsort(indexvec+addvec).squeeze()
        reference = np.argsort(indexvec[permute]).squeeze()

        ## Now order your parameters: 
        mean = self.distrib_params[animal]["mean"]
        cov = self.distrib_params[animal]["cov"]

        mean_sorted = mean[permute] 
        cov_sorted = cov[permute][:,permute]
        vector_sorted = vector[:,permute]

        mean_given = mean_sorted[:nb_given]
        mean_nan = mean_sorted[nb_given:]
        cov_given = cov_sorted[:nb_given,:nb_given]
        cov_nan = cov_sorted[nb_given:,nb_given:]
        cov_gn = cov_sorted[:nb_given,nb_given:]
        cov_ng = cov_gn.T
        vector_given = vector_sorted[:,:nb_given]

        ## Do conditioning:
        prefactor = vector_given - mean_given[None,:]
        factor = np.matmul(cov_ng,np.linalg.solve(cov_given,prefactor.T)).T
        print(prefactor,"prefactor")
        vector_map_est = mean_nan[None,:]+factor

        vector_concatenated = np.concatenate([vector_given,vector_map_est],axis = 1)

        ## Reorder: 
        reordered = vector_concatenated[:,reference]


        return reordered


