'''
A dataset class to hold training data, specifically. Useful to calculate
statistics on the training data to consider as ground truth going forwards.
'''

import numpy as np
import scipy
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os,sys
import tqdm


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
    
    def switch_points(dataset,indices,p):
        """Given a sampled dataset of shape (n,coordinate,part,mouse), switches p randomly selected body parts at each index with the corresponding body part on the other animal.

        """
        pass

    def remove_animal(dataset,indices):
        """Given a sampled dataset of shape (n,coordinate,part,mouse), removes the detetions for one of the animals, and substitutes in a jittered version of the other animal. 

        """
        pass

## Done with sampling. Now consider image statistic functions:
## We assume that if all of the data folders are not subfolders in the current
## directory, that they are at least packaged together.

    def get_images(self,indices):
        imagenames = self.get_imagenames(indices)
        ## Check if the images are somewhere else:
        if self.additionalpath is None:
            pass
        else:
            imagenames = [self.additionalpath+img for img in imagenames]

        ## Now we will load the images:
        images = [plt.imread(image) for image in imagenames]

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
