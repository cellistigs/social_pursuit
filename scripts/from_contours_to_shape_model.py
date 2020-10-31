### Make a statistical shape model for your training data. 
import os 
import numpy as np
from social_pursuit.data import Polar,PursuitVideo,mkdir_notexists
from social_pursuit.labeled import LabeledData,FourierDescriptor
from scipy.io import loadmat
from script_doc_utils import initialize_doc,insert_image,save_and_insert_image,get_relative_image_path,insert_vectors_as_table
from joblib import Memory
import joblib
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
})
from script_doc_utils import initialize_doc,insert_image,get_relative_image_path
cachedir = "/Volumes/TOSHIBA EXT STO/cache"
memory = Memory(cachedir, verbose=1)

@memory.cache
def process_contours(fds):
    fourierdescriptors = {}
    for i,fd in fds.items():
        processed = fd.process_contour()
        fourierdescriptors[i] = processed

    joblib.dump(fourierdescriptors,"/Volumes/TOSHIBA EXT STO/RTTemp_Traces/test_dir/fdsampleset")
    return fourierdescriptors
    
@memory.cache
def plot_basic_statistics(all_fds,fourierdescriptors):
    """Takes in two arguments: the set of all fourier descriptor objects (all_fds), and the correspondingly generated data (fourierdescriptors) for the contours we care about. Both are given as dictionaries. From these, we first reorganize the data into an easier to work with format, then plot the mean contour for both animals. we finally return the figure generated for plotting and the reorganized data, giving all of fourier descriptors as a big numpy array inside an appropriately shaped dictionary.  

    """
    fig,ax = plt.subplots(2,1)
    for i,fd in all_fds.items():
        contour = fd.convert_to_contour(fourierdescriptors[i])
        points = fd.process_points()
        for ji,j in enumerate(contour.keys()):
            ax[ji].plot(contour[j][:,0],contour[j][:,1],alpha = 0.2,color = "red")
            if ji == 0:
                ax[ji].plot(contour[j][0,0],contour[j][0,1],marker = "x",markersize=  5,color = "black",label = "contour starting points")
            else:
                ax[ji].plot(contour[j][0,0],contour[j][0,1],marker = "x",markersize=  5,color = "black")

    
    reorganized = data.organize_fourierdicts(fourierdescriptors)

    mean_fd = np.mean(reorganized["data"],axis = 0)
    std_fd = np.std(reorganized["data"],axis = 0)
    lower = mean_fd-std_fd
    upper = mean_fd+std_fd
    animal_dict = {"virgin":0,"dam":1}
    mean_contourdict = {}
    lower_contourdict = {}
    upper_contourdict = {}
    for an,ai in animal_dict.items():
        mean_contourdict[an] = {"xcoefs":mean_fd[0,:,ai],"ycoefs":mean_fd[1,:,ai]}
        lower_contourdict[an] = {"xcoefs":lower[0,:,ai],"ycoefs":lower[1,:,ai]}
        upper_contourdict[an] = {"xcoefs":upper[0,:,ai],"ycoefs":upper[1,:,ai]}
    mean_contours = fd.convert_to_contour(mean_contourdict)
    lower_contours = fd.convert_to_contour(lower_contourdict)
    upper_contours = fd.convert_to_contour(upper_contourdict)
    for an,ai in animal_dict.items():
        ax[ai].plot(mean_contours[an][:,0],mean_contours[an][:,1],color = "black",label = "mean")
        #ax[ai].plot(lower_contours[an][:,0],lower_contours[an][:,1],color = "black")
        #ax[ai].plot(upper_contours[an][:,0],upper_contours[an][:,1],color = "black")
        ax[ai].axis("equal")
        ax[ai].set_title("Mean contour for {}".format(an))
        ax[ai].axis("off")
    plt.legend()
    return fig,reorganized

@memory.cache
def get_pca(n_componenets,reorganized_data):
    pcadict = data.get_pca(reorganized,n_components)
    return pcadict


@memory.cache
def plot_perturbations(data,dampca,n_components):
    indexvec = np.arange(n_components)
    onehots = [indexvec == i for i in range(4)]
    perturbation_vecs = [2*oh*i*edges for oh in onehots for i in [-1,1]]
    perturbation_vecs.append(np.zeros(n_components,))
    perturbation_vecs = np.array(perturbation_vecs)

    assert perturbation_vecs.shape == (9,n_components)
    damvecs = data.get_contour_from_pc(dampca,perturbation_vecs)
    fig,ax = plt.subplots(3,3,sharey = True)
    ax[1,0].plot(*damvecs[0,:,:])
    ax[1,0].set_title("pc 1, -2")
    ax[1,2].plot(*damvecs[1,:,:])
    ax[1,2].set_title("pc 1, +2")
    ax[0,1].plot(*damvecs[2,:,:])
    ax[0,1].set_title("pc 2, -2")
    ax[2,1].plot(*damvecs[3,:,:])
    ax[2,1].set_title("pc 2, +2")
    ax[0,0].plot(*damvecs[4,:,:])
    ax[0,0].set_title("pc 3, -2")
    ax[2,2].plot(*damvecs[5,:,:])
    ax[2,2].set_title("pc 3, +2")
    ax[2,0].plot(*damvecs[6,:,:])
    ax[2,0].set_title("pc 4, -2")
    ax[0,2].plot(*damvecs[7,:,:])
    ax[0,2].set_title("pc 4, +2")
    ax[1,1].plot(*damvecs[8,:,:])
    ax[1,1].set_title("mean")
    [axy.axis("equal") for a in ax for axy in a]
    plt.tight_layout()
    return fig

@memory.cache
def scatterplot_virg_dam_pca(dampca_items,virgpca_items):
    fig,ax = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            ax[i,j].scatter(dampca_items["weights"][:,i],dampca_items["weights"][:,j],s = 0.5,label = "Dam")
            ax[i,j].scatter(virgpca_items["weights"][:,i],virgpca_items["weights"][:,j],s = 0.5,label = "Virgin")
            ax[i,j].axis("off")
            ax[i,j].set_title("PC {} vs. PC {}".format(i+1,j+1))
    plt.legend()
    plt.tight_layout()
            
    fig = plt.gcf()
    return fig

@memory.cache
def aggregate_evaluations(damweights,dampca,valid_indices,all_fds):
    all_dists ={} 
    all_ims ={} 
    weights_index = 10
    print(all_fds.keys())
    for weights_index,wi in enumerate(valid_indices):
        contour_reconstructed = data.get_contour_from_pc(dampca,damweights[weights_index:weights_index+1,:])
        fd = all_fds[wi]
        dist,img = fd.evaluate_processed_coefs("dam",contour_reconstructed[0,:,:])
        all_dists[wi] = dist
        all_ims[wi] = img
        print(wi)
    vals = all_dists.values()
    n,bins,patches = plt.hist(vals)
    fig = plt.gcf()
    return fig,all_dists,all_ims

@memory.cache
def plot_outlier_frames(all_ims,all_dists,all_fds):
    all_figs ={} 

    outlier_frames = {ind:all_ims[ind] for ind,dist in all_dists.items() if dist > 50}
    for ind,frame in outlier_frames.items():
        contour = all_fds[ind].contours["dam"]
        ## Just show a box around the actual contours: 
        mins = np.min(contour,axis = 0).astype(int)-20
        mins[mins<0] = 0
        maxs = np.max(contour,axis = 0).astype(int)+20

        xcoords = np.array([mins[0],maxs[0]])
        ycoords = np.array([mins[1],maxs[1]])
        print(xcoords,ycoords)

        fig,ax = plt.subplots()
        plt.imshow(frame[slice(*xcoords),slice(*ycoords)])
        plt.plot(contour[:,1]-mins[1],contour[:,0]-mins[0],label = "original_contour")
        plt.title("Comparison of PCA vs. original contour for outlier frame {} (dist = {})".format(ind,str(all_dists[ind])[:4]))
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        all_figs[ind]=fig
    return all_figs

datapath = os.path.join("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/TempTrial2roi_2cropped_part2DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat")
labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"

if __name__ == "__main__":
    md = initialize_doc({"parent":"summary_week_10_9_20","prev":"get_statistical_shape_model"})
    md.new_header(title = "Summary",level = 1)
    md.new_paragraph("We can use the contours extracted by "+md.new_inline_link(link = "./get_statistical_shape_model.md",text = "the previous file")+" to create a shape model. Let's focus for now on creating shape models for the dam, as this is the most reliable. ")
    all_contours = joblib.load("./script_data/all_contours")
    data = LabeledData(labeled_data,additionalpath)

    ## Assume for now that the first contour is the most representative: 
    contourdict = {i:all_contours[i] for i in range(len(all_contours))}
    ## Remove wonky looking contours for now
    defectlist = [14,15,16,55,80,81,97,99]
    ## Iterate through available contours, and generate normalized fourier representations for them.
    for i in defectlist:
        contourdict.pop(i)

    images = data.get_images(np.arange(101))
    fourierdescriptors = {}
    descriptors = {0:"virgin",1:"dam"}
    pointarray = data.dataarray
    
    all_fds = {}

    for i,contour in contourdict.items():
        points = pointarray[i,:,:,:]
        contourdata = {descriptors[j]:contour[j][0] for j in descriptors}
        fd = FourierDescriptor(images[i],contourdata,points)
        all_fds[i] = fd

    fourierdescriptors = process_contours(all_fds)
    ## Generate a corrected set of fourier components
    fig,reorganized = plot_basic_statistics(all_fds,fourierdescriptors)
    save_and_insert_image(md,fig,"../docs/script_docs/images/mean_shape.png")
    md.new_paragraph("We first calculate the mean shape for both animal contours from the training data. Note how reasonable this mean shape looks despite known issues: the existence of the wand, the fact that we lose part of the virgin's head sometimes. We indicate with black x marks the starting point of the contours. We have to believe that we will only get more refined contours as we keep working with this data. Beyond cleaning up the preprocessing, one interesting possibility would be to perform data augmentation by symmetrizing the original contour dataset. We will revisit this later on in this same document. For now, let us explore the shape space we have generated using principal components analysis.")
    md.new_paragraph("We will now describe the pc space spanned by our training dataset of shapes. We will first construct this pc space, then we will consider the shapes generated at specific loadings.")

    ## Get pca items for the virgin and the dam. 
    n_components = 10
    pcadict = get_pca(n_components,reorganized)


    animal = "dam"
    dampca_items = pcadict[animal]
    dampca = dampca_items["pca"]

    md.new_paragraph("Upon application of PCA, we see that the data seems inherently low dimensional, with the first {} components capturing {} percent of the total data variance for the dam. Below, we show the proportions of the variance in each of these eigenvectors:".format(n_components,str(np.sum(dampca.explained_variance_ratio_)*100)[:5]))
    plt.bar(x = np.arange(n_components),height = dampca.explained_variance_ratio_)
    plt.title("Variance Ratio in each of the top {} components".format(n_components))
    fig = plt.gcf()
    save_and_insert_image(md,fig,"../docs/script_docs/images/dam_variance_ratio.png")
    md.new_line("It is apparent that the data variance is realy concentrated in the first 4-5 eigenvectors.")

    md.new_paragraph("We can further characterize this data by exploring the pc space of shapes. Here, we have depicted a grid of the mean contour once again (center), as well as perturbations in the top four eigenvectors. We have scaled these perturbations by the square root of each corresponding eigenvalue, and depicted shapes at two of these units removed from the mean.")
    edges = np.sqrt(dampca.explained_variance_)

    ## Now generate perturbations in the first four directions:
    fig = plot_perturbations(data,dampca,n_components)

    save_and_insert_image(md,fig,"../docs/script_docs/images/pca_one_unit_exploration.png")

    md.new_paragraph("We can see that the first principal captures leaning to the left or right, with a potential displacement of the centroid as well. The second principal component seems to capture elongation and shrinking. PC 3 and 4 seem to capture these behaviors, but at more extreme deformations. Note that the orthogonality of PCs is applied in the space of fourier descriptors, which may lead to non-orthogonal seeming deformations of the shape. However, all of these shapes appear to capture real deformations of the mouse contour- we can explore this hypothesis further by looking at the distribution of points in PC space. Although it is not conclusive, we can compare the distribution of the dam's positions to those of the virgins in terms of pc weights, and look at the distributions comparatively- we see that the virgin distribution has a much higher proportion of outlier points, that most likely correspond to frames where the wand has not been captured, or there are issues with the headplate. We will examine these noise hypotheses more closely later.")

    virgpca_items = pcadict["virgin"]
    fig = scatterplot_virg_dam_pca(dampca_items,virgpca_items)
    save_and_insert_image(md,fig,"../docs/script_docs/images/plot_pc1_2.png")

    testfd = all_fds[0] 
    md.new_paragraph("If we refit some of the recovered weights to the original images and contours, we can evaluate how well pca based contours recover the true data.")
    damweights = pcadict["dam"]["weights"]
    valid_indices = reorganized["indices"]

    #def aggregate_evaluations(damweights,dampca,valid_indices,all_fds):
    #all_dists ={} 
    #all_ims ={} 
    #weights_index = 10
    #print(all_fds.keys())
    #for weights_index,wi in enumerate(valid_indices):
    #    contour_reconstructed = data.get_contour_from_pc(dampca,damweights[weights_index:weights_index+1,:])
    #    fd = all_fds[wi]
    #    dist,img = fd.evaluate_processed_coefs("dam",contour_reconstructed[0,:,:])
    #    all_dists[wi] = dist
    #    all_ims[wi] = img
    #    print(wi)
    #vals = all_dists.values()
    #n,bins,patches = plt.hist(vals)
    #fig = plt.gcf()
    #return fig,all_dists,all_ims
    fig,all_dists,all_ims = aggregate_evaluations(damweights,dampca,valid_indices,all_fds)

    md.new_paragraph("First, we can evaluate a distance metric between the original contour shape and the new contour shape across the entire training set. The distance used here is closely related to the procrustes distance: a euclidean norm on shape keypoints assuming optimal translation and rotation between the two shapes. In our case, we fix translations and rotations to respect the labeled marker points from DLC, instead of aligning the shapes directly. This can be thought of as a lower bound (?) to the procrustes distance (can you prove this?)")
    save_and_insert_image(md,fig,"../docs/script_docs/images/procrustes_distance_lb.png")
    md.new_paragraph("We see that in general, the distance distribution clusters in the range of 20-40 units, with a few outliers. Let us examine these outlier frames in more detail:")

    
    all_figs = plot_outlier_frames(all_ims,all_dists,all_fds)

    for figind,fig in all_figs.items():
        save_and_insert_image(md,fig,"../docs/script_docs/images/outlier_contour{}.png".format(figind))

    md.new_paragraph("We have plotted here three outlier frames, with the original contour given in blue, and the pca reconstruction providing the silhouette of the actual image. It appears that the PCA reconstruction acts as a smoothing operation, and does not in general remove key features of the original image. We will proceed with PCA reconstruction contours, and build a joint distribution of the contours and the point locations.")


    md.new_paragraph("We will fit a gaussian distribution to the marker points and contours, and observe how well they are able to reconstruct each other.")
    animal = "dam"
    mean,cov = data.get_gaussian_contour_pose_distribution(animal,damweights,all_fds)
    print(mean,np.linalg.eigvals(cov))

    md.new_paragraph("We will first use the points to reconstruct pca weights, and look at the resulting distance metrics in contour space.")
    all_processed_points = np.array([fd.process_points()[:,:,1] for fd in all_fds.values()])
    weights = data.get_MAP_weights(mean,cov,all_processed_points)

    contours = data.get_contour_from_pc(dampca,weights)
    out = all_fds[1].superimpose_centered_contour("dam",contours[1,:,:])
    plt.plot(out[0,:],out[1,:],label = "contour given by conditioning")
    plt.plot(all_fds[1].points[1,:,1],all_fds[1].points[0,:,1],"x",label = "original points")
    plt.plot(all_fds[1].contours["dam"][:,0],all_fds[1].contours["dam"][:,1],label = "original contour")
    plt.legend()
    fig = plt.gcf()
    save_and_insert_image(md,fig,"../docs/script_docs/images/gaussian_conditioning_contour.png")
    md.new_paragraph("We can see that a gaussian model seems to give us pretty reasonable contours. We can measure this quatitatively by comparing the distribution of shape distances, just as we did with the pca reconstructed contours.")
    all_dists = {} 
    all_images = {}
    for f,(fi,fd) in enumerate(all_fds.items()):
        contour = contours[f]
        contour[contour> np.array([[700],[480]])] = np.nan 
        dist,image = fd.evaluate_processed_coefs("dam",contour,image = True)
        all_dists[fi] = dist
        all_images[fi] = image
    plt.hist(all_dists.values())
    fig = plt.gcf()
    save_and_insert_image(md,fig,"../docs/script_docs/images/reconstruction_distance_hists.png")
    md.new_paragraph("We see that the distribution of shape distances has a much longer tail when we condition on part locations.")

    all_figs ={} 

    outlier_frames = {ind:all_images[ind] for ind,dist in all_dists.items() if dist > 100}
    for ind,frame in outlier_frames.items():
        contour = all_fds[ind].contours["dam"]
        ## Just show a box around the actual contours: 
        mins = np.min(contour,axis = 0).astype(int)-20
        mins[mins<0] = 0
        maxs = np.max(contour,axis = 0).astype(int)+20

        xcoords = np.array([mins[0],maxs[0]])
        ycoords = np.array([mins[1],maxs[1]])

        fig,ax = plt.subplots()
        plt.imshow(frame[slice(*xcoords),slice(*ycoords)])
        plt.plot(contour[:,1]-mins[1],contour[:,0]-mins[0],label = "original_contour")
        plt.title("Comparison of PCA vs. original contour for outlier frame {} (dist = {})".format(ind,str(all_dists[ind])[:4]))
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        all_figs[ind]=fig

    for figind,fig in all_figs.items():
        save_and_insert_image(md,fig,"../docs/script_docs/images/outlier_contour_gauss{}.png".format(figind))

    md.new_paragraph("However, we see that even outlier contours maintain a pretty high degree of fidelity to the underlying image- it appears that conditioning on the marker points provides a very reliable reconstruction of the contour. While this is to some degree expected for points that this gaussian was trained on, the resulting contours sometimes look to be more accurate than the original, suggesting there is something about our representation that correctly captures the variation of the data. Areas wehre we see some problems come in capturing very intensely bending contours. Note that this is just the MAP estimate- if we had a good way of incorporating image information, we might be able to bias this towards an even better representation.")
    md.new_paragraph("We have done some proof of concept studies to test the feasibility of fine-tuning these contours to better capture obvious cases where the pca contour is over or underestimatimating the actual mouse (see frame 79). It appears that doing gradient descent on the PCA weights directly is not stable, at least for the objectives that I looked at. However, doing gradient descent on the fourier parametrizations of the contours does work "+md.new_inline_link(text = "(see here)",link = "./test_jax.md")+". I will therefore look at the potential to fine-tune these contours using a cost regularized by the posterior probability given the location of annotated markers.")
    md.new_paragraph("Once I have implemented this fine tuning, I will have a custom-built distribution relating contours to marker points, as well as a mechanism for fine tuning contours directly to the image. The next step is to apply these methods to the raw data, specifically in the analysis of pursuit events. For each pursuit event, I will use our new distribution to first detect problem frames, and then correct these problem frames using information from neighboring frames (initializing point reassignment from neighboring frames, for example.)")


    md.create_md_file()
