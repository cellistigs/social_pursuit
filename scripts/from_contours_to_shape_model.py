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

datapath = os.path.join("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/TempTrial2roi_2cropped_part2DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat")
labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"

if __name__ == "__main__":
    md = initialize_doc({"parent":"summary_week_10_9_20","prev":"from_contours_to_shape_model"})
    md.new_header(title = "Summary",level = 1)
    md.new_paragraph("We can use the contours extracted by "+md.new_inline_link(link = "./get_statistical_shape_model.md`",text = "the previous file")+" to create a shape model. Let's focus for now on creating shape models for the dam, as this is the most reliable. ")
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
    save_and_insert_image(md,fig,"../docs/script_docs/images/pca_one_unit_exploration.png")

    md.new_paragraph("We can see that the first principal captures leaning to the left or right, with a potential displacement of the centroid as well. The second principal component seems to capture elongation and shrinking. PC 3 and 4 seem to capture these behaviors, but at more extreme deformations. Note that the orthogonality of PCs is applied in the space of fourier descriptors, which may lead to non-orthogonal seeming deformations of the shape. However, all of these shapes appear to capture real deformations of the mouse contour- we can explore this hypothesis further by looking at the distribution of points in PC space. Although it is not conclusive, we can compare the distribution of the dam's positions to those of the virgins in terms of pc weights, and look at the distributions comparatively- we see that the virgin distribution has a much higher proportion of outlier points, that most likely correspond to frames where the wand has not been captured, or there are issues with the headplate. We will examine these noise hypotheses more closely later.")
    fig,ax = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            ax[i,j].scatter(dampca_items["weights"][:,i],dampca_items["weights"][:,j],s = 0.5,label = "Dam")
            ax[i,j].scatter(pcadict["virgin"]["weights"][:,i],pcadict["virgin"]["weights"][:,j],s = 0.5,label = "Virgin")
            ax[i,j].axis("off")
            ax[i,j].set_title("PC {} vs. PC {}".format(i+1,j+1))
    plt.legend()
    plt.tight_layout()
            
    fig = plt.gcf()
    save_and_insert_image(md,fig,"../docs/script_docs/images/plot_pc1_2.png")

    testfd = all_fds[0] 

    print(testfd.mouseangles,testfd.points[:,3,:])


    md.create_md_file()
