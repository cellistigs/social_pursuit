### Make a statistical shape model for your training data. 
import os 
import datetime
import numpy as np
from social_pursuit.data import Polar,PursuitVideo,mkdir_notexists
from social_pursuit.labeled import LabeledData
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

datapath = os.path.join("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/TempTrial2roi_2cropped_part2DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat")
labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"

@memory.cache 
def show_pipeline_results(md,labeleddata,frameinds,auxpoints = None):
    folder = "../docs/script_docs/images/pipeline_intermediate_steps"
    mkdir_notexists(folder)
    all_labeled = []
    all_markers = []
    for selectid,frameselect in enumerate(frameinds):
        fig,ax = plt.subplots(2,2,figsize = (20,20))
        ax[0,0].imshow(labeleddata.get_images(frameinds)[selectid])
        out = labeleddata.binarize_frames(frameinds)
        clean = labeleddata.clean_binary(out)
        virgin_labeleddata = labeleddata.dataarray[frameselect,:,:,0]
        dam_labeleddata = labeleddata.dataarray[frameselect,:,:,1]
        dcent = labeleddata.get_positions(np.array([frameselect]),8)
        vcent = labeleddata.get_positions(np.array([frameselect]),3)
        ax[0,1].imshow(clean[selectid])
        ax[0,1].plot(dcent[:,0],dcent[:,1],"x")
        ax[0,1].plot(vcent[:,0],vcent[:,1],"x")
        if auxpoints is None:
            labeled,markers = labeleddata.segment_animals(out[selectid],virgin_labeleddata,dam_labeleddata)
        else:
            labeled,markers = labeleddata.segment_animals(out[selectid],virgin_labeleddata,dam_labeleddata,auxpoints = auxpoints[frameselect])
        ax[1,0].imshow(markers)
        ax[1,1].imshow(labeled)
        ax[0,0].set_title("Original image at frame {}".format(selectid))
        ax[0,1].set_title("Cleaned binarized image")
        ax[1,0].set_title("Markers for watershed.")
        ax[1,1].set_title("Segmented Image.")
        save_and_insert_image(md,fig,path = os.path.join(folder,"image_preprocessing{}.png".format(selectid)))
        all_labeled.append(labeled)
        all_markers.append(markers)
    return all_labeled,all_markers

@memory.cache
def extract_and_save_contours(md,labeleddata,auxfolderdict,dryrun = True):
    all_contours = labeleddata.get_contour(np.arange(101),auxfolderdict)
    folder = "../docs/script_docs/images/labeled_contours"
    mkdir_notexists(folder)
    for i,image_contours in enumerate(all_contours):
        fig,ax = plt.subplots()
        #ax[0].plot(image_contours[0][0][:,1],image_contours[0][0][:,0],"b")
        #ax[1].plot(image_contours[1][0][:,1],image_contours[1][0][:,0],"r")
        plt.imshow(labeleddata.get_images([i])[0])
        color = ["b","r"]
        for m in range(2):
            mcontour = image_contours[m]
            for mc in mcontour:
                plt.plot(mc[:,1],mc[:,0],color[m])

        #plt.plot(image_contours[0][0][:,1],image_contours[0][0][:,0],"b")
        #plt.plot(image_contours[1][0][:,1],image_contours[1][0][:,0],"r")
        
        plt.title("Both contours at frame {}".format(i))
        plt.gca().invert_yaxis()
        save_and_insert_image(md,fig,path = os.path.join(folder,"contourframe{}.png".format(i)))

    return all_contours

def refine_dictpoints(refinedict,data,dryrun = False):
    if dryrun == True:
        return
    else:
        foldername = os.path.join(data.additionalpath,"statisticalshapemodelprimary{}".format(datetime.datetime.now()))
        data.save_auxpoints(refinedict,foldername)

if __name__ == "__main__":
    md = initialize_doc()
    md.new_header(title = "Summary",level = 1)
    md.new_paragraph("We will incorporate the training data into our work through the use of a statistical shape model: a model of the contours of an object, encoded low dimensionally through fourier features. It is known that a small proportion of fourier features is sufficient to encode many contours that we may be interested in. Once we have a collection of fourier features per set of object edges, which can then be manipulated to introduce various invariances. We can then look at the covariance of different shape parameters, and create a low dimensional, statistical representation of possible object shapes. Shape models are popular in biomedical imaging data, and even old approaches have shown that it is possible to create shape models that are well constrined by both the underlying image, and user defined boundary points (Neumann et al. 1998). We will apply this framework to our DLC training data, automatically detecting animal contours with python computer vision, and build a model that is also informed by our markers.")

    md.new_paragraph("We start off with some computer vision. We would like to segment out the positions of our two mice, so that we can then construct contours of their positions. One effective way to do this is with the watershed algorithm, which mimics 'flooding' from a set of user-defined marker points, and considers the resulting basins as contiguous objects. This setup gives us a nice way to plug in our DLC training markers into an image segmentation problem: we define a skeleton of points from our markers that certainly belong to a given animal, and apply the watershed algorithm from there. For our purposes, the marker skeleton looks like this: ")
    data = LabeledData(labeled_data,additionalpath)
    frameinds = [0,40,41,42,83,84,85,98]
    selectid = 0
    poses = data.dataarray[frameinds[selectid],:,:,0].T
    lines = data.link_parts(poses)
    fig,ax = plt.subplots()
    ax.plot(poses[:,0],poses[:,1],"bx",label = "markers")
    ax.plot(lines[:,0],lines[:,1],"rx",label = "skeleton")
    ax.set_title("Markers and constructed skeleton")
    plt.legend()
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/linked_skeleton.png")


    
            
    md.new_paragraph("Now we apply the methods of watershed segmentation to our images. First we will threshold our image to create a binary mask. Then we will remove small holes and dots with morphological opening and closing. Then we will apply the watershed transform, with the skeleton shown above on the distance transform of the binary image. This procedure effectively separates the animals from the background, and from each other in most cases. For more info on the watershed transform, see "+md.new_inline_link(link = "https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html",text = "here."))
    all_labeled,all_markers = show_pipeline_results(md,data,frameinds)
    selectid = 1
    labeled = all_labeled[selectid]
    md.new_paragraph("We achieve good performance on most frames in the training set, but we can further refine the results by manually indicating lines indicating lines in the training frames that beling to one animal or another, and adding these to the skeleton. This is achieved via the `LabeledData.save_auxpoints` function, which opens up a bare-bones gui for the user to label. We can see the value of this in an example frame. Before interactive segmentation:")
    insert_image(md,"./images/test_LabeledData_segment_animals.png")
    md.new_paragraph("Compare to after interactive segmentation (see different markers in bottom left)")
    insert_image(md,"./images/test_LabeledData_separate_animals_interactive.png")

    refinementdict = {
            11:["virgin"],
            17:["virgin"],
            18:["dam"],
            22:["virgin"],
            40:["virgin"],
            42:["dam"],
            46:["virgin"],
            47:["virgin"],
            59:["dam"],
            60:["virgin"],
            74:["virgin"],
            75:["virgin"],
            79:["virgin"],
            80:["virgin"],
            81:["dam"],
            84:["virgin"],
            85:["virgin"],
            86:["virgin"],
            87:["virgin"],
            88:["virgin"],
            97:["dam"]}
    refinefolder = os.path.join(data.additionalpath,"statisticalshapemodelprimary")
    refine_dictpoints(refinementdict,data,dryrun = True)
    ## Get back points.
    auxpoints_trained = data.get_auxpoints({refinefolder:list(refinementdict.keys())})

    #refined_labeled,refined_markers = show_pipeline_results(md,data,list(refinementdict.keys()),auxpoints_trained)
    md.new_paragraph("We can further improve the quality of detected points by applying a median filter:")

    smoothed = data.smooth_segmentation(labeled)
    md.new_paragraph("It's also possible to remove the hole in the head by applying a gaussian convoluion and rethresholding. I'm still thinking about the best way to merge across the head barrier/merge unconnected contours when they occur.")
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(labeled == 1)
    ax[1].imshow(smoothed[:,:,0])
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/smoothed_images.png")
    md.new_paragraph("By default, we blur the reference image with a sigma = 0.7, and apply yen thresholding afterwards.")
    md.new_paragraph("Finally, we can extract the contours for these images:")
    contours = data.get_contour_from_seg(smoothed)
    ## We can now examine all of the contours that we extracted:
    md.new_paragraph("This is a big point, so we will show all of the contours that we have extracted so far.")
    contours = extract_and_save_contours(md,data,auxpoints_trained,dryrun = True)
    joblib.dump(contours,"./all_contours{}".format(datetime.datetime.now()))
    md.new_paragraph("We will take these contours, and create fourier descriptors out of them next in the file "+md.new_inline_link("./from_contours_to_shape_model.py"))

    md.create_md_file()

