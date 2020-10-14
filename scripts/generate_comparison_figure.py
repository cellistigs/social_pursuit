## Generate a figure comparing the old way of processing data (centroids) to the new way of processing data (animal contours). 
import os
import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from script_doc_utils import *
from social_pursuit.labeled import LabeledData

labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"

if __name__ == "__main__":
    ld = LabeledData(labeled_data,additionalpath)
    contours = joblib.load("script_data/all_contours")
    md = initialize_doc({"prev":"summary_week_10_9_20"})
    md.new_header(title = "Summary",level = 1)
    md.new_paragraph("Here we're presenting an element-by-element comparison of our existing processing pipeline based on centroids, and a new proposed processing pipeline based on image contours and marked points. I will lay out the advantages and drawbacks of the new proposed method to our data analysis pipeline in a modular way, providing us with concrete 'exit points' from this approach in the event of unforseen delay/difficulty.")
    md.new_header(title = "Related Work and Context",level =2)
    md.new_paragraph("In the literature on recognizing actions from videos, it is thought that the extraction of silhouettes from video is a much easier task than that of localizing key points (Wang et al. 2008), especially in the context of generalization across different movements and settings. It is also shown to produce comparable if not superior results to methods based on key points (Wang et al. 2008, Ling et al. 2007) in downstream action classification at a generally lower computational cost. Recent approaches based on deep nets (El Ghaish et al. 2018) demonstrate that a combination of part-based skeletons, motion trajectories, and body shapes are necessary to achieve state of the art performance on action recognition, both through explicit encoding of body shapes and through silhouette based motion history images.")
    md.new_paragraph("In contrast, the literature on behavior quantification in neuroscience has been dominated by hand labeling approaches (Williamson, 2016), marker points (De Chaumont 2012, Weissbrod 2016, Shemesh 2016, Anpilov 2020), or analyses of the raw image (Burgos-Artizzu 2012, Berman 2014, Wiltschko 2015, Klibaite 2017). Notably, the automated approaches here (marker points and raw image analysis) offer vastly different insights into animal behavior, with marker points yielding simple metrics of chasing, or requiring the use of statistical inference to resolve more intricate categorical behaviors like nose-to-nose or nose-to-tail contact. In contrast, raw image analyses generally yield a high dimensional representation of animal behavior that is post-hoc labeled with unsupervised clustering. The application of these methods to social contexts has been limited to regressing cluster assignments against the kinematic measurements of the two animals relative to one another, potentially in part because these high dimensional representations are difficult to interpret and work with in a social setting. **To my knowledge there has been only one use of animal contours to characterize social behavior, where Thanos et al. 2017 characterized the existence of inter-mouse contacts with body contours**. Thanos et al. 2017 use MoST, a non-deep learning based computer vision approach, to demonstrate that it is possible to track six visually identical animals without markers, and without suffering any identity switching issues. In their analysis, the authors use MoST to characterize points in time when different mice are in contact, a task they note cannot be done simply by tracking marked points. They track the location of each mouse's nose tip, and quantify the position on the other mice that are contacted via the nose tip; note that this same analysis using only marked points requires the use of a classifier on marker points trained on manual annotations (Shemesh 2016). However, Thanos et al. did not characterize the shapes of these body contours or their dynamics beyond contact points: this is a major opportunity for us, where the act of shepherding involves many complex deformations of both mouse shapes. On the technical side, the use of animal contours + marker points to analyze social behavior has not been done, although it is a representation of intermediate complexity between the two extremes of just identity markers, or the raw image. **Body contours offer three distinct advantages to our current, centroid based processing pipeline: 1) The ability to easily quantify the occurence of contacts between animals. 2) The ability to quantify the dynamics of body shape during behaviors of interest. 3) Increased reliability of baseline tracking, espeically with regard to identity switches**. We will leverage these benefits towards the following scientific goals:")

    md.new_header(title = "Proposed benefit for scientific goals", level = 2)
    md.new_paragraph("We have two stated scientific goals in analyzing social behavior data: 1) The detection of stereotyped interactions, and 2) the detection of learning in the interaction between the two animals. We are considering ways to accomplish these goals in the constrained setting of pursuit interactions between the virgin and the dam, in addition to the larger and more general set of interactions that are available between these two animals." )
    md.new_paragraph("I will discuss here the benefits of using animal contours to achieve these goals instead of centroids. Note that there is an 'intermediate' level of movement representation here: the centroid and the other points detected by DLC. However, in general using the full skeleton is less reliable than using the centroid, and any form of full-skeleton analysis will most likely require preprocessing and denoising with the body contour anyway.")
    md.new_header(title= "Subgoal: Analysis of pursuit interactions", level = 3)
    md.new_paragraph("Current workflow:")
    md.new_list(["Extract pursuit interactions with a threshold criterion on the centroid positions and velocities of both mice","Classify interactions into virgin or dam leading based on relative position of centroids","Extract kinematic parameters of pursuits for regression against time (for learning) and clustering (for stereotypy) (i.e, our stated scientific goals)"])
    md.new_paragraph("Point one of this pipeline has beeen characterized against groundtruth labeling data "+md.new_inline_link(link = "./examine_groundtruth_comparisons.md",text = "here.")+" We show that we have high sensitivity to manual scored pursuit events (shepherding and virgin pursuit), but pretty poor selectivity. We could supplement a kinematic detection with body contours, which have a higher baseline correspondence to the information that humans use to make this decision. Point two of this pipeline has not yet been characterized: i.e. the shepherding/virgin pursuit confusion matrix is not known. Anecdotally, it looks like there are some confusing pursuit events for our current criterion (the average relative velocity of both centroid points) that could also benefit from contour analysis. Finally, we have not yet completed step three of this pipeline, as I felt there were still too many unaddressed issues with steps one and two. I would be happy to conduct step three on just the centroid data, following an improved pursuit detection pipeline with contours. However, I suspect that contour methods can also contribute to our stated scientific goals as well.")
    md.new_paragraph("One salient behavioral feature for the study of learning corresponds to changes in shepherding behavior over time. Our collaborators have stated that early on in cohousing, shepherding requires aggressive biting and pushing of the virgin by the dam, while later on it can be triggered by a simple touch (see example frames of contact vs. non-contact below) We already know that contour methods can be used to quantify contacts between animals (Thanos et al. 2017); it would be a simple extension to quantify (average contact duration vs. time) for our animals to support this point qualitatively.")
    contactframes = ld.get_images([79,91])
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(contactframes[0])
    ax[1].imshow(contactframes[1])
    for cix,ci in enumerate([79,91]):
        c = contours[ci]
        ax[cix].plot(c[0][0][:,1],c[0][0][:,0],"b")
        ax[cix].plot(c[1][0][:,1],c[1][0][:,0],"r")
        ax[cix].axis("off")
    ax[0].set_title("no contact event")
    ax[1].set_title("contact event")
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/contactvsnocontactframe.png",align = "center")
    md.new_paragraph("Beyond contact analysis, we would like to identify interesting stereotypy around the pursuit event. A good candidate observed anecdotally is the commonality of 'rotations', where the dam seems to approach the virgin from an angle that rotates her towards the nest (see below). This process involves a deformation of both animals, in close contact with another for an extended amount of time. This deformation could be characterized by considering the shape model descriptors immediately leading up to pursuit events. One concrete implementation would be to consider a few exemplars of this behavior, and look for instances close to it in shape space (as alternatives we could also cluster in shape space, fit an HMM, etc.). By searching for these behaviors, we can measure the proportion of times that they precede known shepherding events, in addition to measuring the other types of shape sequences that can precede a shepherding event. Note that we can find this shape configuration invariant to size, rotation, and location due to the convenience of our fourier shape parametrization. We could also extend this analysis to the entire pursuit event, where it may serve as an interesting way to capture failed pursuits, where the virgin moves to somewhere outside the vicinity of the nest. Note that all of these analyses are not possible with pure centroid based methods, because they do not capture the articulation of either animal in space. Likewise, a skeleton model of all detected points does not yield a representation with the convenient features of our fourier shape description (invariance to rotation, translation; ordering in terms of coarse to fine grained shape description.) As a further direction, we might consider thow the shapes of the dam and virgin contract and expand during interactions, and if we might relate that to a kind of '(soft morphology) interaction physics' as discussed with our collaborators in the past.")
    rotframes = ld.get_images([72,74,77])
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(rotframes[0])
    ax[1].imshow(rotframes[1])
    ax[2].imshow(rotframes[2])
    for cix,ci in enumerate([72,74,77]):
        c = contours[ci]
        ax[cix].plot(c[0][0][:,1],c[0][0][:,0],"b")
        ax[cix].plot(c[1][0][:,1],c[1][0][:,0],"r")
        ax[cix].axis("off")
    ax[0].set_title("Before rotation")
    ax[1].set_title("During rotation")
    ax[2].set_title("After rotation \n (mid-shepherding)")
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/rotationtimecourse.png",align = "center")
    md.new_line("")
    ## Result Metric: Pursuit Detection
    ### Calculate the contours of both animals conditional on the occurence of a pursuit behavior as detected by kinematic criterion. Identify significant patterns in the contour before and during the occurrence of pursuit: one animal circling around the other, squeezing the other (see example frames: 70s, 80s): let's characterize this robustly. It appears there are certain "high energy" configurations of the two animals, which lead to spurts of activity, whether shepherding or not. Construct a joint body contour model of both animals, and identify the joint body shapes that lead to successful vs. failed pursuit events (i.e. the virgin exits a shepherding event to the side).  What are the key indicators of successful shepherding events, prior to shepherding beginning? Additionally, we can robustly detect the occurrence of body contact or not during the actual pursuit (see Thanos et al. 2017)- a point related to the idea that this is a learned behavior that Ioana brought up. If our characterization of body contact is robust, let's analyze this at more detail. Can we distinguish pushing from chasing, and build up a physics of interactions? 

    md.new_header(title = "Subgoal: Analysis of general interactions", level =3)
    md.new_paragraph("Our analysis of general interactions has been limited to low dimensional metrics such as speed made good, and is largely in an exploratory phase. We have also separately discussed with Ioana, Rob and Zahra that interactions in the proximity of food and water sources would be good to monitor, although we have not taken steps towards this. In general, I have been wary of these other analyses until I was more confident in the tracking fidelity of our methods. To that end, improvement of our basic tracking pipeline through contours directly contributes to this line of work. However, just as with specific pursuit interactions, contour based methods can contribute directly to scientific results as well.")
    md.new_paragraph("One thread of the behavior that we have not pulled on is the idea that the dam might display vigilance behavior: a state of heighted awareness towards a potential threat. We have recorded instances where the dam seems to be visually tracking the virgin- although these cases could be localized by cleanly postprocessing the head points of the dam, these cases, and all questions involving the orientation of both animals, could be answered well with a contour model.")
    md.new_paragraph("Another thread here is speed made good. Speed made good is a criterion to measure the motion of two animals in a way that is invariant to their absolute configuration in space. We introduced it as an efficient way to parse the space of possible interactions between these two animals (who is chasing who). We can consider extensions to this metric, but one novel use of contours is a direct query for points in time where the two mice are in a certain configuration. Given a pair of contours (see any of the frames above), one can translate it to the origin, rotate it to some canonical orientation, and scale appropriately, then _query_ the entire behavioral sequence for other similar configurations. This idea of similarity can be considered at different levels of granularity by truncating contours, capturing only coarse or increasingly fine grained similarity in shape configuration. In theory, it is even possible to have the user directly draw two closed curves representing the orientations of the two mice, and use that as a basis for search. It would be interesting to consider extensions of this method to dynamical sequences ('find all pursuits','find all cases where the animals are oriented directly away from one another', 'find all cases where the animals circle each other'), based on raw user input. While we could do this same kind of search with centroid points, I suspect that it would be too unconstrained to specifically isolate sequences of particular scientific interest.")
    ## Result Metric: Speed Made Good 
    ## Fast query: tree structured representations are possible here, making it easy to query 
    ### Speed made good is a criterion to frame the motion of the two animals in a way that is invariant to aboslution configuration of the animals in space. We introduced it as a more general metric than standard pursuit detection, to better characterize the large space of interaction modes between the two animals. A natural extension of speed made good is to the first few fourier modes of a joint body contour model, perhaps with some canonical orientation relative to the nest and centered at mean position between the two animals. This "speed and deformation made good" signal is immediately richer than a purely kinematic one, and more suited for the exploratory nature of the motivating question. We can easily consider more and more "normalized" ways to characterize the motion by taking fewer fourier modes, or even throwing away the phase information in our contour representation entirely to give a characterization of how close together the animals are, and how much space they occupy. 

    md.new_header(title = "Auxiliary Subgoal: Generative Modeling", level = 3)
    md.new_paragraph("Finally, as an auxiliary subgoal, I am very interested in the idea that body contours could be useful for generative modeling, especially in cases with a static background, static camera and moving agents. Recent approaches (Johnander et al. 2019) achived state of the art performance in object segmentation by incorporating a generative appearance model that segmented foreground and background. While they did not go so far as sampling objects themselves, I am curious about the benefit of introducing an explicit object contour model for generative approaches to video data, which are constantly plagued by criticisms of being blurry. In relation to this project, generative modeling could lead to a moseq like approach, or integrate the work that Luhuan has already done with autoregressive models on the centroid data. Generative modeling approaches with amortization could also allow us to deploy a contour + markers pipeline at faster speeds (Johnander et al. 2019) report analysis at 15 fps, while fine-tuning the network in real time.")

    md.new_header(title = "Auxiliary Subgoal: Improvement of overall reliability", level =3)
    md.new_paragraph("As referenced several times in the preceeding text, one of the major roadblocks I have had with processing centroids is the fundamental error rate of our current post-processing pipeline. It has proved difficult to find the right cross-validated hyper parameters to conduct our post processing across different data sets. One obstacle is the fact that with centroid detections, oftentimes it's very unclear if there is an error or not when just examining a single frame of data- useful cues of potential errors always involve looking at preceeding and following frames, or the rest of the skeleton (which can have its own problems). By using body contours, we establish a way to more easily check the validity of our extracted representation at any given point in time, either by 1) looking at the body contours, 2) extracting out the corresponding segmented part of the image, or 3) checking against an appropriae model (like joint gaussians in the markers and body contours). Furthermore, body contours give us a way to fine tune detections with reliable information from the image without going all the way back to DLC, which could save significant time if one is on a laptop and is using NeuroCAAS/the cluster etc. This particular subgoal is intimately related to our current marker postprocessing pipeline, and I will add below the revisions to this pipeline using contour models that I am envisioning.")

    md.new_header(title = "Pipeline",level = 2)
    md.new_paragraph("We now describe our current marker based pipeline and image contour based modifications.")
    ## Image. This is the shared starting point. 
    #```
    md.new_header(title = "Initial Animal Localization", level = 3)
    frame = 10
    image = ld.get_images([frame])[0]
    fig = plt.figure(constrained_layout = True)
    gs = GridSpec(2,2,figure = fig,wspace = 0.4,width_ratios = [0.5,1])

    axsource = fig.add_subplot(gs[:,0])
    axorig = fig.add_subplot(gs[0,1])
    axnew = fig.add_subplot(gs[1,1])
    axes = {"source":axsource,"orig":axorig,"new":axnew}
    [axes[a].axis("off") for a in axes]
    axes["source"].imshow(image)
    axes["source"].set_title("Original Image")

    points = ld.dataarray[frame,:,:,:]
    axes["orig"].plot(*points[:,:,0],"bx")
    axes["orig"].plot(*points[:,:,1],"rx")
    axes["orig"].imshow(image)
    axes["orig"].set_title("Current Output")

    axes["new"].plot(*points[:,:,0],"bx")
    axes["new"].plot(*points[:,:,1],"rx")
    c = contours[10]
    axes["new"].plot(c[0][0][:,1],c[0][0][:,0],"b")
    axes["new"].plot(c[1][0][:,1],c[1][0][:,0],"r")
    axes["new"].set_title("Proposed Output")

    axes["new"].imshow(image)

    ## Create arrows from image to the two initial processing results
    axsourcetr = axes["source"].transData
    axorigtr = axes["orig"].transData
    axnewtr = axes["new"].transData
    figtr = fig.transFigure.inverted()
    ptB = figtr.transform(axsourcetr.transform((280,200))) 
    ptEO = figtr.transform(axorigtr.transform((50,200))) 
    ptEN = figtr.transform(axnewtr.transform((50,200))) 
    ### Create arrow patch
    arroworig = matplotlib.patches.FancyArrowPatch(ptB,ptEO,transform = fig.transFigure,fc = "black",arrowstyle = "-|>,head_width=7,head_length=10")
    arrownew = matplotlib.patches.FancyArrowPatch(ptB,ptEN,transform = fig.transFigure,fc = "black",arrowstyle = "-|>,head_width=7,head_length=10")
    fig.patches.append(arroworig)
    fig.patches.append(arrownew)
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/initial_animal_localization.png",align = "center")

    #```
    

    ## Initial animal localization (Deep Lab Cut) 
    ### Construct a joint model of animal contour and body part. Fit that model to the initial animal localizations.
    md.new_paragraph("Currently, we initially localize animals with the raw output of a DeepLabCut trained model. We are proposing an extension where we supplement this output with a closed body contour around each animal. In order to keep the proposed procedure fast, we will calculate this proposed body contour by constructing a **joint gaussian model of marker positions and fourier descriptors on the labeled training data**. We can then condition this model on the detected markers at any point in time to arrive at a body contour estimate.")
    md.new_paragraph("Pros: In the case where all goes well, we have an additional readout of animal position almost instantaneously.")
    md.new_paragraph("Cons: It's entirely possible that a gaussianity will not be a good assumption for a joint model of markers and body contours. Having to construct and condition on more complicated distributions could significantly slow down the detection process. However, AFAIK a joint model of marker positions and body contours could be a novel and interesting technical contribution in its own right.")

    md.new_header(title = "Anomaly Detection (Body Parts)",level = 3)
    ## Problem Frame (Current)  ## Detection Methods 
    ## Problem Frame (Proposed) ## Above + Contour methods: do the contours overlap?   
    md.new_paragraph()
    ## Anomalous Part Detection  
    ### Evaluate the density of the marker position + body contour. Look for issues like markers outside of body contour, switched markers, big jumps in detected body contour across time.  

    ## Anomalous Part Fill-In
    ### In problem frames, re-fit the body model contour to the image + markers with watershed segmentation, detection of mouselike patches, active contour methods, or projections into the PCA space of contours extracted from the training data. Adjust the markers according to the best estimate given the body model contour. 
    ### Downside: slower. 

    ## Anomalous Identity Detection (Branch and Bound.) 
    ### Do branch and bound reassignment directly on the body model contours, not individual tracked points.
    md.create_md_file()


