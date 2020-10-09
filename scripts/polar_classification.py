## What happens if we represent our movement data in a hierarchical polar coordinate set? Does this make analysis/denoising easier?
import os 
import numpy as np
from social_pursuit.data import Polar,PursuitVideo,mkdir_notexists
from scipy.io import loadmat
from script_doc_utils import initialize_doc,insert_image,save_and_insert_image,get_relative_image_path,insert_vectors_as_table
from joblib import Memory
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

@memory.cache()
def calculate_eigenvalue_spectrum(obj,matrices):
    """Wrapper function for Polar.angular_eig written for caching purposes. 

    """
    return obj.angular_eig(matrices)

@memory.cache()
def plot_cosine_and_spectrum(polar,dists,w,v,frame):
    """Plotting function for cosine similarity matrix and eigenvalue spectrum. 

    """
    fig,ax = plt.subplots(2,1,figsize = (10,15))
    mat = ax[0].matshow(np.cos(dists[frame,:,:]))
    ax[0].set_xticks(np.arange(10))
    ax[0].set_xticklabels("{} {}".format(n,p) for n in polar.nameroots for p in polar.partroots)
    ax[0].set_yticks(np.arange(10))
    ax[0].set_yticklabels("{} {}".format(n,p) for n in polar.nameroots for p in polar.partroots)
    fig.colorbar(mat,ax = ax[0])
    ax[0].set_title("Cosine of distance matrix at frame {}".format(frame),y = 1)
    plt.setp(ax[0].get_xticklabels(), rotation=45)
    ax[1].bar(np.arange(10),w[frame,:])
    ax[1].set_title("Eigenvalue spectrum for cosine of distance matrix at frame {}".format(frame))
    #save_and_insert_image(md,fig,path = "../docs/script_docs/images/distance_and_spectrum_frame{}.png".format(frame),align = "center")
    return fig

@memory.cache
def get_testpoints(polar):
    traj = polar.load_data()
    testpoints = traj[:10,:,:,:]
    return testpoints

@memory.cache
def generate_example_frames(score,scorevec,vid,max_frames):
    zz  = np.where(scorevec == score)
    path = "../docs/script_docs/images/exampleimages_score_{}/".format(score)
    mkdir_notexists(path)
    for i in range(len(zz[0][:max_frames])):
        inspectpoint = zz[0][i]
        frame = vid.get_frame_from_trace_indices("TempTrial2",2,2,inspectpoint)
        vfocus = vcent[inspectpoint,:,:] + avg[inspectpoint,:,None]
        dfocus = dcent[inspectpoint,:,:] + avg[inspectpoint,:,None]

        allfocus = np.concatenate([vfocus,dfocus],axis = 0).astype(int)
        buff = 10
        xlims = (np.min(allfocus[0,:])-buff,np.max(allfocus[0,:])+buff)
        ylims = (np.min(allfocus[1,:])-buff,np.max(allfocus[1,:])+buff)

        plt.imshow(frame)
        plt.plot(vfocus[0,0],vfocus[1,0],"bx")
        plt.plot(dfocus[0,0],dfocus[1,0],"rx")
        plt.plot(vfocus[0,1:3],vfocus[1,1:3],"bo")
        plt.plot(dfocus[0,1:3],dfocus[1,1:3],"ro")
        plt.plot(vfocus[0,3],vfocus[1,3],"b2")
        plt.plot(dfocus[0,3],dfocus[1,3],"r2")
        plt.plot(vfocus[0,4],vfocus[1,4],"bP")
        plt.plot(dfocus[0,4],dfocus[1,4],"rP")
        plt.savefig(os.path.join(path,"frame{}.png".format(inspectpoint)))
        plt.close()
        print(v[inspectpoint,:,-2:])

if __name__ == "__main__":
    md = initialize_doc()
    md.new_header(title = "Classification based on hierarchical polar representation",level =1)
    md.new_paragraph("Describes a multi-step classification process for detecting false/missing body parts. We start with the angle representation we showed in the "+md.new_inline_link(link = "./polar_representation.md",text = "polar representation")+" file:")

    polar = Polar(datapath)
    traj = polar.load_data()
    avg = polar.get_average(traj)
    vcent,dcent = polar.get_centered(traj,avg)
    diff = polar.cast_centered_polar(vcent,dcent)

    angular = diff[:,1,:,:]
    dists = polar.angular_distances(diff[:,1,:,:])
    all_angles=np.concatenate([diff[10000:20000,1,:,0],diff[10000:20000,1,:,1]],axis = 1)

    plt.imshow(all_angles,cmap = "twilight",aspect = 0.002)
    plt.colorbar()
    fig = plt.gcf()
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/angles_recreate.png",align = "center")

    md.new_paragraph(r"The first thing that we will do will be to infer identity assignments based on these angles. If we look at the angles at a representative time point (see a slice of the first figure), they clearly cluster into two distinct groups at most times, indicated by the color of the angle. We want to quantify this clustering in some easy to characterize way. Importantly, we would like to be able to tell when data is not well clustered into two categories without invoking a large computational overhead- this corresponds to the case where all detections actually correspond to a single animal. Although we could enlist a variety of clustering algorithms (gmm,k means, etc), the clustering seems well defined and small enough that these methods would be overkill. To start, consider constructing a distance matrix between the angles for the polar representations of different body parts. The values of this matrix are cos(d$\theta$), where $\theta$ is the cosine of the difference angle between the two detected body parts. Our approach is related to spectral clustering (see bibliography); and we might consider working directly with a Laplacian matrix instead of our current one if we want more clusters/cleaner results. This metric can be interpreted as the covariance between body part vectors projected to the unit circle centered on the mean position.") 

    clean_frame = 10
    w,v = calculate_eigenvalue_spectrum(polar,np.cos(dists)) 
    fig = plot_cosine_and_spectrum(polar,dists,w,v,clean_frame)
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/distance_and_spectrum_frame{}.png".format(clean_frame),align = "center")

    md.new_paragraph("We can see that there is one large eigenvalue, and the rest are of negligible magnitude. Quantitatively, it captures ({p}%) of the total variance. If we examine the corresponding eigenvector, it has values follows: ".format(p = str(100*w[clean_frame,-1]/np.sum(w[clean_frame,:]))[:4]))
    insert_vectors_as_table(md,v[clean_frame,:,-1:],["v_1"])
            
    md.new_paragraph("The largest eigenvector cleanly separates the body parts of the two animals.")
    md.new_paragraph("Now if we choose a frame where the two animals are not cleanly separated, we get the following plots by comparison: ")

    nondisjoint_frame = 4699
    fig = plot_cosine_and_spectrum(polar,dists,w,v,frame=nondisjoint_frame)
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/distance_and_spectrum_frame{}.png".format(nondisjoint_frame),align = "center")
    ndjpercentage = str(100*w[nondisjoint_frame,-1]/np.sum(w[nondisjoint_frame,:]))[:4]

    md.new_paragraph("The corresponding eigenvector correctly assigns the dam's tail trajectory to the virgin animal. The largest eigenvalue still captures ({p}%) of the total variance.".format(p = ndjpercentage))
    insert_vectors_as_table(md,v[nondisjoint_frame,:,-1:],["v_1"])

    md.new_paragraph("Finally, let's consider what happens if we simulate a time point where all of the points correspond to detections on a single animal (see below)")

    testpoints = get_testpoints(polar)
    testpoints[:,:,:,1] = testpoints[:,:,:,0]+np.random.randn(*testpoints.shape[:-1])
    part = 1
    for part in range(5):
        for mouse in range(2):
            plt.plot(testpoints[:,0,part,mouse],testpoints[:,1,part,mouse])
    plt.title("Degenerate pose tracking (simulated)")
    fig = plt.gcf()
    save_and_insert_image(md,fig,"../docs/script_docs/images/degenerate_tracking_example.png",align = "center") 

    degen_frame = 0
    md.new_line("If we apply these same methods to this simulated trajectory, we get the following:")
    avg_traj = polar.get_average(testpoints)
    vcentered,dcentered = polar.get_centered(testpoints,avg_traj)
    diff_sim = polar.cast_centered_polar(vcentered,dcentered)
    dist_sim = polar.angular_distances(diff_sim[:,1,:,:])
    wdg,vdg = calculate_eigenvalue_spectrum(polar,np.cos(dist_sim)) 
    fig = plot_cosine_and_spectrum(polar,dist_sim,wdg,vdg,frame=degen_frame)
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/distance_and_spectrum_frame{}.png".format(degen_frame),align = "center")

    specamp = np.sum(wdg[degen_frame,:])
    top_percent = str(100*wdg[degen_frame,-1]/specamp)[:4]
    next_percent = str(100*wdg[degen_frame,-2]/specamp)[:4]
    fig,ax = plt.subplots(2,1)
    ax[0].plot(vdg[degen_frame,:,-1],vdg[degen_frame,:,-2],"x")
    ax[1].plot(diff_sim[degen_frame,1,:,:].flatten(),"x")

    plt.show()

    md.new_line("We can see that when our detections are degenerate, the distance matrix we constructed does not concentrate its signal as heavily in the largest component. In this case, we get a non-trivial contribution to the second eigenvalue as well. The top two eigenvectors account for {p1}% and {p2}% of the variance, respectively , with the corresponding eigenvectors:".format(p1 = top_percent, p2 = next_percent))
    insert_vectors_as_table(md,np.flip(vdg[degen_frame,:,-2::],axis = 1),["v_1","v_2"])

    md.new_line("Already, we can see that two important quantities for clustering will be the magnitude of the given eigenvalues, and the entries of the corresponding eigenvectors.")
    md.new_line("Many spectral clustering methods are based on the idea of partitioning a dataset based on the values of a given vector. Let's take inspiration from this, and cluster based on the sign of the entries in the top eigenvector.")
    sign_vec = polar.get_classification_features(v)
    fig,ax = plt.subplots(2,1,sharex = True, figsize = (15,15))
    ax[0].set_title("Sample of signed principal eigenvector (top) and eigenvalue spectrum (bottom)",y = 2)
    mat0 = ax[0].matshow(sign_vec[600:1000,:].T,aspect = 20)
    mat1 = ax[1].matshow(w[600:1000,:].T,aspect = 20)
    fig.colorbar(mat0,ax = ax[0])
    fig.colorbar(mat1,ax = ax[1])
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/featurized_criterion.png",align = "center")

    md.new_line("We can use this representation to build an anomaly detector. First, let's detimerine the locations where we have failed to detect an animal. We can detect points like this by constructing a _score_ _vector_ (ones for dam body parts, negative ones for virgin body parts), and looking at the dot product of this score vector with the sign of the top eigenvector of our distance matrix. If this dot product is 10, this means that animal body parts are correctly clustered in the way we would expect in this distance matrix. This procedure asks the question, how much does large scale structure in our distance matrix reflect what we would expect given that they belong to body parts on two different bodies?")
    scorevec = polar.classify_errors(v)
    scorevec_bool = (scorevec == 10)
    plt.matshow(scorevec_bool[600:1000,None],aspect = 0.02)
    fig = plt.gcf()
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/score_vector.png",align = "center")
    percentagewrong = 100*np.sum(scorevec_bool)/len(scorevec_bool)
    md.new_line("Overall, applying this criterion eliminates {p}% of the example trajectory analyzed here from further consideration.".format(p = str(100-percentagewrong)[:4]))
    md.new_paragraph("What we need is a way to exclude those states where we're only tracking one animal.")
    md.new_paragraph("Next, we should confirm that the eliminated points actually check out. construct a within animal distance matrix, and record the statistics of it. Consider that what we're doing here is like 'second level' classification, we could do better by going down a level.") 
    #plt.hist(scorevec,bins = 21)
    #plt.xticks(np.arange(-10,11))
    vid = PursuitVideo("../src/social_pursuit/trace_template_trial2.json")
    score =0 
    max_frames = 20

    generate_example_frames(score,scorevec,vid,max_frames = 20)

    md.new_line("we can see that the inner product of the sign vector with an identity preserving vector gives a good general characterization of the accuracy of detected poses. See "+md.new_inline_link(link = "./images/",text = "here ")+ "for examples of good and bad clustering, in folders marked (exampleimages_score_[score]). Notably, we see that there are edge cases where we would not be able to reliably extract out meaningful pose information from the model, because we can only localize one point to either the virgin or dam. For example, see "+md.new_inline_link(link="./images/exampleimages_score_0/frame1508.png",text = "this frame") +" or "+md.new_inline_link(link="./images/exampleimages_score_0/frame1509.png",text = "this frame")+ " or "+ md.new_inline_link(link = "./images/exampleimages_score_0/frame274.png",text = "this frame."))
    md.new_list(["Concrete Todo Points", ["00), Make a habit of checking your master plan." ,"0) estimate the distribution of part lengths in the score = 10 portion of the dataset.","1) can we estimate the trajectory of individuals accurately, even if we lose some points? To do this, we could estimate the distribution of the mean given all other body parts. Do we need smoothness again for this?", "2) can we reconstruct bodies?"]])
    md.new_paragraph("From the progress we have made so far, we can write down an algorithm for part resolution.")
    md.new_list(["Get mean position from all marked points ","Cluster other body parts based on partition given by mean position","Evaluate cluster formation, and detect outlier frames where marker points are not reliable (0 or 1 detections for a single animal)","For reliable frames, calculate per-identity distance matrices + image contours through conditioning","Do within-animal hypothesis testing on part distribution AND image contours to detect anomalous parts","Evaluate fit of anomalous parts to other animal through distance matrices + image contours.","for outlier frames, search with image contours/rerun DLC on detected contours."],marked_with = "1")
    md.new_paragraph("In this algorithm, steps 1-3 can be thought of as initially resolving the identities of the two animals. Steps 4-6 can be thought of as resolving the features/parts of these detected identities, and potentially revising detected identities based on this fit. Note here that we will use animal contours in addition to detected points, which is a new innovation. If this approach bears out, we may wish to use the image contours as the primary animal representation, instead of the marked points themselves.")


    
    md.create_md_file()
