## What happens if we represent our movement data in a hierarchical polar coordinate set? Does this make analysis/denoising easier?
import os 
import numpy as np
from scipy.io import loadmat
from script_doc_utils import initialize_doc,insert_image,save_and_insert_image,get_relative_image_path
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

if __name__ == "__main__":
    md = initialize_doc()
    md.new_header(level = 1,title = "Summary")
    md.new_paragraph("This script explains idea for a new way of representing our dataset using a hierarchical polar representation. This idea is in part inspired by the literature from robotics (think of linkage models). The representation works as follows: First, take the average of all detected mouse body parts. This should be a point between the body parts of both animals. We will represent this average point in polar coordinates. Now, the positions of individual body parts can be recovered relative to this average point, again in polar coordinates. See below, where we give the representation of the centroid coordinates only:")
    data = loadmat(os.path.join("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/TempTrial2roi_2cropped_part2DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat"))
    start = 15000
    end = 15500
    vtraj_sample = data["virgin_centroid_traj"][start:end]
    dtraj_sample = data["dam_centroid_traj"][start:end]
    mean_sample = (vtraj_sample+dtraj_sample)/2
    origin = np.array([300,300])
    l1 = np.stack([origin,mean_sample[-1,:]],axis = 0)
    l2 = np.stack([vtraj_sample[-1,:],dtraj_sample[-1,:]])
    ## Plot trajectories
    plt.plot(vtraj_sample[:,0],vtraj_sample[:,1],color = "blue",label = "virgin")
    plt.plot(dtraj_sample[:,0],dtraj_sample[:,1],color = "red",label = "dam")
    plt.plot(mean_sample[:,0],mean_sample[:,1],color = "black",label = "average")
    ## Plot endpoints
    plt.plot(vtraj_sample[-1,0],vtraj_sample[-1,1],color = "blue",marker = "o")
    plt.plot(dtraj_sample[-1,0],dtraj_sample[-1,1],color = "red",marker = "o")
    plt.plot(mean_sample[-1,0],mean_sample[-1,1],color = "black",marker = "o")
    ## Plot L1
    plt.plot(l1[:,0],l1[:,1],color = "black",linestyle = "--")
    plt.text(260,290,r"($l_1,\theta_1$)")
    ## Plot L2
    plt.plot(l2[:,0],l2[:,1],color = "black",linestyle = ":")
    plt.text(220,100,r"($l_2,\theta_2$)")

    plt.ylim([300,0])
    plt.xlim([0,300])
    plt.title("Example Centroid Traces")
    plt.legend()
    fig = plt.gcf()
    save_and_insert_image(md,fig,path="../docs/script_docs/images/examplecentroidhierarchicalpolar.png",align = "center")
    md.new_paragraph("Here, the coordinates $\l_1,\theta_1$ give the location of the mean, while $l_2,\theta_2$ give the location of the dam and virgin positions, respectively. The value of this representation is that it cleanly separates the representation of the two mice, or it should in the case where detections are correctly assigned. Let's see what happens when we examine $\theta_2$ for all coordinates, not just the centroid, as a function of time:" )
    ## First, get all virgin and dam coordinates:
    vdatakeys = ["virgin_tip_traj","virgin_centroid_traj","virgin_leftear_traj","virgin_rightear_traj","virgin_tailbase_traj"]
    ddatakeys = ["dam_tip_traj","dam_centroid_traj","dam_leftear_traj","dam_rightear_traj","dam_tailbase_traj"]
    vdata = [data[vd] for vd in vdatakeys]
    ddata = [data[dd] for dd in ddatakeys]
    ## Now concatenate appropriately:
    all_data = np.stack([*vdata,*ddata],axis = 2)
    avg = np.mean(all_data,axis = 2)
    vcentered = all_data[:,:,:5] - avg[:,:,None]
    dcentered = all_data[:,:,5:] - avg[:,:,None]
    ## Cast to complex: 
    vc = vcentered[:,0]+1j*vcentered[:,1]
    dc = dcentered[:,0]+1j*dcentered[:,1]
    labels = ["tip","centroid","left ear","right ear","tailbase"]
    [plt.plot(np.angle(dc[:,i]),"o",markersize = 0.5,label = labels[i]) for i in range(vc.shape[-1])]
    plt.title(r"$\theta_2$ for Dam's body parts")
    plt.legend(loc = "center right")
    plt.xlabel("frame")
    plt.ylabel(r"$\theta_2$ (radians)")
    fig = plt.gcf()
    impath = "../docs/script_docs/images/theta2_representation_dam.png"
    save_and_insert_image(md,fig,path = impath,align = "center")

    md.new_paragraph("Seeing $\theta_2$ coordinates for the dam's body parts reveals points in time when the body part angles are fairly close together and smooth, and others when they are very discontinuous. In this representation, it is difficult to tell when the discontinuities are from the periodicity of the representation and when they are from the data: see below for a colormap of both mice, which preserves periodicity information.")
            
    all_angles = np.concatenate([np.angle(vc),np.angle(dc)],axis = 1)
    plt.imshow(all_angles[10000:20000,:],cmap = "twilight",aspect = 0.002)
    plt.colorbar()
    plt.title(r"\theta_2 for both pairs of mice over first 10000 time frames.")
    fig = plt.gcf()
    save_and_insert_image(md,fig,path = "../docs/script_docs/images/full_theta2.png",align = "center")
    md.new_paragraph("A good next step would be to characterize the clustering of these angles through time. We can come up with a criterion for when we are correctly tracking two animals at all, and when we degenerate to tracking the body parts on just one. From this, it might be possible to reconstruct trajectories at various different levels.")
    md.new_header(title = "Error detection and Reconstruction",level = 2)
    md.new_paragraph("We need a good policy to handle errors in the raw DLC tracking data. This policy should distinguish error cases where we have switches from those where we have completely lost the position of an animal, and treat them differently. Let's first come up with a classifier that determines when traces are clean, when there are switches, and when an animal is wholly missing.")
    plt.hist(all_angles[0,:])
    plt.show()

    md.create_md_file()
