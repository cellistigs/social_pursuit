## Script to generate figure 1 of a cosyne abstract. 
import numpy as np
from script_doc_utils import *
from social_pursuit.labeled import LabeledData,FourierDescriptor
import matplotlib.pyplot as plt
import joblib

datapath = os.path.join("/Volumes/TOSHIBA EXT STO/RTTemp_Traces/TempTrial2roi_2cropped_part2DeepCut_resnet50_social_carceaAug29shuffle1_1030000processed.mat")
labeled_data = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/CollectedData_Taiga.h5"
additionalpath = "/Volumes/TOSHIBA EXT STO/Video_Pipelining/training_data/"

if __name__ == "__main__":
    md = initialize_doc({"prev":"refine_contours"})
    md.new_header(title = "Summary",level = 1)
    md.new_paragraph("This file will be used to generate figure 1 for our cosyne abstract to the extent that it can be doen through matplotlib. Figure 1 will be a description of the image processing and statistical pipeline that we used to get to this point.")
    md.new_header(title = "Plot 1: Train Frame to Distribution",level = 2)
    data = LabeledData(labeled_data,additionalpath)

    frameind  = 40
    frameinds = [frameind]
    pose = data.dataarray[frameind]
    image = data.get_images(frameinds)[0]
    cropdims = {"x":(0,150),"y":(0,100)}

    out = data.binarize_frames(frameinds)
    clean = data.clean_binary(out)

    labeled,markers = data.segment_animals(out[0],pose[:,:,0],pose[:,:,1])


    smoothed = data.smooth_segmentation(labeled)

    contours = data.get_contour_from_seg(smoothed)

    ## Now get the fourier descriptor for the dam:
    fd = FourierDescriptor(image,{"virgin":contours[0][0],"dam":contours[1][0]},pose)

    contour_fft = fd.process_contour()
    xcoefs = contour_fft["dam"]["xcoefs"][:257]
    ycoefs = contour_fft["dam"]["ycoefs"][:257]
    all_coefs = np.stack([xcoefs,ycoefs],axis = 0)
    all_coefs[:,0] = 0
    contour_reconstruct = fd.superimpose_centered_contour("dam",np.fft.irfft(all_coefs,axis = 1))

    len_series = 3
    ovals = {}
    for n in range(len_series):
        coefs = all_coefs.copy()
        coefs[:,:n+1] = 0
        coefs[:,n+2:] = 0
        oval = fd.superimpose_centered_contour("dam",np.fft.irfft(coefs,axis = 1))
        ovals[n] = oval

    all_contours = joblib.load("./script_data/all_contours")
    data = LabeledData(labeled_data,additionalpath)

    ## Assume for now that the first contour is the most representative: 
    all_fds = {}
    fourierdescriptors = {}
    contourdict = {i:all_contours[i] for i in range(len(all_contours))}
    descriptors = {0:"virgin",1:"dam"}
    for i,contour in contourdict.items():
        points = data.dataarray[i,:,:,:]
        contourdata = {descriptors[j]:contour[j][0] for j in descriptors}
        fd = FourierDescriptor(data.get_images([i])[0],contourdata,points)
        all_fds[i] = fd
        fourierdescriptors[i] = fd.process_contour()

    reorganized = data.organize_fourierdicts(fourierdescriptors) 
    n_components = 10
    pcadict = data.get_pca(reorganized,n_components)

    indexvec = np.arange(n_components)
    onehots = [indexvec == i for i in range(4)]
    edges = np.sqrt(pcadict["dam"]["pca"].explained_variance_)
    perturbation_vecs = [2*oh*i*edges for oh in onehots for i in [-1,1]]
    perturbation_vecs.append(np.zeros(n_components,))
    perturbation_vecs = np.array(perturbation_vecs)

    assert perturbation_vecs.shape == (9,n_components)
    damvecs = data.get_contour_from_pc(pcadict["dam"]["pca"],perturbation_vecs)


    #fig,ax = plt.subplots(1,4,figsize = (20,4))
    fig = plt.figure(constrained_layout = True,figsize = (20,14))
    gs = fig.add_gridspec(2,6)
    ax0 = fig.add_subplot(gs[0,:2])
    ax1 = fig.add_subplot(gs[1,:2])
    ax2 = fig.add_subplot(gs[0,2:4])
    ax3 = fig.add_subplot(gs[1,2:4])
    ax4 = fig.add_subplot(gs[:,4:])


    ax0.imshow(image[slice(*cropdims["y"]),slice(*cropdims["x"])])
    ax0.plot(*pose,"x")
    ax0.axis("off")
    ax0.set_title("Labeled Training Frame")


    ax1.imshow(labeled[slice(*cropdims["y"]),slice(*cropdims["x"])])
    ax1.axis("off")
    ax1.set_title("Watershed Segmentation")

    ax2.plot(contour_reconstruct[1,:],contour_reconstruct[0,:]) 
    intervals = [1,1,0.85]
    for o,oval in ovals.items():
        offset = np.array([[0,50+intervals[o]*(o+1)*50]]).T
        offset_image = oval+offset
        ax2.plot(offset_image[1,:],offset_image[0,:])
    ax2.plot(23,33,marker = "$F($",markersize = 30,color = "black")
    ax2.plot(110,33,marker = "$)$",markersize = 30,color = "black")
    ax2.plot(120,33,marker = "$=$",markersize = 10,color = "black")
    ax2.plot(200,33,marker = "$+$",markersize = 10,color = "black")
    ax2.plot(230,33,marker = "$+$",markersize = 10,color = "black")
    ax2.plot(260,33,marker = "$+$",markersize = 10,color = "black")
    ax2.plot(280,33,marker = "$\dots$",markersize = 10,color = "black")
    ax2.set_title("Fourier Representation of Contours")
        
    print(contour_fft["dam"]["xcoefs"].shape)
    ax2.set_ylim([0,170])
    ax2.set_xlim(0,300)
    ax2.invert_yaxis()
    ax2.axis("equal")
    ax2.axis("off")
    ax2.annotate('$f_{dam}(t)$', xy=(0.72, 0.32), xytext=(0.72, 0.25), xycoords='axes fraction',
            fontsize=15, ha='center', va='bottom',
            bbox=dict(boxstyle=None, fc='white',edgecolor = 'none'),
            arrowprops=dict(arrowstyle='-[, widthB=11.0, lengthB=3.0', lw=2.0))

    ax3.set_title("PCA of $f_{animal}$")
    ax3.plot(*damvecs[0,:,:]+np.array([[-120,0]]).T)
    ax3.plot(*damvecs[1,:,:]+np.array([[120,0]]).T)
    ax3.plot(*damvecs[2,:,:]+np.array([[0,-120]]).T)
    ax3.plot(*damvecs[3,:,:]+np.array([[0,120]]).T)
    ax3.plot(*damvecs[4,:,:]+np.array([[-120,-120]]).T)
    ax3.plot(*damvecs[5,:,:]+np.array([[120,120]]).T)
    ax3.plot(*damvecs[6,:,:]+np.array([[120,-120]]).T)
    ax3.plot(*damvecs[7,:,:]+np.array([[-120,120]]).T)
    ax3.plot(*damvecs[8,:,:])
    ax3.axvline(x = -60,color = "black")
    ax3.axvline(x = 60,color = "black")
    ax3.axhline(y = -60,color = "black")
    ax3.axhline(y = 60,color = "black")
    ax3.axis("equal")
    ax3.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    
    contours = data.get_contour_from_pc(pcadict["dam"]["pca"],pcadict["dam"]["weights"])
    
    
    ax4.plot(all_fds[0].process_points()[1,:,1],all_fds[0].process_points()[0,:,1])
    ax4.plot(*contours[0])
    ax4.plot(all_fds[0].process_points()[1,:,1],all_fds[0].process_points()[0,:,1])
    ax4.plot(*contours[0])

    animal = "virgin"

    fd_info = joblib.load("./trained_contours/{}_trained_contour_frame{}_it{}".format(animal,frameind,2000))
    print(fd_info.keys())
    trainedcontour = fd_info["trained_contour_fft"]
    #ax[3].plot(*np.fft.irfft(trainedcontour))
    contour = fd_info["fd"].contours[animal]
    fig = plt.gcf()

    save_and_insert_image(md,fig,"../docs/script_docs/images/cosyne_fig_1.png")

    md.create_md_file()


