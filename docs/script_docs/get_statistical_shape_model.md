
Script documentation for file: get_statistical_shape_model, Updated on:2020-10-23 18:49:41.176273
=================================================================================================
 
  
**parent file: [summary_week_10_9_20](./summary_week_10_9_20.md)**  
**next file: [from_contours_to_shape_model](./from_contours_to_shape_model.md)**
# Summary


We will incorporate the training data into our work through the use of a statistical shape model: a model of the contours of an object, encoded low dimensionally through fourier features. It is known that a small proportion of fourier features is sufficient to encode many contours that we may be interested in. Once we have a collection of fourier features per set of object edges, which can then be manipulated to introduce various invariances. We can then look at the covariance of different shape parameters, and create a low dimensional, statistical representation of possible object shapes. Shape models are popular in biomedical imaging data, and even old approaches have shown that it is possible to create shape models that are well constrined by both the underlying image, and user defined boundary points (Neumann et al. 1998). We will apply this framework to our DLC training data, automatically detecting animal contours with python computer vision, and build a model that is also informed by our markers.

We start off with some computer vision. We would like to segment out the positions of our two mice, so that we can then construct contours of their positions. One effective way to do this is with the watershed algorithm, which mimics 'flooding' from a set of user-defined marker points, and considers the resulting basins as contiguous objects. This setup gives us a nice way to plug in our DLC training markers into an image segmentation problem: we define a skeleton of points from our markers that certainly belong to a given animal, and apply the watershed algorithm from there. For our purposes, the marker skeleton looks like this:   
<img src="./images/linked_skeleton.png" />

Now we apply the methods of watershed segmentation to our images. First we will threshold our image to create a binary mask. Then we will remove small holes and dots with morphological opening and closing. Then we will apply the watershed transform, with the skeleton shown above on the distance transform of the binary image. This procedure effectively separates the animals from the background, and from each other in most cases. For more info on the watershed transform, see [here.](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html)  
<img src="./images/pipeline_intermediate_steps/image_preprocessing0.png" />  
<img src="./images/pipeline_intermediate_steps/image_preprocessing1.png" />  
<img src="./images/pipeline_intermediate_steps/image_preprocessing2.png" />  
<img src="./images/pipeline_intermediate_steps/image_preprocessing3.png" />  
<img src="./images/pipeline_intermediate_steps/image_preprocessing4.png" />  
<img src="./images/pipeline_intermediate_steps/image_preprocessing5.png" />  
<img src="./images/pipeline_intermediate_steps/image_preprocessing6.png" />  
<img src="./images/pipeline_intermediate_steps/image_preprocessing7.png" />

We achieve good performance on most frames in the training set, but we can further refine the results by manually indicating lines indicating lines in the training frames that beling to one animal or another, and adding these to the skeleton. This is achieved via the `LabeledData.save_auxpoints` function, which opens up a bare-bones gui for the user to label. We can see the value of this in an example frame. Before interactive segmentation:  
<img src="./images/test_LabeledData_segment_animals.png" />

Compare to after interactive segmentation (see different markers in bottom left)  
<img src="./images/test_LabeledData_separate_animals_interactive.png" />

We can further improve the quality of detected points by applying a median filter:

It's also possible to remove the hole in the head by applying a gaussian convoluion and rethresholding. I'm still thinking about the best way to merge across the head barrier/merge unconnected contours when they occur.  
<img src="./images/smoothed_images.png" />

By default, we blur the reference image with a sigma = 0.7, and apply yen thresholding afterwards.

Finally, we can extract the contours for these images:

This is a big point, so we will show all of the contours that we have extracted so far.  
<img src="./images/labeled_contours/contourframe0.png" />  
<img src="./images/labeled_contours/contourframe1.png" />  
<img src="./images/labeled_contours/contourframe2.png" />  
<img src="./images/labeled_contours/contourframe3.png" />  
<img src="./images/labeled_contours/contourframe4.png" />  
<img src="./images/labeled_contours/contourframe5.png" />  
<img src="./images/labeled_contours/contourframe6.png" />  
<img src="./images/labeled_contours/contourframe7.png" />  
<img src="./images/labeled_contours/contourframe8.png" />  
<img src="./images/labeled_contours/contourframe9.png" />  
<img src="./images/labeled_contours/contourframe10.png" />  
<img src="./images/labeled_contours/contourframe11.png" />  
<img src="./images/labeled_contours/contourframe12.png" />  
<img src="./images/labeled_contours/contourframe13.png" />  
<img src="./images/labeled_contours/contourframe14.png" />  
<img src="./images/labeled_contours/contourframe15.png" />  
<img src="./images/labeled_contours/contourframe16.png" />  
<img src="./images/labeled_contours/contourframe17.png" />  
<img src="./images/labeled_contours/contourframe18.png" />  
<img src="./images/labeled_contours/contourframe19.png" />  
<img src="./images/labeled_contours/contourframe20.png" />  
<img src="./images/labeled_contours/contourframe21.png" />  
<img src="./images/labeled_contours/contourframe22.png" />  
<img src="./images/labeled_contours/contourframe23.png" />  
<img src="./images/labeled_contours/contourframe24.png" />  
<img src="./images/labeled_contours/contourframe25.png" />  
<img src="./images/labeled_contours/contourframe26.png" />  
<img src="./images/labeled_contours/contourframe27.png" />  
<img src="./images/labeled_contours/contourframe28.png" />  
<img src="./images/labeled_contours/contourframe29.png" />  
<img src="./images/labeled_contours/contourframe30.png" />  
<img src="./images/labeled_contours/contourframe31.png" />  
<img src="./images/labeled_contours/contourframe32.png" />  
<img src="./images/labeled_contours/contourframe33.png" />  
<img src="./images/labeled_contours/contourframe34.png" />  
<img src="./images/labeled_contours/contourframe35.png" />  
<img src="./images/labeled_contours/contourframe36.png" />  
<img src="./images/labeled_contours/contourframe37.png" />  
<img src="./images/labeled_contours/contourframe38.png" />  
<img src="./images/labeled_contours/contourframe39.png" />  
<img src="./images/labeled_contours/contourframe40.png" />  
<img src="./images/labeled_contours/contourframe41.png" />  
<img src="./images/labeled_contours/contourframe42.png" />  
<img src="./images/labeled_contours/contourframe43.png" />  
<img src="./images/labeled_contours/contourframe44.png" />  
<img src="./images/labeled_contours/contourframe45.png" />  
<img src="./images/labeled_contours/contourframe46.png" />  
<img src="./images/labeled_contours/contourframe47.png" />  
<img src="./images/labeled_contours/contourframe48.png" />  
<img src="./images/labeled_contours/contourframe49.png" />  
<img src="./images/labeled_contours/contourframe50.png" />  
<img src="./images/labeled_contours/contourframe51.png" />  
<img src="./images/labeled_contours/contourframe52.png" />  
<img src="./images/labeled_contours/contourframe53.png" />  
<img src="./images/labeled_contours/contourframe54.png" />  
<img src="./images/labeled_contours/contourframe55.png" />  
<img src="./images/labeled_contours/contourframe56.png" />  
<img src="./images/labeled_contours/contourframe57.png" />  
<img src="./images/labeled_contours/contourframe58.png" />  
<img src="./images/labeled_contours/contourframe59.png" />  
<img src="./images/labeled_contours/contourframe60.png" />  
<img src="./images/labeled_contours/contourframe61.png" />  
<img src="./images/labeled_contours/contourframe62.png" />  
<img src="./images/labeled_contours/contourframe63.png" />  
<img src="./images/labeled_contours/contourframe64.png" />  
<img src="./images/labeled_contours/contourframe65.png" />  
<img src="./images/labeled_contours/contourframe66.png" />  
<img src="./images/labeled_contours/contourframe67.png" />  
<img src="./images/labeled_contours/contourframe68.png" />  
<img src="./images/labeled_contours/contourframe69.png" />  
<img src="./images/labeled_contours/contourframe70.png" />  
<img src="./images/labeled_contours/contourframe71.png" />  
<img src="./images/labeled_contours/contourframe72.png" />  
<img src="./images/labeled_contours/contourframe73.png" />  
<img src="./images/labeled_contours/contourframe74.png" />  
<img src="./images/labeled_contours/contourframe75.png" />  
<img src="./images/labeled_contours/contourframe76.png" />  
<img src="./images/labeled_contours/contourframe77.png" />  
<img src="./images/labeled_contours/contourframe78.png" />  
<img src="./images/labeled_contours/contourframe79.png" />  
<img src="./images/labeled_contours/contourframe80.png" />  
<img src="./images/labeled_contours/contourframe81.png" />  
<img src="./images/labeled_contours/contourframe82.png" />  
<img src="./images/labeled_contours/contourframe83.png" />  
<img src="./images/labeled_contours/contourframe84.png" />  
<img src="./images/labeled_contours/contourframe85.png" />  
<img src="./images/labeled_contours/contourframe86.png" />  
<img src="./images/labeled_contours/contourframe87.png" />  
<img src="./images/labeled_contours/contourframe88.png" />  
<img src="./images/labeled_contours/contourframe89.png" />  
<img src="./images/labeled_contours/contourframe90.png" />  
<img src="./images/labeled_contours/contourframe91.png" />  
<img src="./images/labeled_contours/contourframe92.png" />  
<img src="./images/labeled_contours/contourframe93.png" />  
<img src="./images/labeled_contours/contourframe94.png" />  
<img src="./images/labeled_contours/contourframe95.png" />  
<img src="./images/labeled_contours/contourframe96.png" />  
<img src="./images/labeled_contours/contourframe97.png" />  
<img src="./images/labeled_contours/contourframe98.png" />  
<img src="./images/labeled_contours/contourframe99.png" />  
<img src="./images/labeled_contours/contourframe100.png" />

We will take these contours, and create fourier descriptors out of them next in the file [./from_contours_to_shape_model.md](./from_contours_to_shape_model.md)